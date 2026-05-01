"""Utilities for online-readiness analysis plots and dashboard video frames."""

from __future__ import annotations

import io
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CURVE_STYLES = {
    "plugin": ("With Plugin", "#1f77b4", "-", "o", 2.2, 1.0),
    "ego_only": ("Ego-only", "#2ca02c", "--", "s", 1.7, 0.95),
    "no_plugin": ("No-plugin", "#6f6f6f", "-.", "^", 1.7, 0.95),
}


def load_font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def _to_numpy(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_box_centers_xy(box_tensor) -> np.ndarray:
    arr = _to_numpy(box_tensor)
    if arr is None or arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] >= 4:
        return arr[:, :, :2].mean(axis=1).astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2].astype(np.float32)
    raise ValueError(f"Unexpected box tensor shape for center extraction: {arr.shape}")


def _get_pred_score(infer_result) -> Optional[torch.Tensor]:
    pred_score = infer_result.get("pred_score", infer_result.get("score_tensor"))
    if pred_score is None:
        return None
    return pred_score.detach().cpu() if torch.is_tensor(pred_score) else torch.as_tensor(pred_score)


def filter_pred_boxes_by_score(infer_result, score_thresh: float):
    pred_box = infer_result.get("pred_box_tensor")
    if pred_box is None or score_thresh <= 0:
        return pred_box
    pred_score = _get_pred_score(infer_result)
    if pred_score is None:
        return pred_box
    if pred_score.dim() > 1:
        pred_score = pred_score.max(dim=-1)[0]
    mask = pred_score > score_thresh
    return pred_box[mask] if torch.is_tensor(pred_box) else np.asarray(pred_box)[mask.numpy()]


def summarize_infer_result(infer_result: Dict, pc_range: Sequence[float], score_thresh: float) -> Dict:
    gt_centers = _extract_box_centers_xy(infer_result.get("gt_box_tensor"))
    pred_centers = _extract_box_centers_xy(filter_pred_boxes_by_score(infer_result, score_thresh))
    return {
        "gt_centers": gt_centers,
        "pred_centers": pred_centers,
        "pc_range": [float(v) for v in pc_range],
    }


def _make_square_crop(
    center_xy: np.ndarray,
    crop_size: float,
    pc_range: Sequence[float],
) -> Tuple[float, float, float, float]:
    half = crop_size / 2.0
    xmin_lim, ymin_lim, xmax_lim, ymax_lim = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    cx, cy = float(center_xy[0]), float(center_xy[1])

    xmin = cx - half
    xmax = cx + half
    ymin = cy - half
    ymax = cy + half

    if xmin < xmin_lim:
        xmax += xmin_lim - xmin
        xmin = xmin_lim
    if xmax > xmax_lim:
        xmin -= xmax - xmax_lim
        xmax = xmax_lim
    if ymin < ymin_lim:
        ymax += ymin_lim - ymin
        ymin = ymin_lim
    if ymax > ymax_lim:
        ymin -= ymax - ymax_lim
        ymax = ymax_lim

    xmin = max(xmin_lim, xmin)
    xmax = min(xmax_lim, xmax)
    ymin = max(ymin_lim, ymin)
    ymax = min(ymax_lim, ymax)
    return (xmin, ymin, xmax, ymax)


def _centers_in_crop(centers: np.ndarray, crop_range: Sequence[float]) -> np.ndarray:
    if centers is None or len(centers) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    xmin, ymin, xmax, ymax = crop_range
    mask = (
        (centers[:, 0] >= xmin)
        & (centers[:, 0] <= xmax)
        & (centers[:, 1] >= ymin)
        & (centers[:, 1] <= ymax)
    )
    return centers[mask]


def _match_score(gt_local: np.ndarray, pred_local: np.ndarray, hit_radius: float = 4.0) -> Tuple[float, int]:
    if len(gt_local) == 0:
        return 0.0, 0
    score = 0.0
    hit_count = 0
    for gt_center in gt_local:
        if len(pred_local) == 0:
            continue
        distance = np.linalg.norm(pred_local - gt_center[None, :], axis=1).min()
        if distance < hit_radius:
            hit_count += 1
            score += max(0.0, hit_radius - distance) / hit_radius
    return score, hit_count


def select_auto_crop(
    noplugin_infer: Dict,
    plugin_infer: Dict,
    pc_range: Sequence[float],
    crop_size: float = 32.0,
    score_thresh: float = 0.3,
) -> Tuple[Tuple[float, float, float, float], Dict]:
    noplugin_summary = summarize_infer_result(noplugin_infer, pc_range, score_thresh)
    plugin_summary = summarize_infer_result(plugin_infer, pc_range, score_thresh)

    gt_centers = plugin_summary["gt_centers"]
    plugin_centers = plugin_summary["pred_centers"]
    noplugin_centers = noplugin_summary["pred_centers"]
    pc_range = plugin_summary["pc_range"]

    candidates: List[List[float]] = []
    for centers in (gt_centers, plugin_centers, noplugin_centers):
        if centers is not None and len(centers) > 0:
            candidates.extend(centers.tolist())

    if not candidates:
        center = np.array([(pc_range[0] + pc_range[3]) / 2.0, (pc_range[1] + pc_range[4]) / 2.0])
        return _make_square_crop(center, crop_size, pc_range), {"scene_score": 0.0}

    best_crop = None
    best_score = None
    best_meta = None
    for candidate in candidates:
        crop = _make_square_crop(np.asarray(candidate, dtype=np.float32), crop_size, pc_range)
        gt_local = _centers_in_crop(gt_centers, crop)
        plugin_local = _centers_in_crop(plugin_centers, crop)
        noplugin_local = _centers_in_crop(noplugin_centers, crop)
        plugin_score, plugin_hits = _match_score(gt_local, plugin_local)
        noplugin_score, noplugin_hits = _match_score(gt_local, noplugin_local)

        scene_score = 0.0
        scene_score += 4.0 * max(0, plugin_hits - noplugin_hits)
        scene_score += 2.0 * max(0.0, plugin_score - noplugin_score)
        scene_score += 0.3 * max(0, len(plugin_local) - len(noplugin_local))
        scene_score += 0.15 * len(gt_local)
        packed = (
            scene_score,
            float(len(gt_local)),
            float(plugin_hits - noplugin_hits),
            float(len(plugin_local) - len(noplugin_local)),
        )
        if best_score is None or packed > best_score:
            best_score = packed
            best_crop = crop
            best_meta = {
                "scene_score": float(scene_score),
                "plugin_hits": int(plugin_hits),
                "noplugin_hits": int(noplugin_hits),
                "gt_local_count": int(len(gt_local)),
            }
    return best_crop, best_meta


def bev_xy_to_pixel(
    xy: Sequence[float],
    pc_range: Sequence[float],
    left_hand: bool,
    image_size: Tuple[int, int],
) -> Tuple[float, float]:
    x, y = float(xy[0]), float(xy[1])
    if not left_hand:
        y = -y
    xmin, ymin, xmax, ymax = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    width, height = image_size
    px = (x - xmin) / (xmax - xmin) * (width - 1)
    py = (y - ymin) / (ymax - ymin) * (height - 1)
    return px, py


def get_roi_bbox_pixels(
    crop_range: Sequence[float],
    pc_range: Sequence[float],
    left_hand: bool,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    p0 = bev_xy_to_pixel((crop_range[0], crop_range[1]), pc_range, left_hand, image_size)
    p1 = bev_xy_to_pixel((crop_range[2], crop_range[3]), pc_range, left_hand, image_size)
    left = int(round(min(p0[0], p1[0])))
    top = int(round(min(p0[1], p1[1])))
    right = int(round(max(p0[0], p1[0])))
    bottom = int(round(max(p0[1], p1[1])))
    return left, top, right, bottom


def render_bev_image(
    infer_result: Dict,
    pcd: torch.Tensor,
    pc_range: Sequence[float],
    left_hand: bool,
    crop_range: Optional[Sequence[float]] = None,
    point_radius: int = 2,
    box_thickness: int = 4,
    score_thresh: float = 0.3,
    ppm: int = 10,
) -> Image.Image:
    from opencood.visualization.simple_plot3d.canvas_bev import Canvas_BEV_heading_right

    xmin, ymin = pc_range[0], pc_range[1]
    xmax, ymax = pc_range[3], pc_range[4]
    if crop_range is not None:
        xmin, crop_ymin, xmax, crop_ymax = crop_range
        if left_hand:
            ymin, ymax = crop_ymin, crop_ymax
        else:
            ymin, ymax = -crop_ymax, -crop_ymin

    width = int((xmax - xmin) * ppm)
    height = int((ymax - ymin) * ppm)
    canvas = Canvas_BEV_heading_right(
        canvas_shape=(height, width),
        canvas_x_range=(xmin, xmax),
        canvas_y_range=(ymin, ymax),
        canvas_bg_color=(15, 15, 15),
        left_hand=left_hand,
    )

    pcd_np = pcd.cpu().numpy()
    canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
    canvas.draw_canvas_points(canvas_xy[valid_mask], radius=point_radius, colors="viridis")

    gt_box = infer_result.get("gt_box_tensor")
    if gt_box is not None:
        gt_np = gt_box.cpu().numpy() if torch.is_tensor(gt_box) else np.asarray(gt_box)
        canvas.draw_boxes(gt_np, colors=(0, 255, 0), texts=None, box_line_thickness=box_thickness)

    pred_box = filter_pred_boxes_by_score(infer_result, score_thresh)
    if pred_box is not None:
        pred_np = pred_box.cpu().numpy() if torch.is_tensor(pred_box) else np.asarray(pred_box)
        canvas.draw_boxes(pred_np, colors=(255, 80, 80), texts=None, box_line_thickness=box_thickness)

    return Image.fromarray(canvas.canvas)


def compose_inset_panel(
    full_img: Image.Image,
    crop_img: Image.Image,
    crop_range: Sequence[float],
    pc_range: Sequence[float],
    left_hand: bool,
) -> Image.Image:
    panel = full_img.copy()
    draw = ImageDraw.Draw(panel)
    bbox = get_roi_bbox_pixels(crop_range, pc_range, left_hand, panel.size)
    roi_color = (255, 219, 88)
    outline_width = max(4, panel.width // 220)
    draw.rectangle(bbox, outline=roi_color, width=outline_width)

    inset_width = int(panel.width * 0.28)
    inset_height = int(round(crop_img.height * inset_width / crop_img.width))
    resized_crop = crop_img.resize((inset_width, inset_height), Image.LANCZOS)
    bordered_crop = ImageOps.expand(resized_crop, border=8, fill=(245, 245, 245))

    roi_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    candidates = [
        (28, 28),
        (panel.width - bordered_crop.width - 28, 28),
        (28, panel.height - bordered_crop.height - 28),
        (
            panel.width - bordered_crop.width - 28,
            panel.height - bordered_crop.height - 28,
        ),
    ]
    inset_x, inset_y = max(
        candidates,
        key=lambda anchor: (
            (anchor[0] + bordered_crop.width / 2.0 - roi_center[0]) ** 2
            + (anchor[1] + bordered_crop.height / 2.0 - roi_center[1]) ** 2
        ),
    )
    panel.paste(bordered_crop, (inset_x, inset_y))
    draw.rectangle(
        (inset_x, inset_y, inset_x + bordered_crop.width, inset_y + bordered_crop.height),
        outline=roi_color,
        width=max(3, outline_width - 1),
    )
    font = load_font(max(22, panel.width // 54))
    text_bbox = draw.textbbox((0, 0), "Zoom", font=font)
    label_x = inset_x + 14
    label_y = inset_y + 12
    draw.rectangle(
        (
            label_x - 6,
            label_y - 4,
            label_x + (text_bbox[2] - text_bbox[0]) + 6,
            label_y + (text_bbox[3] - text_bbox[1]) + 4,
        ),
        fill=(245, 245, 245),
        outline=roi_color,
        width=2,
    )
    draw.text((label_x, label_y), "Zoom", fill=(20, 20, 20), font=font)
    return panel


def build_compare_panel(
    noplugin_infer: Dict,
    plugin_infer: Dict,
    pcd: torch.Tensor,
    pc_range: Sequence[float],
    left_hand: bool,
    score_thresh: float = 0.3,
    crop_size: float = 32.0,
    render_mode: str = "inset",
    ppm: int = 10,
) -> Tuple[Image.Image, Dict]:
    crop_range, crop_meta = select_auto_crop(
        noplugin_infer,
        plugin_infer,
        pc_range,
        crop_size=crop_size,
        score_thresh=score_thresh,
    )
    panels = []
    for title, infer_result in (("No Plugin", noplugin_infer), ("With Plugin", plugin_infer)):
        full_img = render_bev_image(
            infer_result, pcd, pc_range, left_hand, score_thresh=score_thresh, ppm=ppm
        )
        crop_img = render_bev_image(
            infer_result,
            pcd,
            pc_range,
            left_hand,
            crop_range=crop_range,
            score_thresh=score_thresh,
            ppm=ppm,
        )
        if render_mode == "full":
            panel_img = full_img
        elif render_mode == "crop":
            panel_img = crop_img
        else:
            panel_img = compose_inset_panel(full_img, crop_img, crop_range, pc_range, left_hand)
        panels.append((title, panel_img))

    gap = 18
    title_h = 54
    width = sum(panel.width for _, panel in panels) + gap
    height = max(panel.height for _, panel in panels) + title_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = load_font(30)
    x_offset = 0
    for title, panel in panels:
        canvas.paste(panel, (x_offset, title_h))
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (x_offset + (panel.width - tw) // 2, 10),
            title,
            fill=(25, 25, 25),
            font=font,
        )
        x_offset += panel.width + gap
    return canvas, {"crop_range": [float(v) for v in crop_range], **crop_meta}


def compute_status(point: Dict[str, float]) -> str:
    if point["plugin_ap50"] >= point["ego_only_ap50"]:
        return "Above Ego-only"
    if point["plugin_ap50"] >= point["no_plugin_ap50"]:
        return "Above No-plugin"
    return "Below No-plugin"


def status_color(status: str) -> Tuple[int, int, int]:
    return {
        "Below No-plugin": (155, 49, 49),
        "Above No-plugin": (184, 119, 23),
        "Above Ego-only": (23, 127, 79),
    }.get(status, (80, 80, 80))


def plot_ap50_curves(
    ax,
    aligned_points: Sequence[Dict],
    highlight_latest: bool = True,
) -> None:
    xs = [point["sample"] for point in aligned_points]
    field_map = {
        "plugin": "plugin_ap50",
        "ego_only": "ego_only_ap50",
        "no_plugin": "no_plugin_ap50",
    }
    for key, field in field_map.items():
        label, color, linestyle, marker, linewidth, alpha = CURVE_STYLES[key]
        ys = [point[field] for point in aligned_points]
        ax.plot(
            xs,
            ys,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=2.8,
            alpha=alpha,
            label=label,
        )

    if xs and highlight_latest:
        ax.scatter(
            [xs[-1]],
            [aligned_points[-1]["plugin_ap50"]],
            s=56,
            color="#d62728",
            zorder=5,
        )


def render_convergence_plot(
    aligned_points: Sequence[Dict],
    width: int = 760,
    height: int = 420,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(width / 120, height / 120), dpi=120)
    plot_ap50_curves(ax, aligned_points, highlight_latest=True)

    ax.set_xlabel("Frames processed")
    ax.set_ylabel("Running AP@50")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: Tuple[int, int, int]) -> Tuple[int, int]:
    font = load_font(28)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + 28
    height = bbox[3] - bbox[1] + 20
    draw.rounded_rectangle((x, y, x + width, y + height), radius=18, fill=fill)
    draw.text((x + 14, y + 10), text, fill=(255, 255, 255), font=font)
    return width, height


def _draw_info_lines(
    draw: ImageDraw.ImageDraw,
    origin: Tuple[int, int],
    lines: Iterable[str],
    font_size: int = 26,
    line_gap: int = 12,
    fill: Tuple[int, int, int] = (35, 35, 35),
) -> int:
    x, y = origin
    font = load_font(font_size)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def _lookup_point(aligned_points: Sequence[Dict], sample_index: int) -> Dict:
    for point in aligned_points:
        if int(point["sample"]) == int(sample_index):
            return point
    raise KeyError(f"Cannot find readiness point for sample {sample_index}")


def _format_time(sample: int, online_step_ms: float) -> str:
    if online_step_ms <= 0:
        return ""
    return f" ({sample * online_step_ms / 1000.0:.1f}s)"


def _format_stable_line(label: str, milestone: Optional[Dict], online_step_ms: float) -> str:
    if milestone is None:
        return f"Stable above {label}: not yet"
    sample = int(milestone["sample"])
    return f"Stable above {label}: sample {sample}{_format_time(sample, online_step_ms)}"


def render_dashboard_frame(
    compare_panel: Image.Image,
    convergence_plot: Image.Image,
    frame_record: Dict,
    readiness_summary: Dict,
    title: str,
    subtitle: str = "",
    canvas_size: Tuple[int, int] = (1920, 1080),
) -> Image.Image:
    canvas = Image.new("RGB", canvas_size, (246, 247, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(44)
    subtitle_font = load_font(24)
    draw.text((48, 28), title, fill=(20, 20, 24), font=title_font)
    if subtitle:
        draw.text((50, 84), subtitle, fill=(80, 80, 88), font=subtitle_font)

    left_w = 1230
    left_x = 40
    top_y = 138

    panel = ImageOps.contain(compare_panel, (left_w, 520))
    panel_bg = Image.new("RGB", (left_w + 24, panel.height + 24), (255, 255, 255))
    panel_bg.paste(panel, (12, 12))
    draw_panel = ImageDraw.Draw(panel_bg)
    draw_panel.rounded_rectangle(
        (0, 0, panel_bg.width - 1, panel_bg.height - 1),
        radius=24,
        outline=(223, 226, 232),
        width=2,
    )
    canvas.paste(panel_bg, (left_x, top_y))

    plot = ImageOps.contain(convergence_plot, (left_w, 430))
    plot_bg = Image.new("RGB", (left_w + 24, 454), (255, 255, 255))
    plot_bg.paste(plot, ((plot_bg.width - plot.width) // 2, (plot_bg.height - plot.height) // 2))
    draw_plot = ImageDraw.Draw(plot_bg)
    draw_plot.rounded_rectangle(
        (0, 0, plot_bg.width - 1, plot_bg.height - 1),
        radius=24,
        outline=(223, 226, 232),
        width=2,
    )
    canvas.paste(plot_bg, (left_x, top_y + panel_bg.height + 20))

    card = Image.new("RGB", (590, 878), (255, 255, 255))
    draw_card = ImageDraw.Draw(card)
    draw_card.rounded_rectangle((0, 0, 589, 877), radius=24, outline=(223, 226, 232), width=2)
    draw_card.text((28, 26), "Online Readiness", fill=(20, 20, 24), font=load_font(34))

    online_step_ms = readiness_summary.get("online_step_ms", 0.0)
    current_point = _lookup_point(readiness_summary["aligned_points"], frame_record["sample_index"])
    status = compute_status(current_point)
    badge_fill = status_color(status)
    _draw_badge(draw_card, 28, 80, status, badge_fill)

    seconds = (frame_record["sample_index"] * online_step_ms / 1000.0) if online_step_ms > 0 else None
    lines = [
        f"Frame: {frame_record['sample_index']}",
        f"Plugin AP@50: {current_point['plugin_ap50']:.2f}",
        f"Ego-only AP@50: {current_point['ego_only_ap50']:.2f}",
        f"No-plugin AP@50: {current_point['no_plugin_ap50']:.2f}",
    ]
    if seconds is not None:
        lines.append(f"Estimated time: {seconds:.1f}s")
    lines.extend(
        [
            f"Margin vs ego-only: {current_point['margin_vs_ego_only']:+.2f}",
            f"Margin vs no-plugin: {current_point['margin_vs_no_plugin']:+.2f}",
            f"Now above ego-only: {'yes' if current_point['margin_vs_ego_only'] >= 0 else 'no'}",
            f"Now above no-plugin: {'yes' if current_point['margin_vs_no_plugin'] >= 0 else 'no'}",
        ]
    )
    y_end = _draw_info_lines(draw_card, (28, 150), lines, font_size=24)

    milestones = readiness_summary["milestones"]
    draw_card.text((28, y_end + 26), "Stable Milestones", fill=(20, 20, 24), font=load_font(30))
    milestone_lines = [
        _format_stable_line(
            "ego-only",
            milestones.get("ego_only", {}).get("stable_above"),
            online_step_ms,
        ),
        _format_stable_line(
            "no-plugin",
            milestones.get("no_plugin", {}).get("stable_above"),
            online_step_ms,
        ),
    ]
    _draw_info_lines(
        draw_card,
        (28, y_end + 72),
        milestone_lines,
        font_size=24,
        line_gap=14,
        fill=(36, 86, 51),
    )

    card_x = left_x + panel_bg.width + 20
    canvas.paste(card, (card_x, top_y))
    return canvas


def write_mp4_from_frames(frame_paths: Sequence[str], output_path: str, fps: int = 10) -> None:
    if not frame_paths:
        raise ValueError("No frames provided for video writing.")

    first = cv2.imread(frame_paths[0])
    if first is None:
        raise FileNotFoundError(frame_paths[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                raise FileNotFoundError(frame_path)
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
