"""
Online Test-Time Training (TTT) for base-free heterogeneous cooperative perception.

v3.1: Ego-as-Teacher distillation with enhancement signal.
    - Preservation loss: distill where teacher is confident (don't hurt ego detections)
    - Enhancement loss: boost student confidence where teacher is uncertain but has
      some signal (encourage fusion to leverage neighbor info)
    - Default protocol is single-pass; optional multi-pass warmup is supported when
      epochs > 1

All encoders / fusion / heads remain frozen; only plugin parameters are updated.

Typical usage:
  python -m opencood.tools.online_adapt \
    --model_dir /path/to/DirectHeter_base_free_lidar_camera \
    --output_dir /path/to/output \
    --lr 1e-4 --epochs 1 --teacher_conf_thresh 0.3 \
    --boost_weight 0.1 --boost_lo 0.1 --boost_hi 0.3 \
    --plugin_adain_alpha_init_logit -10
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import opencood.data_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils, eval_utils_mc
from opencood.utils.common_utils import update_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online TTT with ego-as-teacher (base-free)")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    # Data
    p.add_argument("--comm_range", type=float, default=100.0)
    p.add_argument("--use_cav", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shuffle_test", action="store_true",
                    help="Shuffle the test stream order for ordering-sensitivity experiments")
    p.add_argument("--test_order_seed", type=int, default=0,
                    help="Random seed used only for test-stream shuffling when --shuffle_test is set")
    p.add_argument("--fuse_method", type=str, default="weighted",
                    choices=["weighted", "max", "mean", "attn", "v2xvit"],
                    help="Fusion aggregation method inside PyramidFusion")

    # Plugin config
    p.add_argument("--src_modality", type=str, default="m2")
    p.add_argument(
        "--src_modalities",
        type=str,
        default="",
        help="Comma-separated modality names for multi-src plugin routing, e.g. m2,m3,m4",
    )
    p.add_argument("--plugin_hidden", type=int, default=128)
    p.add_argument("--plugin_blocks", type=int, default=3)
    p.add_argument("--plugin_gn_groups", type=int, default=16)
    p.add_argument("--plugin_adain_alpha_init_logit", type=float, default=10.0)
    p.add_argument("--plugin_gate_init_logit", type=float, default=0.0)

    # Optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=1,
                    help="Number of passes over the test stream. Default 1; if >1, epochs-1 are warmup-only and the last pass trains+evaluates.")

    # Distillation — preservation
    p.add_argument("--cls_weight", type=float, default=1.0)
    p.add_argument("--reg_weight", type=float, default=1.0)
    p.add_argument("--dir_weight", type=float, default=0.5)
    p.add_argument("--teacher_conf_thresh", type=float, default=0.0,
                    help="Only distill where teacher confidence > thresh (0 = distill everywhere)")

    # Distillation — enhancement
    p.add_argument("--boost_weight", type=float, default=0.0,
                    help="Weight for enhancement loss (0 = disabled)")
    p.add_argument("--boost_lo", type=float, default=0.1,
                    help="Lower bound of teacher confidence for boost region")
    p.add_argument("--boost_hi", type=float, default=0.3,
                    help="Upper bound of teacher confidence for boost region")

    # Logging
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_log", action="store_true")
    p.add_argument("--no_plugin", action="store_true",
                    help="Disable plugin entirely (for no-plugin baselines)")
    p.add_argument("--train_fusion", action="store_true",
                    help="Also unfreeze fusion module params (e.g. v2xvit_levels) during TTT")
    p.add_argument("--teacher_mode", type=str, default="ego",
                    choices=["ego", "neighbor"],
                    help="Teacher source: 'ego' (default) or 'neighbor' (use neighbor's single-agent prediction)")
    p.add_argument("--score_threshold", type=float, default=None,
                    help="Override postprocessor score_threshold (default: use config value)")
    p.add_argument("--convergence_interval", type=int, default=0,
                    help="Compute running AP every N samples during eval epoch (0 = disabled)")
    p.add_argument("--export_frame_interval", type=int, default=0,
                    help="Export readiness comparison panels every N samples during the eval epoch (0 = disabled)")
    p.add_argument("--export_frame_dir", type=str, default="",
                    help="Directory for exported readiness panels and frame metadata (default: <output_dir>/readiness_frames)")
    p.add_argument("--export_frame_limit", type=int, default=0,
                    help="Maximum number of readiness panels to export (0 = no limit)")
    p.add_argument("--export_frame_render_mode", type=str, default="inset",
                    choices=["full", "crop", "inset"],
                    help="Panel variant to export for readiness visualization")
    p.add_argument("--export_frame_crop_size", type=float, default=32.0,
                    help="Crop size in meters used for automatic zoom selection when exporting readiness frames")
    p.add_argument("--export_frame_score_thresh", type=float, default=0.3,
                    help="Score threshold used for readiness BEV rendering")
    p.add_argument("--export_frame_ppm", type=int, default=10,
                    help="Pixels-per-meter used for exported BEV readiness panels")
    p.add_argument("--strict_n_car", type=int, default=0,
                    help="Strict N-car filtering: only evaluate scenes with exactly N agents (0=disabled)")
    p.add_argument("--max_eval_samples", type=int, default=0,
                    help="Optional early stop after N samples during evaluation/debugging (0 = full stream)")
    p.add_argument("--assignment_path", type=str, default=None,
                    help="Override modality assignment JSON file path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _inject_plugin_cfg(hypes: dict, opt: argparse.Namespace) -> dict:
    hypes = copy.deepcopy(hypes)
    hypes.setdefault("model", {}).setdefault("args", {})
    if str(getattr(opt, "src_modalities", "")).strip():
        src_modalities = [
            x.strip() for x in str(opt.src_modalities).split(",") if x.strip()
        ]
    else:
        src_modalities = [opt.src_modality]
    hypes["model"]["args"]["plugin"] = {
        "enable": True,
        "src_modality": src_modalities[0],
        "src_modalities": src_modalities,
        "args": {
            "in_channels": 64,
            "hidden_channels": opt.plugin_hidden,
            "num_blocks": opt.plugin_blocks,
            "gn_groups": opt.plugin_gn_groups,
            "adain_alpha_init_logit": opt.plugin_adain_alpha_init_logit,
            "gate_init_logit": opt.plugin_gate_init_logit,
        },
    }
    return hypes


_EVAL_IOUS = (0.3, 0.5, 0.7)


def _init_result_stats_single():
    """Create fresh result_stat dicts for single-class AP evaluation."""
    def _make():
        return {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }
    return _make(), _make(), _make(), _make()  # overall, short, middle, long


def _init_result_stats_mc() -> dict:
    """Create fresh result_stat dict for multi-class overall mAP evaluation."""
    return {
        class_name: {
            iou_thresh: {"tp": [], "fp": [], "gt": 0}
            for iou_thresh in _EVAL_IOUS
        }
        for class_name in opencood.data_utils.SUPER_CLASS_MAP.keys()
    }


def _use_multi_class_eval(hypes: dict) -> bool:
    fusion_cfg = hypes.get("fusion", {})
    dataset_name = str(fusion_cfg.get("dataset", "")).lower()
    fusion_core = str(fusion_cfg.get("core_method", "")).lower()
    num_class = int(hypes.get("num_class", hypes.get("postprocess", {}).get("num_class", 1)))
    return dataset_name == "v2xreal" or "3class" in fusion_core or num_class > 1


def _parse_mc_eval_tensors(post_ret):
    pred_box_tensor, pred_score, gt_box_tensor = post_ret[0], post_ret[1], post_ret[2]
    gt_label_tensor = post_ret[3] if len(post_ret) > 3 else None
    if gt_label_tensor is None:
        raise RuntimeError("multi-class evaluation requires gt_label_tensor from dataset.post_process")

    device = gt_box_tensor.device
    gt_label_tensor = gt_label_tensor.view(-1).to(device=device, dtype=torch.long)

    if pred_score is None:
        pred_conf = torch.empty((0,), device=device)
        pred_label_tensor = torch.empty((0,), device=device, dtype=torch.long)
    elif pred_score.dim() == 2 and pred_score.shape[1] >= 2:
        pred_conf = pred_score[:, 0]
        pred_label_tensor = pred_score[:, 1].to(device=device, dtype=torch.long)
    else:
        raise RuntimeError("MC eval expects pred_score to contain both score and label")

    if pred_box_tensor is None:
        pred_box_tensor = gt_box_tensor.new_empty((0,) + tuple(gt_box_tensor.shape[1:]))

    return pred_box_tensor, pred_conf, pred_label_tensor, gt_box_tensor, gt_label_tensor


def _update_result_stats_mc(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    result_stat: dict,
) -> None:
    for class_id, class_name in enumerate(result_stat.keys(), start=1):
        keep_pred = pred_labels == class_id
        keep_gt = gt_labels == class_id
        for iou_thresh in _EVAL_IOUS:
            eval_utils_mc.caluclate_tp_fp(
                pred_boxes[keep_pred, ...],
                pred_scores[keep_pred],
                gt_boxes[keep_gt, ...],
                result_stat[class_name],
                iou_thresh,
            )


def _calculate_map_mc(result_stat: dict, iou_thresh: float) -> float:
    aps = [eval_utils_mc.calculate_ap(result_stat[class_name], iou_thresh)[0]
           for class_name in result_stat.keys()]
    return float(sum(aps) / len(aps)) if aps else 0.0


def _maybe_shuffle_test_dataset(dataset, opt: argparse.Namespace, out_dir: str):
    if not opt.shuffle_test:
        return dataset

    num_samples = len(dataset)
    order = np.random.RandomState(opt.test_order_seed).permutation(num_samples).tolist()
    order_path = os.path.join(out_dir, "test_order.json")
    with open(order_path, "w") as f:
        json.dump(
            {
                "shuffle_test": True,
                "test_order_seed": int(opt.test_order_seed),
                "num_samples": int(num_samples),
                "order": order,
            },
            f,
        )
    print(
        f"[online-ttt] shuffle_test enabled: test_order_seed={opt.test_order_seed} "
        f"(saved to {order_path})"
    )
    return Subset(dataset, order)


def _expand_anchor_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand per-anchor mask [B,A,H,W] to match target channels [B,C,H,W].

    When A does not divide C evenly (e.g. multi-class cls has A=18 but reg
    has C=42), reduce mask to GCD(A,C) groups first, then expand.
    """
    from math import gcd
    a, c = mask.shape[1], target.shape[1]
    if c % a == 0:
        return mask.repeat_interleave(c // a, dim=1)
    # Reduce mask to g = gcd(a, c) groups via max-pool, then expand
    g = gcd(a, c)
    b, _, h, w = mask.shape
    reduced = mask.view(b, g, a // g, h, w).amax(dim=2)  # [B, g, H, W]
    return reduced.repeat_interleave(c // g, dim=1)


def _distillation_loss(student_out, teacher_out, opt):
    """Preservation + Enhancement distillation loss."""
    s_cls = student_out["cls_preds"]
    t_cls = teacher_out["cls_preds"].detach()
    s_reg = student_out["reg_preds"]
    t_reg = teacher_out["reg_preds"].detach()
    s_dir = student_out["dir_preds"]
    t_dir = teacher_out["dir_preds"].detach()

    t_prob = torch.sigmoid(t_cls)
    s_prob = torch.sigmoid(s_cls)

    # ── Preservation loss: distill where teacher is confident ──────────
    if opt.teacher_conf_thresh > 0:
        conf_mask = (t_prob > opt.teacher_conf_thresh).float()
        cls_loss = F.binary_cross_entropy_with_logits(
            s_cls, t_prob, reduction="none"
        )
        cls_loss = (cls_loss * conf_mask).sum() / conf_mask.sum().clamp(min=1)

        reg_mask = _expand_anchor_mask(conf_mask, s_reg)
        reg_loss = F.smooth_l1_loss(s_reg, t_reg, reduction="none")
        reg_loss = (reg_loss * reg_mask).sum() / reg_mask.sum().clamp(min=1)

        dir_mask = _expand_anchor_mask(conf_mask, s_dir)
        dir_loss = F.mse_loss(s_dir, t_dir, reduction="none")
        dir_loss = (dir_loss * dir_mask).sum() / dir_mask.sum().clamp(min=1)
    else:
        cls_loss = F.mse_loss(s_prob, t_prob)
        reg_loss = F.smooth_l1_loss(s_reg, t_reg)
        dir_loss = F.mse_loss(s_dir, t_dir)

    preserve = (opt.cls_weight * cls_loss
                + opt.reg_weight * reg_loss
                + opt.dir_weight * dir_loss)

    # ── Enhancement loss: boost student in teacher's uncertain region ──
    boost_loss_val = 0.0
    if opt.boost_weight > 0:
        boost_mask = ((t_prob > opt.boost_lo) & (t_prob <= opt.boost_hi)).float()
        n_boost = boost_mask.sum().clamp(min=1)
        # Encourage student to be more confident than teacher in these regions
        # -log(s_prob) weighted by boost_mask: pushes s_prob toward 1
        boost_loss = (-torch.log(s_prob + 1e-7) * boost_mask).sum() / n_boost
        boost_loss_val = float(boost_loss)
        total = preserve + opt.boost_weight * boost_loss
    else:
        total = preserve

    return total, float(cls_loss), float(reg_loss), float(dir_loss), boost_loss_val


def _run_teacher_forward(model, cav_content, device):
    """Ego-only teacher forward (no plugin, record_len=1)."""
    with torch.no_grad():
        saved_plugin = model.plugin_enabled
        model.plugin_enabled = False

        saved_rl = cav_content["record_len"].clone()
        saved_ml = cav_content["agent_modality_list"]

        cav_content["record_len"] = torch.ones(1, dtype=saved_rl.dtype, device=device)
        cav_content["agent_modality_list"] = saved_ml[:1]

        teacher_out = model(cav_content)

        cav_content["record_len"] = saved_rl
        cav_content["agent_modality_list"] = saved_ml
        model.plugin_enabled = saved_plugin

    return teacher_out


def _run_neighbor_teacher_forward(model, cav_content, device):
    """Neighbor-only teacher forward (no plugin, record_len=1).

    Swaps agent order so the neighbor (index 1) becomes the sole agent,
    then runs single-agent inference.  Requires record_len sum >= 2.
    """
    with torch.no_grad():
        saved_plugin = model.plugin_enabled
        model.plugin_enabled = False

        saved_rl = cav_content["record_len"].clone()
        saved_ml = cav_content["agent_modality_list"]
        total_agents = int(saved_rl.sum())

        if total_agents < 2:
            # Fallback: only ego available, use ego as teacher
            return _run_teacher_forward(model, cav_content, device)

        # Save and reorder spatial tensors: move neighbor (idx 1) to position 0
        saved_tensors = {}
        for key in list(cav_content.keys()):
            val = cav_content[key]
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == total_agents:
                saved_tensors[key] = val
                cav_content[key] = val[1:2]  # neighbor slice only

        cav_content["record_len"] = torch.ones(1, dtype=saved_rl.dtype, device=device)
        cav_content["agent_modality_list"] = saved_ml[1:2]

        teacher_out = model(cav_content)

        # Restore
        for key, val in saved_tensors.items():
            cav_content[key] = val
        cav_content["record_len"] = saved_rl
        cav_content["agent_modality_list"] = saved_ml
        model.plugin_enabled = saved_plugin

    return teacher_out


def _run_noplugin_coop_forward(model, cav_content):
    """Cooperative forward with plugin disabled, preserving the original agent set."""
    with torch.no_grad():
        saved_plugin = model.plugin_enabled
        model.plugin_enabled = False
        output = model(cav_content)
        model.plugin_enabled = saved_plugin
    return output


def _left_hand_from_hypes(hypes: dict) -> bool:
    test_dir = str(hypes.get("test_dir", ""))
    return any(flag in test_dir for flag in ("OPV2V", "V2XSET", "V2XREAL"))


def _post_ret_to_vis_result(post_ret):
    pred_box_tensor, pred_score, gt_box_tensor = post_ret[0], post_ret[1], post_ret[2]
    vis_result = {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "score_tensor": pred_score,
        "gt_box_tensor": gt_box_tensor,
    }
    if len(post_ret) > 3:
        vis_result["gt_label_tensor"] = post_ret[3]
    return vis_result


def _compute_running_metrics_single(result_stat):
    snapshot = copy.deepcopy(result_stat)
    ap30, _, _ = eval_utils.calculate_ap(snapshot, 0.3)
    ap50, _, _ = eval_utils.calculate_ap(snapshot, 0.5)
    ap70, _, _ = eval_utils.calculate_ap(snapshot, 0.7)
    return float(ap30), float(ap50), float(ap70)


def _maybe_export_readiness_panel(
    *,
    batch_data,
    sample_index: int,
    noplugin_post_ret,
    plugin_post_ret,
    pc_range,
    left_hand: bool,
    frames_dir: str,
    render_mode: str,
    crop_size: float,
    score_thresh: float,
    ppm: int,
    running_metrics: tuple[float, float, float],
):
    from scripts.readiness_vis_utils import build_compare_panel

    pcd = batch_data["ego"]["origin_lidar"][0].detach().cpu()
    panel_img, crop_meta = build_compare_panel(
        _post_ret_to_vis_result(noplugin_post_ret),
        _post_ret_to_vis_result(plugin_post_ret),
        pcd=pcd,
        pc_range=pc_range,
        left_hand=left_hand,
        score_thresh=score_thresh,
        crop_size=crop_size,
        render_mode=render_mode,
        ppm=ppm,
    )
    panel_filename = f"frame_{sample_index:05d}.png"
    panel_path = os.path.join(frames_dir, panel_filename)
    panel_img.save(panel_path)
    ap30, ap50, ap70 = running_metrics
    return {
        "sample_index": int(sample_index),
        "panel_path": panel_filename,
        "running_ap30": ap30 * 100.0,
        "running_ap50": ap50 * 100.0,
        "running_ap70": ap70 * 100.0,
        "crop_range": crop_meta.get("crop_range"),
        "scene_score": crop_meta.get("scene_score", 0.0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    opt = _parse_args()
    _set_seed(opt.seed)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    out_dir = opt.output_dir or (opt.model_dir.rstrip("/") + "_online_ttt")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────
    class _Opt:
        model_dir = opt.model_dir
        fusion_method = "intermediate"

    hypes = yaml_utils.load_yaml(None, _Opt())
    if not opt.no_plugin:
        hypes = _inject_plugin_cfg(hypes, opt)

    hypes["validate_dir"] = hypes["test_dir"]
    hypes["comm_range"] = float(opt.comm_range)
    hypes["use_cav"] = int(opt.use_cav)
    fusion_core = str(hypes["fusion"]["core_method"])
    if fusion_core == "intermediateheter":
        hypes["fusion"]["core_method"] = "intermediateheterinfer"
    elif fusion_core != "intermediateheterinfer":
        hypes["fusion"]["core_method"] = fusion_core + "infer"
    hypes = update_dict(hypes, {"ego_modality": "m1"})

    if opt.strict_n_car > 0:
        hypes['strict_n_car'] = opt.strict_n_car
        print(f"[online-ttt] strict_n_car={hypes['strict_n_car']} (only evaluate scenes with exactly {opt.strict_n_car} agents)")

    if opt.assignment_path:
        hypes['heter']['assignment_path'] = opt.assignment_path
        print(f"[online-ttt] override assignment_path={opt.assignment_path}")

    # Fix mapping_dict: training may save 'none' for unused modalities in v2v mode
    md = hypes.get("heter", {}).get("mapping_dict", {})
    ego_mod = hypes.get("heter", {}).get("ego_modality", "m1")
    if any(v == "none" for v in md.values()):
        hypes["heter"]["mapping_dict"] = {k: (ego_mod if v == "none" else v) for k, v in md.items()}

    # Ensure add_data_extension includes bev_visibility.png when camera modality is present
    ms = hypes.get("heter", {}).get("modality_setting", {})
    has_camera = any(
        isinstance(v, dict) and v.get("sensor_type") == "camera"
        for v in ms.values()
    )
    if has_camera:
        ext = hypes.get("add_data_extension", [])
        if "bev_visibility.png" not in ext:
            hypes["add_data_extension"] = ext + ["bev_visibility.png"]

    # Inject fuse_method into fusion backbone config
    if "model" in hypes and "args" in hypes["model"] and "fusion_backbone" in hypes["model"]["args"]:
        hypes["model"]["args"]["fusion_backbone"]["fuse_method"] = opt.fuse_method

    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    parser_func = None
    for name, func in yaml_utils_lib.__dict__.items():
        if name == hypes["yaml_parser"]:
            parser_func = func
            break
    if parser_func is None:
        raise RuntimeError(f"Cannot find yaml_parser={hypes.get('yaml_parser')}")
    hypes = parser_func(hypes)

    # ── Build model ────────────────────────────────────────────────────────
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.to(device)

    if opt.no_plugin:
        # Disable plugin entirely for baseline evaluation
        if hasattr(model, "plugin_enabled"):
            model.plugin_enabled = False
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        print(f"[online-ttt] no_plugin mode — evaluation only (fuse_method={opt.fuse_method})")
        optimizer = None
    else:
        for name, p in model.named_parameters():
            trainable = name.startswith("plugin") or name.startswith("plugins.")
            if opt.train_fusion and "v2xvit_levels" in name:
                trainable = True
            p.requires_grad_(trainable)

        train_params = [p for p in model.parameters() if p.requires_grad]
        if len(train_params) == 0:
            raise RuntimeError("No trainable plugin parameters found.")

        n_params = sum(p.numel() for p in train_params)
        print(f"[online-ttt] trainable params: {n_params:,} (train_fusion={opt.train_fusion})")
        print(f"[online-ttt] epochs={opt.epochs} (warmup={max(opt.epochs-1,0)}, eval=last)")
        print(f"[online-ttt] teacher_conf_thresh={opt.teacher_conf_thresh}")
        print(f"[online-ttt] boost_weight={opt.boost_weight} boost_range=[{opt.boost_lo}, {opt.boost_hi}]")
        print(f"[online-ttt] teacher_mode={opt.teacher_mode}")

        optimizer = torch.optim.Adam(train_params, lr=opt.lr, weight_decay=opt.weight_decay)

        model.eval()
        if hasattr(model, "plugin") and model.plugin is not None:
            model.plugin.train()
        if hasattr(model, "plugins") and model.plugins is not None:
            model.plugins.train()

    # ── Build dataset / loader ─────────────────────────────────────────────
    if opt.score_threshold is not None:
        hypes["postprocess"]["target_args"]["score_threshold"] = opt.score_threshold
        print(f"[online-ttt] override score_threshold={opt.score_threshold}")
    export_frames_enabled = (not opt.no_plugin) and opt.export_frame_interval > 0
    export_frames_dir = opt.export_frame_dir or os.path.join(out_dir, "readiness_frames")
    export_frames_meta = []
    if export_frames_enabled:
        os.makedirs(export_frames_dir, exist_ok=True)
        print(
            f"[online-ttt] readiness frame export enabled: interval={opt.export_frame_interval}, "
            f"dir={export_frames_dir}"
        )

    dataset = build_dataset(hypes, visualize=export_frames_enabled, train=False)
    dataset_for_loader = _maybe_shuffle_test_dataset(dataset, opt, out_dir)
    loader = DataLoader(
        dataset_for_loader,
        batch_size=1,
        num_workers=opt.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    total_samples = len(dataset_for_loader)
    total_epochs = 1 if opt.no_plugin else max(opt.epochs, 1)
    convergence_log = []  # [(sample_idx, ap30, ap50, ap70), ...]
    use_mc_eval = _use_multi_class_eval(hypes)
    if use_mc_eval:
        result_stat = _init_result_stats_mc()
        result_stat_short = result_stat_middle = result_stat_long = None
        print("[online-ttt] using overall multi-class evaluator (no short/middle/long split)")
    else:
        result_stat, result_stat_short, result_stat_middle, result_stat_long = _init_result_stats_single()
    pc_range = hypes["postprocess"]["gt_range"]
    left_hand = _left_hand_from_hypes(hypes)

    # ── Multi-epoch loop ───────────────────────────────────────────────────
    for epoch in range(total_epochs):
        is_eval_epoch = (epoch == total_epochs - 1)
        tag = "eval" if is_eval_epoch else "warmup"

        running_loss = 0.0
        print(f"\n[online-ttt] epoch {epoch+1}/{total_epochs} ({tag}) — {total_samples} samples")

        for i, batch_data in enumerate(loader):
            if batch_data is None:
                continue
            if opt.max_eval_samples > 0 and i >= opt.max_eval_samples:
                print(f"[online-ttt] max_eval_samples reached at sample {i}; stopping early for debug.")
                break

            batch_data = train_utils.to_device(batch_data, device)
            cav_content = batch_data["ego"]
            sample_index = i + 1
            should_export_frame = (
                export_frames_enabled
                and is_eval_epoch
                and sample_index % opt.export_frame_interval == 0
                and (
                    opt.export_frame_limit <= 0
                    or len(export_frames_meta) < opt.export_frame_limit
                )
            )
            noplugin_out = None

            if opt.no_plugin:
                # ── No-plugin baseline: inference only ────────────────
                with torch.no_grad():
                    student_out = model(cav_content)
            else:
                # ── Teacher forward ────────────────────────────────────────
                if opt.teacher_mode == "neighbor":
                    teacher_out = _run_neighbor_teacher_forward(model, cav_content, device)
                else:
                    teacher_out = _run_teacher_forward(model, cav_content, device)

                if should_export_frame:
                    noplugin_out = _run_noplugin_coop_forward(model, cav_content)

                # ── Student forward: full fusion with plugin ───────────────
                student_out = model(cav_content)

                # ── Loss + update ──────────────────────────────────────────
                loss, cls_l, reg_l, dir_l, boost_l = _distillation_loss(
                    student_out, teacher_out, opt
                )

                if loss.requires_grad:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if opt.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(train_params, opt.grad_clip)
                    optimizer.step()

                running_loss += float(loss)

            # ── Evaluate on last epoch only ────────────────────────────
            if is_eval_epoch:
                with torch.no_grad():
                    output_dict = {"ego": student_out}
                    post_ret = dataset.post_process(
                        batch_data, output_dict
                    )
                    if use_mc_eval:
                        pred_box_tensor, pred_conf, pred_label_tensor, gt_box_tensor, gt_label_tensor = \
                            _parse_mc_eval_tensors(post_ret)
                        _update_result_stats_mc(
                            pred_box_tensor,
                            pred_conf,
                            pred_label_tensor,
                            gt_box_tensor,
                            gt_label_tensor,
                            result_stat,
                        )
                    else:
                        pred_box_tensor, pred_score, gt_box_tensor = post_ret[0], post_ret[1], post_ret[2]
                        if pred_score is not None and pred_score.dim() == 2:
                            pred_score = pred_score[:, 0]
                        for iou_thresh in _EVAL_IOUS:
                            eval_utils.caluclate_tp_fp(
                                pred_box_tensor, pred_score, gt_box_tensor,
                                result_stat, iou_thresh
                            )
                            eval_utils.caluclate_tp_fp(
                                pred_box_tensor, pred_score, gt_box_tensor,
                                result_stat_short, iou_thresh,
                                left_range=0, right_range=30
                            )
                            eval_utils.caluclate_tp_fp(
                                pred_box_tensor, pred_score, gt_box_tensor,
                                result_stat_middle, iou_thresh,
                                left_range=30, right_range=50
                            )
                            eval_utils.caluclate_tp_fp(
                                pred_box_tensor, pred_score, gt_box_tensor,
                                result_stat_long, iou_thresh,
                                left_range=50, right_range=100
                            )

                current_metrics = None
                if should_export_frame and not use_mc_eval:
                    current_metrics = _compute_running_metrics_single(result_stat)
                    noplugin_post_ret = dataset.post_process(batch_data, {"ego": noplugin_out})
                    export_meta = _maybe_export_readiness_panel(
                        batch_data=batch_data,
                        sample_index=sample_index,
                        noplugin_post_ret=noplugin_post_ret,
                        plugin_post_ret=post_ret,
                        pc_range=pc_range,
                        left_hand=left_hand,
                        frames_dir=export_frames_dir,
                        render_mode=opt.export_frame_render_mode,
                        crop_size=opt.export_frame_crop_size,
                        score_thresh=opt.export_frame_score_thresh,
                        ppm=opt.export_frame_ppm,
                        running_metrics=current_metrics,
                    )
                    export_frames_meta.append(export_meta)
                    print(
                        f"[readiness-vis] sample {sample_index}: "
                        f"AP@50={export_meta['running_ap50']:.2f} "
                        f"saved {export_meta['panel_path']}"
                    )

                # ── Convergence checkpoint ─────────────────────────────
                if opt.convergence_interval > 0 and (i + 1) % opt.convergence_interval == 0:
                    if use_mc_eval:
                        _snap = copy.deepcopy(result_stat)
                        _m30 = _calculate_map_mc(_snap, 0.3)
                        _m50 = _calculate_map_mc(_snap, 0.5)
                        _m70 = _calculate_map_mc(_snap, 0.7)
                        convergence_log.append((i + 1, _m30, _m50, _m70))
                        print(
                            f"[convergence] sample {i+1}: "
                            f"mAP@30={_m30:.4f} mAP@50={_m50:.4f} mAP@70={_m70:.4f}"
                        )
                    else:
                        if current_metrics is None:
                            _a30, _a50, _a70 = _compute_running_metrics_single(result_stat)
                        else:
                            _a30, _a50, _a70 = current_metrics
                        convergence_log.append((i + 1, float(_a30), float(_a50), float(_a70)))
                        print(
                            f"[convergence] sample {i+1}: "
                            f"AP@30={_a30:.4f} AP@50={_a50:.4f} AP@70={_a70:.4f}"
                        )

            # ── Logging ────────────────────────────────────────────────
            if not opt.no_plugin and opt.log_interval > 0 and (i + 1) % opt.log_interval == 0:
                avg_loss = running_loss / opt.log_interval
                boost_str = f" boost={boost_l:.4f}" if opt.boost_weight > 0 else ""
                print(
                    f"[online-ttt] e{epoch+1} sample {i+1}/{total_samples} "
                    f"loss={avg_loss:.4f} (cls={cls_l:.4f} reg={reg_l:.4f} "
                    f"dir={dir_l:.4f}{boost_str})"
                )
                running_loss = 0.0

    # ── Final evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if use_mc_eval:
        eval_utils_mc.eval_final_results(result_stat, out_dir)
    else:
        for tag, stat in [
            ("short", result_stat_short),
            ("middle", result_stat_middle),
            ("long", result_stat_long),
            ("overall", result_stat),
        ]:
            ap30, _, _ = eval_utils.calculate_ap(stat, 0.3)
            ap50, _, _ = eval_utils.calculate_ap(stat, 0.5)
            ap70, _, _ = eval_utils.calculate_ap(stat, 0.7)
            print(
                f"{tag:>8s} The Average Precision at IOU 0.3 is {ap30:.4f}, "
                f"The Average Precision at IOU 0.5 is {ap50:.4f}, "
                f"The Average Precision at IOU 0.7 is {ap70:.4f}"
            )

        eval_utils.eval_final_results(result_stat, out_dir, "online_ttt")

    # ── Save convergence log ──────────────────────────────────────────────
    if convergence_log:
        conv_path = os.path.join(out_dir, "convergence.json")
        with open(conv_path, "w") as f:
            json.dump(convergence_log, f)
        print(f"[convergence] saved {len(convergence_log)} checkpoints to {conv_path}")

    if export_frames_meta:
        frames_payload = {
            "model_dir": opt.model_dir,
            "output_dir": out_dir,
            "export_frame_interval": int(opt.export_frame_interval),
            "render_mode": opt.export_frame_render_mode,
            "crop_size": float(opt.export_frame_crop_size),
            "score_thresh": float(opt.export_frame_score_thresh),
            "ppm": int(opt.export_frame_ppm),
            "frames": export_frames_meta,
        }
        frames_json_path = os.path.join(export_frames_dir, "frames.json")
        with open(frames_json_path, "w") as f:
            json.dump(frames_payload, f, indent=2)
        print(f"[readiness-vis] saved {len(export_frames_meta)} frames metadata to {frames_json_path}")

    # ── Save checkpoint ────────────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "net_epoch_bestval_at0.pth")
    torch.save(model.state_dict(), ckpt_path)

    cfg_path = os.path.join(out_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        import yaml
        yaml.dump(hypes, f, default_flow_style=False, allow_unicode=True)

    if opt.save_log:
        with open(os.path.join(out_dir, "adapt_log.json"), "w") as f:
            json.dump(
                {
                    "model_dir": opt.model_dir,
                    "out_dir": out_dir,
                    "loss_type": "distillation_ego_teacher_v3.1",
                    "total_samples": total_samples,
                    "epochs": opt.epochs,
                    "lr": opt.lr,
                    "teacher_conf_thresh": opt.teacher_conf_thresh,
                    "boost_weight": opt.boost_weight,
                    "boost_lo": opt.boost_lo,
                    "boost_hi": opt.boost_hi,
                    "src_modality": opt.src_modality,
                    "src_modalities": opt.src_modalities,
                    "plugin_adain_alpha_init_logit": opt.plugin_adain_alpha_init_logit,
                    "teacher_mode": opt.teacher_mode,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"\nSaved checkpoint → {ckpt_path}")
    print(f"Saved config     → {cfg_path}")
    print(f"Saved eval       → {out_dir}/eval.yaml")


if __name__ == "__main__":
    main()
