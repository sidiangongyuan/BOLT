"""Render a paper-style dashboard MP4 from exported readiness frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.readiness_vis_utils import (
    render_convergence_plot,
    render_dashboard_frame,
    write_mp4_from_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a dashboard MP4 from online-readiness exports.")
    parser.add_argument("--frames_json", required=True)
    parser.add_argument("--readiness_json", required=True)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--frame_dir", default="")
    parser.add_argument("--output_frame_dir", default="")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--title", default="BOLT Online Adaptation Readiness")
    parser.add_argument("--subtitle", default="DAIR-V2X  |  PP->SECOND  |  single-pass online TTT")
    parser.add_argument("--max_frames", type=int, default=0)
    return parser.parse_args()


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_panel_path(frame_record: Dict, frames_json_path: str, override_frame_dir: str) -> str:
    if override_frame_dir:
        return os.path.join(override_frame_dir, os.path.basename(frame_record["panel_path"]))
    if os.path.isabs(frame_record["panel_path"]):
        return frame_record["panel_path"]
    return os.path.join(os.path.dirname(frames_json_path), frame_record["panel_path"])


def filter_points_upto(aligned_points: List[Dict], sample_index: int) -> List[Dict]:
    return [point for point in aligned_points if int(point["sample"]) <= int(sample_index)]


def main() -> None:
    args = parse_args()
    frames_payload = load_json(args.frames_json)
    readiness_payload = load_json(args.readiness_json)
    frame_records = frames_payload["frames"]
    if args.max_frames > 0:
        frame_records = frame_records[: args.max_frames]

    out_frame_dir = args.output_frame_dir or os.path.join(
        os.path.dirname(args.output_video),
        os.path.splitext(os.path.basename(args.output_video))[0] + "_frames",
    )
    os.makedirs(out_frame_dir, exist_ok=True)

    aligned_points = readiness_payload["aligned_points"]
    rendered_paths = []
    for export_index, frame_record in enumerate(frame_records, start=1):
        panel_path = resolve_panel_path(frame_record, args.frames_json, args.frame_dir)
        compare_panel = Image.open(panel_path).convert("RGB")
        visible_points = filter_points_upto(aligned_points, frame_record["sample_index"])
        convergence_plot = render_convergence_plot(
            visible_points,
            width=1180,
            height=360,
        )
        dashboard = render_dashboard_frame(
            compare_panel=compare_panel,
            convergence_plot=convergence_plot,
            frame_record=frame_record,
            readiness_summary=readiness_payload,
            title=args.title,
            subtitle=args.subtitle,
        )
        frame_path = os.path.join(out_frame_dir, f"dashboard_{export_index:04d}.png")
        dashboard.save(frame_path)
        rendered_paths.append(frame_path)

    output_video_dir = os.path.dirname(args.output_video)
    if output_video_dir:
        os.makedirs(output_video_dir, exist_ok=True)
    write_mp4_from_frames(rendered_paths, args.output_video, fps=args.fps)
    print(f"Saved readiness video to {args.output_video}")


if __name__ == "__main__":
    main()
