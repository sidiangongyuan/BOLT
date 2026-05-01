"""Analyze dynamic online-readiness milestones from aligned convergence curves."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.readiness_vis_utils import plot_ap50_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dynamic online-readiness milestones from aligned convergence logs."
    )
    parser.add_argument("--plugin_convergence_json", required=True)
    parser.add_argument("--ego_only_convergence_json", required=True)
    parser.add_argument("--no_plugin_convergence_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--output_plot",
        action="append",
        default=[],
        help="Optional plot path. Provide multiple times to save PNG/PDF variants.",
    )
    parser.add_argument("--sustained_window", type=int, default=3)
    parser.add_argument("--online_step_ms", type=float, default=0.0)
    parser.add_argument("--title", default="BOLT Online Readiness")
    return parser.parse_args()


def load_convergence_points(convergence_json: str) -> List[Dict[str, float]]:
    with open(convergence_json, "r", encoding="utf-8") as handle:
        raw_points = json.load(handle)

    return [
        {
            "sample": int(sample),
            "ap30": float(ap30) * 100.0,
            "ap50": float(ap50) * 100.0,
            "ap70": float(ap70) * 100.0,
        }
        for sample, ap30, ap50, ap70 in raw_points
    ]


def align_points(
    plugin_points: Sequence[Dict[str, float]],
    ego_only_points: Sequence[Dict[str, float]],
    no_plugin_points: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    if not plugin_points or not ego_only_points or not no_plugin_points:
        raise ValueError("All three convergence logs must contain at least one checkpoint.")

    if not (
        len(plugin_points) == len(ego_only_points) == len(no_plugin_points)
    ):
        raise ValueError(
            "Convergence logs must have the same number of checkpoints: "
            f"plugin={len(plugin_points)}, ego_only={len(ego_only_points)}, "
            f"no_plugin={len(no_plugin_points)}"
        )

    aligned_points: List[Dict[str, float]] = []
    for plugin_point, ego_point, no_plugin_point in zip(
        plugin_points,
        ego_only_points,
        no_plugin_points,
    ):
        samples = {
            int(plugin_point["sample"]),
            int(ego_point["sample"]),
            int(no_plugin_point["sample"]),
        }
        if len(samples) != 1:
            raise ValueError(
                "Convergence checkpoints do not align: "
                f"plugin={plugin_point['sample']}, "
                f"ego_only={ego_point['sample']}, "
                f"no_plugin={no_plugin_point['sample']}"
            )

        sample = int(plugin_point["sample"])
        plugin_ap50 = float(plugin_point["ap50"])
        ego_only_ap50 = float(ego_point["ap50"])
        no_plugin_ap50 = float(no_plugin_point["ap50"])
        aligned_points.append(
            {
                "sample": sample,
                "plugin_ap50": plugin_ap50,
                "ego_only_ap50": ego_only_ap50,
                "no_plugin_ap50": no_plugin_ap50,
                "margin_vs_ego_only": plugin_ap50 - ego_only_ap50,
                "margin_vs_no_plugin": plugin_ap50 - no_plugin_ap50,
            }
        )
    return aligned_points


def compute_crossings(
    aligned_points: Sequence[Dict[str, float]],
    baseline_key: str,
    margin_key: str,
    sustained_window: int,
) -> Dict[str, Optional[Dict]]:
    first_above = None
    for point in aligned_points:
        if point["plugin_ap50"] >= point[baseline_key]:
            first_above = point
            break

    stable_above = None
    for index in range(len(aligned_points) - sustained_window + 1):
        window = aligned_points[index : index + sustained_window]
        if all(point["plugin_ap50"] >= point[baseline_key] for point in window):
            stable_above = {
                "sample": int(window[0]["sample"]),
                "window_end_sample": int(window[-1]["sample"]),
                "plugin_window": [float(point["plugin_ap50"]) for point in window],
                "reference_window": [float(point[baseline_key]) for point in window],
                "margin_window": [float(point[margin_key]) for point in window],
            }
            break

    return {
        "first_above": {
            "sample": int(first_above["sample"]),
            "plugin_ap50": float(first_above["plugin_ap50"]),
            "reference_ap50": float(first_above[baseline_key]),
            "margin_ap50": float(first_above[margin_key]),
        }
        if first_above is not None
        else None,
        "stable_above": stable_above,
    }


def add_time_fields(summary: Dict, online_step_ms: float) -> Dict:
    if online_step_ms <= 0:
        return summary
    for record in summary.values():
        for key in ("first_above", "stable_above"):
            entry = record.get(key)
            if entry is None:
                continue
            entry["seconds"] = float(entry["sample"] * online_step_ms / 1000.0)
            if "window_end_sample" in entry:
                entry["window_end_seconds"] = float(
                    entry["window_end_sample"] * online_step_ms / 1000.0
                )
    return summary


def render_plot(
    aligned_points: Sequence[Dict[str, float]],
    out_path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    plot_ap50_curves(ax, aligned_points, highlight_latest=False)
    ax.set_title(title)
    ax.set_xlabel("Frames processed")
    ax.set_ylabel("Running AP@50")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plugin_points = load_convergence_points(args.plugin_convergence_json)
    ego_only_points = load_convergence_points(args.ego_only_convergence_json)
    no_plugin_points = load_convergence_points(args.no_plugin_convergence_json)
    aligned_points = align_points(plugin_points, ego_only_points, no_plugin_points)

    milestones = {
        "ego_only": compute_crossings(
            aligned_points,
            baseline_key="ego_only_ap50",
            margin_key="margin_vs_ego_only",
            sustained_window=args.sustained_window,
        ),
        "no_plugin": compute_crossings(
            aligned_points,
            baseline_key="no_plugin_ap50",
            margin_key="margin_vs_no_plugin",
            sustained_window=args.sustained_window,
        ),
    }
    milestones = add_time_fields(milestones, args.online_step_ms)

    checkpoint_interval = None
    if len(aligned_points) >= 2:
        checkpoint_interval = aligned_points[1]["sample"] - aligned_points[0]["sample"]

    last_point = aligned_points[-1]
    payload = {
        "title": args.title,
        "plugin_convergence_json": args.plugin_convergence_json,
        "ego_only_convergence_json": args.ego_only_convergence_json,
        "no_plugin_convergence_json": args.no_plugin_convergence_json,
        "num_points": len(aligned_points),
        "checkpoint_interval": checkpoint_interval,
        "sustained_window": int(args.sustained_window),
        "online_step_ms": float(args.online_step_ms),
        "aligned_points": aligned_points,
        "milestones": milestones,
        "final_ap50": {
            "plugin": float(last_point["plugin_ap50"]),
            "ego_only": float(last_point["ego_only_ap50"]),
            "no_plugin": float(last_point["no_plugin_ap50"]),
        },
    }
    output_json_dir = os.path.dirname(args.output_json)
    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved readiness summary to {args.output_json}")

    for out_path in args.output_plot:
        output_plot_dir = os.path.dirname(out_path)
        if output_plot_dir:
            os.makedirs(output_plot_dir, exist_ok=True)
        render_plot(aligned_points, out_path, args.title)
        print(f"Saved readiness plot to {out_path}")


if __name__ == "__main__":
    main()
