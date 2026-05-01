#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two AP@50 precision-recall curves from eval YAML files."
    )
    parser.add_argument("--baseline", required=True, help="Dir or eval YAML for baseline/no-boost result")
    parser.add_argument("--default", required=True, help="Dir or eval YAML for default/boost result")
    parser.add_argument("--baseline_label", default="Filtered Distillation")
    parser.add_argument("--default_label", default="+ Enhancement (Default)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def resolve_eval_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    for name in ("eval_online_ttt.yaml", "eval.yaml"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find eval YAML under {path}")


def load_curve(eval_file: Path) -> dict[str, Any]:
    data = yaml.safe_load(eval_file.read_text())
    recall = [float(x) for x in data["mrec_50"]]
    precision = [float(x) for x in data["mpre_50"]]
    if len(recall) != len(precision):
        raise ValueError(f"Recall/precision length mismatch in {eval_file}")
    return {
        "eval_file": str(eval_file),
        "ap50": float(data["ap_50"]),
        "recall": recall,
        "precision": precision,
    }


def best_f1_point(recall: list[float], precision: list[float]) -> dict[str, float]:
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "index": 0}
    for idx, (r, p) in enumerate(zip(recall, precision)):
        denom = r + p
        f1 = 0.0 if denom <= 0 else 2.0 * r * p / denom
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": p, "recall": r, "index": idx}
    return best


def save_curve_csv(curves: list[dict[str, Any]], output_csv: Path) -> None:
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "point_index", "recall", "precision"])
        for curve in curves:
            for idx, (recall, precision) in enumerate(zip(curve["recall"], curve["precision"])):
                writer.writerow([curve["label"], idx, recall, precision])


def maybe_plot(curves: list[dict[str, Any]], output_dir: Path, title: str) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not available; skipped figure export"

    plt.figure(figsize=(5.2, 4.2))
    for curve in curves:
        plt.plot(curve["recall"], curve["precision"], label=curve["label"], linewidth=2.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    output_png = output_dir / "pr_curve_ap50.png"
    output_pdf = output_dir / "pr_curve_ap50.pdf"
    plt.savefig(output_png, dpi=200)
    plt.savefig(output_pdf)
    plt.close()
    return f"saved figure to {output_png} and {output_pdf}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_curve(resolve_eval_file(args.baseline))
    baseline["label"] = args.baseline_label
    baseline["best_f1"] = best_f1_point(baseline["recall"], baseline["precision"])

    default = load_curve(resolve_eval_file(args.default))
    default["label"] = args.default_label
    default["best_f1"] = best_f1_point(default["recall"], default["precision"])

    curves = [baseline, default]
    save_curve_csv(curves, output_dir / "pr_curve_ap50.csv")
    plot_status = maybe_plot(curves, output_dir, args.title)

    summary = {
        "baseline": {
            "label": baseline["label"],
            "eval_file": baseline["eval_file"],
            "ap50": baseline["ap50"],
            "best_f1": baseline["best_f1"],
        },
        "default": {
            "label": default["label"],
            "eval_file": default["eval_file"],
            "ap50": default["ap50"],
            "best_f1": default["best_f1"],
        },
        "delta_ap50": default["ap50"] - baseline["ap50"],
        "delta_best_f1": default["best_f1"]["f1"] - baseline["best_f1"]["f1"],
        "plot_status": plot_status,
    }
    summary_path = output_dir / "pr_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        f"[pr] {baseline['label']}: AP@50={baseline['ap50']:.4f}, "
        f"best-F1 P/R=({baseline['best_f1']['precision']:.4f}, "
        f"{baseline['best_f1']['recall']:.4f})"
    )
    print(
        f"[pr] {default['label']}: AP@50={default['ap50']:.4f}, "
        f"best-F1 P/R=({default['best_f1']['precision']:.4f}, "
        f"{default['best_f1']['recall']:.4f})"
    )
    print(f"[pr] delta AP@50={summary['delta_ap50']:.4f}")
    print(f"[pr] {plot_status}")
    print(f"[pr] saved summary to {summary_path}")


if __name__ == "__main__":
    main()
