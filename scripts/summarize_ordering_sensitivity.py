#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ordering-sensitivity runs from eval_online_ttt.yaml files."
    )
    parser.add_argument(
        "--glob",
        required=True,
        help="Glob pattern for run directories, e.g. /path/to/ordering_pp_second_seed*_comm100",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path. Defaults to <glob_parent>/ordering_summary.json",
    )
    return parser.parse_args()


def resolve_eval_file(result_dir: Path) -> Path:
    for name in ("eval_online_ttt.yaml", "eval.yaml"):
        candidate = result_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find eval YAML in {result_dir}")


def load_metrics(eval_file: Path) -> dict[str, float]:
    data = yaml.safe_load(eval_file.read_text())
    return {
        "ap30": float(data["ap30"]),
        "ap50": float(data["ap_50"]),
        "ap70": float(data["ap_70"]),
    }


def load_order_seed(result_dir: Path) -> int | None:
    order_file = result_dir / "test_order.json"
    if not order_file.exists():
        return None
    data = json.loads(order_file.read_text())
    return int(data.get("test_order_seed", -1))


def summarize(values: list[float]) -> dict[str, float]:
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.mean(values), "std": std}


def main() -> None:
    args = parse_args()
    result_dirs = [Path(path) for path in sorted(glob.glob(args.glob))]
    if not result_dirs:
        raise FileNotFoundError(f"No result directories matched: {args.glob}")

    runs: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        eval_file = resolve_eval_file(result_dir)
        metrics = load_metrics(eval_file)
        runs.append(
            {
                "result_dir": str(result_dir),
                "eval_file": str(eval_file),
                "test_order_seed": load_order_seed(result_dir),
                **metrics,
            }
        )

    summary = {
        "num_runs": len(runs),
        "runs": runs,
        "ap30": summarize([run["ap30"] for run in runs]),
        "ap50": summarize([run["ap50"] for run in runs]),
        "ap70": summarize([run["ap70"] for run in runs]),
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = result_dirs[0].parent / "ordering_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    print(f"[ordering] runs={summary['num_runs']}")
    print(
        "[ordering] "
        f"AP@30 {summary['ap30']['mean']:.4f} ± {summary['ap30']['std']:.4f} | "
        f"AP@50 {summary['ap50']['mean']:.4f} ± {summary['ap50']['std']:.4f} | "
        f"AP@70 {summary['ap70']['mean']:.4f} ± {summary['ap70']['std']:.4f}"
    )
    print(f"[ordering] saved summary to {output_path}")


if __name__ == "__main__":
    main()
