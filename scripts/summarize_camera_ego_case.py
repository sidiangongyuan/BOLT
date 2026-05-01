#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one camera-ego limitation case from existing eval files."
    )
    parser.add_argument("--ego", required=True, help="Ego-only result dir or eval YAML")
    parser.add_argument("--noplugin", required=True, help="No-plugin result dir or eval YAML")
    parser.add_argument("--plugin", required=True, help="Plugin result dir or eval YAML")
    parser.add_argument("--label", default="DAIR LSS-E->PP")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def resolve_eval_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path

    direct_candidates = [path / "eval_online_ttt.yaml", path / "eval.yaml"]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    pattern_candidates = sorted(
        [
            candidate
            for candidate in path.glob("eval_*.yaml")
            if not any(tag in candidate.stem for tag in ("_short", "_middle", "_long"))
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pattern_candidates:
        return pattern_candidates[0]

    raise FileNotFoundError(f"Cannot resolve eval YAML from {path}")


def load_metrics(eval_file: Path) -> dict[str, Any]:
    data = yaml.safe_load(eval_file.read_text())
    return {
        "eval_file": str(eval_file),
        "ap30": float(data["ap30"]),
        "ap50": float(data["ap_50"]),
        "ap70": float(data["ap_70"]),
    }


def main() -> None:
    args = parse_args()
    payload = {
        "label": args.label,
        "ego_only": load_metrics(resolve_eval_file(args.ego)),
        "w_o_plugin": load_metrics(resolve_eval_file(args.noplugin)),
        "w_plugin": load_metrics(resolve_eval_file(args.plugin)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"[camera-ego] {args.label}")
    print(
        "[camera-ego] "
        f"ego-only {payload['ego_only']['ap30']:.4f}/{payload['ego_only']['ap50']:.4f}/"
        f"{payload['ego_only']['ap70']:.4f}"
    )
    print(
        "[camera-ego] "
        f"w/o plugin {payload['w_o_plugin']['ap30']:.4f}/{payload['w_o_plugin']['ap50']:.4f}/"
        f"{payload['w_o_plugin']['ap70']:.4f}"
    )
    print(
        "[camera-ego] "
        f"w/ plugin {payload['w_plugin']['ap30']:.4f}/{payload['w_plugin']['ap50']:.4f}/"
        f"{payload['w_plugin']['ap70']:.4f}"
    )
    print(f"[camera-ego] saved summary to {output_path}")


if __name__ == "__main__":
    main()
