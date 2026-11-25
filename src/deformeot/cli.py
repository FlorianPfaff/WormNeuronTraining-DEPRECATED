"""Command-line interface for reproducible DeformEOT experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .data import read_tracks, track_summary
from .metrics import evaluate_partial_tracks


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return data


def evaluate_ascent(config_path: str | Path) -> dict:
    config = _load_yaml(config_path)
    dataset = config["dataset"]
    evaluation = config["evaluation"]
    gt = read_tracks(dataset["gt_tracks"])
    pred = read_tracks(dataset["pred_tracks"])
    metrics, _ = evaluate_partial_tracks(
        gt,
        pred,
        spacing_zyx=dataset["spacing_zyx_um"],
        threshold_um=float(evaluation["distance_threshold_um"]),
    )
    return {
        "protocol": "ASCENT partial-label mutual-nearest modified-MOTA",
        "ground_truth": track_summary(gt),
        "prediction": track_summary(pred),
        "metrics": metrics.to_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deformeot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ascent = subparsers.add_parser("evaluate-ascent")
    ascent.add_argument("--config", default="configs/ascent.yaml")
    ascent.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate-ascent":
        result = evaluate_ascent(args.config)
    else:  # pragma: no cover - argparse enforces known subcommands.
        raise AssertionError(args.command)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

