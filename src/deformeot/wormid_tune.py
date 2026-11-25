"""Freeze the dual-anchor weight on one designated WormID recording."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .wormid import identity_blind_sequence, load_wormid_recording
from .wormid_benchmark import ReferenceTrackerConfig, score_tracks, track_sequence


def tune_dual_anchor(
    recording_path: str | Path,
    *,
    weights: Sequence[float],
    strides: Sequence[int],
    output_path: str | Path,
    seed: int = 20260720,
) -> dict[str, object]:
    """Select by macro gap accuracy; stable identities are evaluation-only."""

    if not weights or not strides:
        raise ValueError("weights and strides cannot be empty")
    recording = load_wormid_recording(recording_path)
    rows: list[dict[str, float | int]] = []
    for weight in weights:
        for stride in strides:
            selected = tuple(range(0, recording.frames, int(stride)))
            if selected[-1] != recording.frames - 1:
                selected = (*selected, recording.frames - 1)
            sequence = identity_blind_sequence(
                recording, frame_indices=selected, seed=seed + int(stride)
            )
            config = ReferenceTrackerConfig(
                reference_weight=float(weight), descriptor_weight=0.0
            )
            metrics = score_tracks(
                sequence, track_sequence(sequence, "dual_anchor", config=config)
            )
            rows.append(
                {
                    "reference_weight": float(weight),
                    "stride": int(stride),
                    "identity_accuracy": metrics.identity_accuracy,
                    "severe_identity_accuracy": metrics.severe_identity_accuracy,
                    "identity_switches": metrics.identity_switches,
                    "runtime_seconds": metrics.runtime_seconds,
                }
            )

    candidates = []
    for weight in weights:
        selected_rows = [row for row in rows if row["reference_weight"] == weight]
        candidates.append(
            {
                "reference_weight": float(weight),
                "macro_identity_accuracy": float(
                    np.mean([row["identity_accuracy"] for row in selected_rows])
                ),
                "macro_severe_identity_accuracy": float(
                    np.mean([row["severe_identity_accuracy"] for row in selected_rows])
                ),
                "identity_switches": int(
                    sum(int(row["identity_switches"]) for row in selected_rows)
                ),
            }
        )
    # The smaller weight wins an exact metric tie, making selection deterministic.
    chosen = max(candidates, key=lambda row: (row["macro_identity_accuracy"], -row["reference_weight"]))
    payload: dict[str, object] = {
        "schema_version": 1,
        "recording_id": recording.recording_id,
        "recording_path": recording.path,
        "seed": seed,
        "strides": [int(value) for value in strides],
        "selection_metric": "macro_identity_accuracy_across_strides",
        "identity_labels_used_for_association": False,
        "identity_labels_used_for_hyperparameter_evaluation": True,
        "base_config": asdict(ReferenceTrackerConfig()),
        "candidates": candidates,
        "rows": rows,
        "selected_reference_weight": chosen["reference_weight"],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--strides", nargs="+", type=int, default=[5, 10, 20, 40])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = tune_dual_anchor(
        args.recording,
        weights=args.weights,
        strides=args.strides,
        output_path=args.output,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
