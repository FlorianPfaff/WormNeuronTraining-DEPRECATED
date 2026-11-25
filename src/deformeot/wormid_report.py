"""Summarize WormID reference-tracking evidence at the recording level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _bootstrap_mean(
    values: np.ndarray, *, samples: int, rng: np.random.Generator
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "recordings": int(len(values)),
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": int(samples),
    }


def summarize_benchmark(
    payload: dict[str, object],
    *,
    samples: int = 20_000,
    seed: int = 20260720,
    improved: str = "dual_anchor",
    baseline: str = "first_affine",
) -> dict[str, object]:
    """Calculate held-out recording bootstrap intervals, never frame intervals."""

    raw_rows = payload.get("results")
    if not isinstance(raw_rows, list):
        raise ValueError("benchmark payload has no result rows")
    rows = [row for row in raw_rows if row.get("split") == "held_out"]
    recording_ids = sorted({str(row["recording_id"]) for row in rows})
    strides = sorted({int(row["stride"]) for row in rows})
    methods = tuple(str(value) for value in payload["methods"])
    rng = np.random.default_rng(seed)

    aggregate: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    for stride in strides:
        selected = [row for row in rows if int(row["stride"]) == stride]
        for method in methods:
            method_rows = [row for row in selected if row["method"] == method]
            accuracy = np.asarray([row["identity_accuracy"] for row in method_rows])
            severe = np.asarray([row["severe_identity_accuracy"] for row in method_rows])
            aggregate.append(
                {
                    "stride": stride,
                    "method": method,
                    "identity_accuracy": _bootstrap_mean(
                        accuracy, samples=samples, rng=rng
                    ),
                    "severe_identity_accuracy": _bootstrap_mean(
                        severe, samples=samples, rng=rng
                    ),
                    "micro_identity_accuracy": float(
                        sum(int(row["correct"]) for row in method_rows)
                        / sum(int(row["total"]) for row in method_rows)
                    ),
                    "identity_switches": int(
                        sum(int(row["identity_switches"]) for row in method_rows)
                    ),
                    "severe_frames": int(
                        sum(int(row["severe_frames"]) for row in method_rows)
                    ),
                    "eligible_transition_frames": int(
                        sum(int(row["frames"]) - 1 for row in method_rows)
                    ),
                    "runtime_seconds": float(
                        sum(float(row["runtime_seconds"]) for row in method_rows)
                    ),
                }
            )

        indexed = {
            (str(row["recording_id"]), str(row["method"])): row for row in selected
        }
        accuracy_difference = []
        severe_difference = []
        switch_rate_difference = []
        for recording_id in recording_ids:
            better = indexed[(recording_id, improved)]
            control = indexed[(recording_id, baseline)]
            accuracy_difference.append(
                float(better["identity_accuracy"]) - float(control["identity_accuracy"])
            )
            severe_difference.append(
                float(better["severe_identity_accuracy"])
                - float(control["severe_identity_accuracy"])
            )
            better_edges = (int(better["frames"]) - 1) * int(better["neurons"])
            control_edges = (int(control["frames"]) - 1) * int(control["neurons"])
            switch_rate_difference.append(
                int(better["identity_switches"]) / better_edges
                - int(control["identity_switches"]) / control_edges
            )
        paired.append(
            {
                "stride": stride,
                "contrast": f"{improved}_minus_{baseline}",
                "identity_accuracy_difference": _bootstrap_mean(
                    np.asarray(accuracy_difference), samples=samples, rng=rng
                ),
                "severe_identity_accuracy_difference": _bootstrap_mean(
                    np.asarray(severe_difference), samples=samples, rng=rng
                ),
                "identity_switch_rate_difference": _bootstrap_mean(
                    np.asarray(switch_rate_difference), samples=samples, rng=rng
                ),
            }
        )

    # One independent value per held-out recording: average its result across
    # the five frozen temporal gaps, then resample complete recordings.
    macro_accuracy_difference = []
    macro_severe_difference = []
    for recording_id in recording_ids:
        better = [
            row
            for row in rows
            if row["recording_id"] == recording_id and row["method"] == improved
        ]
        control = [
            row
            for row in rows
            if row["recording_id"] == recording_id and row["method"] == baseline
        ]
        macro_accuracy_difference.append(
            float(np.mean([row["identity_accuracy"] for row in better]))
            - float(np.mean([row["identity_accuracy"] for row in control]))
        )
        macro_severe_difference.append(
            float(np.mean([row["severe_identity_accuracy"] for row in better]))
            - float(np.mean([row["severe_identity_accuracy"] for row in control]))
        )

    return {
        "schema_version": 1,
        "source_schema_version": payload.get("schema_version"),
        "held_out_recording_ids": recording_ids,
        "strides": strides,
        "bootstrap_seed": seed,
        "aggregate": aggregate,
        "paired_by_stride": paired,
        "paired_macro_across_strides": {
            "contrast": f"{improved}_minus_{baseline}",
            "identity_accuracy_difference": _bootstrap_mean(
                np.asarray(macro_accuracy_difference), samples=samples, rng=rng
            ),
            "severe_identity_accuracy_difference": _bootstrap_mean(
                np.asarray(macro_severe_difference), samples=samples, rng=rng
            ),
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    summary = summarize_benchmark(payload, samples=args.samples, seed=args.seed)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
