"""Audit identity switches inside the exact high-cloud-motion stratum.

The primary WormID report already scores identity accuracy in an exact top
quartile of identity-blind cloud motion, but its switch counts aggregate all
transitions.  This module applies the same exact, deterministic transition
selection to switch counts so the prespecified severe-stratum criterion can be
read literally.  No tracker parameter is selected here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .wormid import IdentityBlindSequence, identity_blind_sequence, load_wormid_recording
from .wormid_benchmark import ReferenceTrackerConfig, TrackResult, track_sequence


METHODS = ("first_affine", "dual_anchor")


def benchmark_frame_indices(frame_count: int, stride: int) -> tuple[int, ...]:
    """Select regular frames and retain the final frame exactly as the main benchmark."""

    if frame_count < 2:
        raise ValueError("at least two frames are required")
    if stride < 1:
        raise ValueError("stride must be positive")
    selected = tuple(range(0, frame_count, stride))
    return selected if selected[-1] == frame_count - 1 else (*selected, frame_count - 1)


def identity_matrix(sequence: IdentityBlindSequence, result: TrackResult) -> np.ndarray:
    """Recover seeded hidden identities for evaluation only."""

    if len(sequence.frames) != len(result.assignments):
        raise ValueError("sequence and result lengths differ")
    canonical = np.asarray(sequence.truth_indices[0], dtype=int)
    rows = []
    for hidden_order, assignment in zip(
        sequence.truth_indices, result.assignments, strict=True
    ):
        assignment = np.asarray(assignment, dtype=int)
        if assignment.shape != canonical.shape or np.any(assignment < 0):
            raise ValueError("assignments must exhaust the seeded cohort")
        rows.append(np.asarray(hidden_order, dtype=int)[assignment])
    return np.stack(rows)


def exact_high_motion_mask(
    result: TrackResult,
    source_frame_indices: Sequence[int],
    *,
    fraction: float = 0.25,
) -> np.ndarray:
    """Select exactly the highest-motion transition fraction with fixed tie breaking."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    motion = np.asarray(result.cloud_motion_um[1:], dtype=float)
    source_indices = np.asarray(source_frame_indices[1:], dtype=int)
    if motion.shape != source_indices.shape or not len(motion):
        raise ValueError("motion and source-frame indices must align")
    selected_count = max(1, int(np.ceil(fraction * len(motion))))
    ranking = np.lexsort((source_indices, -motion))
    mask = np.zeros(len(motion), dtype=bool)
    mask[ranking[:selected_count]] = True
    return mask


def score_switch_stratum(
    sequence: IdentityBlindSequence,
    result: TrackResult,
    transition_mask: np.ndarray,
) -> dict[str, int | float]:
    """Count switches and opportunities only on selected transitions."""

    identities = identity_matrix(sequence, result)
    changes = np.diff(identities, axis=0) != 0
    mask = np.asarray(transition_mask, dtype=bool)
    if mask.shape != (len(sequence.frames) - 1,):
        raise ValueError("transition mask has the wrong shape")
    selected = changes[mask]
    switches = int(np.count_nonzero(selected))
    opportunities = int(selected.size)
    return {
        "switches": switches,
        "opportunities": opportunities,
        "switch_rate": float(switches / opportunities),
        "selected_transitions": int(np.count_nonzero(mask)),
    }


def _bootstrap_rate_difference(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    differences = np.asarray(values, dtype=float)
    if differences.ndim != 1 or not len(differences) or np.any(~np.isfinite(differences)):
        raise ValueError("finite recording differences are required")
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        differences,
        size=(samples, len(differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "recordings": int(len(differences)),
        "mean_switch_rate_reduction": float(np.mean(differences)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": int(samples),
        "per_recording_differences": differences.tolist(),
    }


def run_severe_switch_audit(
    paths: Sequence[str | Path],
    *,
    output_path: str | Path,
    tuning_recording_id: str,
    strides: Sequence[int] = (1, 5, 10, 20, 40),
    seed: int = 20260720,
    bootstrap_samples: int = 10_000,
    config: ReferenceTrackerConfig = ReferenceTrackerConfig(
        reference_weight=6.0,
        descriptor_weight=0.0,
    ),
) -> dict[str, object]:
    """Run frozen geometric trackers and audit switches in severe transitions."""

    recordings = tuple(load_wormid_recording(path) for path in paths)
    if not recordings:
        raise ValueError("at least one recording is required")
    held_out = tuple(
        recording for recording in recordings if recording.recording_id != tuning_recording_id
    )
    if len(held_out) != len(recordings) - 1:
        raise ValueError("tuning_recording_id must identify exactly one input")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    input_index = {recording.recording_id: index for index, recording in enumerate(recordings)}
    rows: list[dict[str, object]] = []
    for recording in held_out:
        path_index = input_index[recording.recording_id]
        for stride in strides:
            selected = benchmark_frame_indices(recording.frames, int(stride))
            sequence = identity_blind_sequence(
                recording,
                frame_indices=selected,
                seed=seed + 1000 * path_index + int(stride),
            )
            results = {
                method: track_sequence(sequence, method, config=config) for method in METHODS
            }
            masks = {
                method: exact_high_motion_mask(result, selected)
                for method, result in results.items()
            }
            if not np.array_equal(masks[METHODS[0]], masks[METHODS[1]]):
                raise AssertionError("cloud-motion selection must be assignment invariant")
            for method, result in results.items():
                row: dict[str, object] = score_switch_stratum(
                    sequence,
                    result,
                    masks[method],
                )
                row.update(
                    recording_id=recording.recording_id,
                    method=method,
                    stride=int(stride),
                    frames_evaluated=len(selected),
                    motion_threshold_um=float(
                        np.min(result.cloud_motion_um[1:][masks[method]])
                    ),
                )
                rows.append(row)

    by_stride: dict[str, dict[str, object]] = {}
    for stride in strides:
        selected_rows = [row for row in rows if row["stride"] == int(stride)]
        totals = {
            method: sum(
                int(row["switches"])
                for row in selected_rows
                if row["method"] == method
            )
            for method in METHODS
        }
        opportunities = sum(
            int(row["opportunities"])
            for row in selected_rows
            if row["method"] == METHODS[0]
        )
        by_stride[str(stride)] = {
            "first_affine_switches": totals["first_affine"],
            "dual_anchor_switches": totals["dual_anchor"],
            "opportunities": opportunities,
            "relative_switch_reduction": float(
                (totals["first_affine"] - totals["dual_anchor"])
                / totals["first_affine"]
            ),
            "switch_rate_reduction": float(
                (totals["first_affine"] - totals["dual_anchor"]) / opportunities
            ),
        }

    recording_differences = []
    recording_rows: dict[str, dict[str, object]] = {}
    for recording in held_out:
        method_totals: dict[str, dict[str, int | float]] = {}
        for method in METHODS:
            selected_rows = [
                row
                for row in rows
                if row["recording_id"] == recording.recording_id
                and row["method"] == method
            ]
            switches = sum(int(row["switches"]) for row in selected_rows)
            opportunities = sum(int(row["opportunities"]) for row in selected_rows)
            method_totals[method] = {
                "switches": switches,
                "opportunities": opportunities,
                "switch_rate": switches / opportunities,
            }
        difference = float(
            method_totals["first_affine"]["switch_rate"]
            - method_totals["dual_anchor"]["switch_rate"]
        )
        recording_differences.append(difference)
        recording_rows[recording.recording_id] = {
            **method_totals,
            "switch_rate_reduction": difference,
        }

    all_base = sum(int(row["switches"]) for row in rows if row["method"] == METHODS[0])
    all_better = sum(int(row["switches"]) for row in rows if row["method"] == METHODS[1])
    payload: dict[str, object] = {
        "schema_version": 1,
        "analysis_role": (
            "literal severe-stratum audit using the already specified exact top-quartile "
            "identity-blind cloud-motion rule; computed after the initial aggregate report"
        ),
        "tracker_parameters_selected_here": False,
        "seed": seed,
        "seed_rule": "seed + 1000*input_path_index + stride",
        "tuning_recording_id": tuning_recording_id,
        "held_out_recording_ids": [recording.recording_id for recording in held_out],
        "strides": [int(stride) for stride in strides],
        "methods": list(METHODS),
        "geometry_config": asdict(config),
        "stratum": {
            "definition": (
                "exact top quartile of identity-blind symmetric cloud motion within "
                "each recording and stride"
            ),
            "tie_break": "earlier source-frame index",
        },
        "rows": rows,
        "pooled_by_stride": by_stride,
        "recording_summaries": recording_rows,
        "complete_recording_bootstrap": _bootstrap_rate_difference(
            recording_differences,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "pooled_across_strides": {
            "first_affine_switches": all_base,
            "dual_anchor_switches": all_better,
            "relative_switch_reduction": float((all_base - all_better) / all_base),
        },
        "target_audit": {
            "criterion": "at least 25% fewer switches in a severe stratum",
            "minimum_stride_reduction": min(
                float(row["relative_switch_reduction"]) for row in by_stride.values()
            ),
            "all_strides_pass_25_percent": all(
                float(row["relative_switch_reduction"]) >= 0.25
                for row in by_stride.values()
            ),
            "note": (
                "The stratum and threshold were prespecified, but this stratum-specific "
                "switch aggregation was added after the first held-out report."
            ),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="WormID EY NWB recordings")
    parser.add_argument("--tuning-recording-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_severe_switch_audit(
        args.paths,
        output_path=args.output,
        tuning_recording_id=args.tuning_recording_id,
        strides=args.strides,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(payload["target_audit"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
