"""Post-hoc sensitivity to the definition of a dense neuron neighborhood.

The preregistration described density using a nucleus-radius idea but did not
define a measurable nucleus radius.  The primary WormID extent analysis instead
used the calibration animal's pooled nearest-neighbor Q25.  This module audits
that protocol deviation without changing any association decision: the frozen
dual-anchor and reliability-gated extent trackers are each run once per
recording/stride block, and their resulting identities are rescored under a
prespecified grid of calibration-only distance cutoffs.

Every result produced here is a post-hoc protocol-deviation sensitivity.  It
cannot retroactively satisfy the preregistered density target.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from .wormid import IdentityBlindSequence, WormIDRecording, identity_blind_sequence
from .wormid import load_wormid_recording
from .wormid_benchmark import ReferenceTrackerConfig, TrackResult, track_sequence
from .wormid_extent import (
    DualAnchorExtentConfig,
    PatchFeatureSet,
    fit_robust_feature_scaler,
    load_patch_feature_cache,
    pair_patch_features_to_anonymous_detections,
    track_dual_anchor_extent,
)
from .wormid_extent import _benchmark_frame_indices as benchmark_frame_indices
from .wormid_extent import _subset_patch_features as subset_patch_features


ANALYSIS_ROLE = "post-hoc protocol-deviation sensitivity"
PROTOCOL_DEVIATION = (
    "The preregistration's nucleus-radius density concept was undefined, while the "
    "reported primary analysis used calibration nearest-neighbor Q25. These post-hoc "
    "sensitivity results cannot retroactively satisfy the preregistered density target."
)
DEFAULT_STRIDES = (1, 5, 10, 20, 40)
SIGMA_FEATURE_NAMES = (
    "log_std_small",
    "log_std_middle",
    "log_std_large",
)


@dataclass(frozen=True)
class DensityDefinition:
    """One calibration-only distance definition in the sensitivity grid."""

    definition_id: str
    label: str
    threshold_um: float
    source: str

    def __post_init__(self) -> None:
        if not self.definition_id:
            raise ValueError("definition_id cannot be empty")
        if not np.isfinite(self.threshold_um) or self.threshold_um <= 0:
            raise ValueError("threshold_um must be positive and finite")


@dataclass(frozen=True)
class TrackedDensityBlock:
    """Held-out correctness and geometry retained after one tracking pass."""

    recording_id: str
    stride: int
    nearest_neighbor_um: np.ndarray
    baseline_correct: np.ndarray
    extent_correct: np.ndarray

    def __post_init__(self) -> None:
        nearest = np.asarray(self.nearest_neighbor_um, dtype=float)
        baseline = np.asarray(self.baseline_correct, dtype=bool)
        extent = np.asarray(self.extent_correct, dtype=bool)
        if nearest.ndim != 1 or baseline.shape != nearest.shape or extent.shape != nearest.shape:
            raise ValueError("tracked block arrays must be aligned one-dimensional vectors")
        if len(nearest) == 0 or np.any(~np.isfinite(nearest)) or np.any(nearest < 0):
            raise ValueError("nearest-neighbor distances must be nonempty, finite, and nonnegative")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        object.__setattr__(self, "nearest_neighbor_um", nearest.copy())
        object.__setattr__(self, "baseline_correct", baseline.copy())
        object.__setattr__(self, "extent_correct", extent.copy())


def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("points must have shape (objects, 3) with at least two objects")
    distances = cdist(values, values)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def calibration_density_definitions(
    recording: WormIDRecording,
    features: PatchFeatureSet,
) -> tuple[tuple[DensityDefinition, ...], dict[str, object]]:
    """Derive the complete sensitivity grid from one calibration animal only.

    Gaussian-axis standard deviations are encoded in the local extent
    descriptor.  The equivalent isotropic sigma is the geometric mean of the
    three axes, RMS sigma is the root mean square, and largest sigma is the
    largest covariance axis.  Only patches with nonzero reliability, the same
    criterion used to fit the gated calibration scaler, enter these medians.
    """

    if features.descriptors.shape[:2] != (recording.frames, recording.neurons):
        raise ValueError("calibration feature dimensions do not match the recording")
    try:
        sigma_indices = tuple(features.feature_names.index(name) for name in SIGMA_FEATURE_NAMES)
    except ValueError as error:
        raise ValueError("calibration features lack Gaussian-axis log standard deviations") from error

    reliable = np.asarray(features.reliability, dtype=float) > 0
    if not np.any(reliable):
        raise ValueError("calibration cache contains no reliable patches")
    sigmas = np.exp(features.descriptors[..., sigma_indices])
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("calibration Gaussian-axis standard deviations must be positive")

    equivalent_sigma = np.cbrt(np.prod(sigmas, axis=-1))
    rms_sigma = np.sqrt(np.mean(np.square(sigmas), axis=-1))
    largest_sigma = np.max(sigmas, axis=-1)
    proxy_medians = {
        "equivalent_isotropic_sigma_um": float(np.median(equivalent_sigma[reliable])),
        "rms_sigma_um": float(np.median(rms_sigma[reliable])),
        "largest_sigma_um": float(np.median(largest_sigma[reliable])),
    }

    nearest = np.concatenate(
        [_nearest_neighbor_distances(points) for points in recording.positions_zyx_um]
    )
    q25 = float(np.quantile(nearest, 0.25))
    definitions = (
        DensityDefinition("fixed_1_00_um", "fixed 1.00 um", 1.0, "fixed sensitivity grid"),
        DensityDefinition("fixed_1_25_um", "fixed 1.25 um", 1.25, "fixed sensitivity grid"),
        DensityDefinition(
            "calibration_nn_q25",
            "calibration nearest-neighbor Q25",
            q25,
            "pooled calibration-animal nearest-neighbor Q25",
        ),
        DensityDefinition("fixed_1_50_um", "fixed 1.50 um", 1.5, "fixed sensitivity grid"),
        DensityDefinition(
            "two_x_median_equivalent_sigma",
            "2 x median equivalent isotropic Gaussian sigma",
            2 * proxy_medians["equivalent_isotropic_sigma_um"],
            "reliable calibration patches only",
        ),
        DensityDefinition(
            "two_x_median_rms_sigma",
            "2 x median RMS Gaussian sigma",
            2 * proxy_medians["rms_sigma_um"],
            "reliable calibration patches only",
        ),
        DensityDefinition(
            "two_x_median_largest_sigma",
            "2 x median largest Gaussian sigma",
            2 * proxy_medians["largest_sigma_um"],
            "reliable calibration patches only",
        ),
        DensityDefinition("fixed_2_00_um", "fixed 2.00 um", 2.0, "fixed sensitivity grid"),
        DensityDefinition("fixed_2_50_um", "fixed 2.50 um", 2.5, "fixed sensitivity grid"),
        DensityDefinition("fixed_3_00_um", "fixed 3.00 um", 3.0, "fixed sensitivity grid"),
        DensityDefinition("fixed_3_50_um", "fixed 3.50 um", 3.5, "fixed sensitivity grid"),
        DensityDefinition("fixed_4_00_um", "fixed 4.00 um", 4.0, "fixed sensitivity grid"),
    )
    metadata: dict[str, object] = {
        "reliable_patch_definition": "reliability > 0",
        "reliable_patch_count": int(np.sum(reliable)),
        "total_patch_count": int(reliable.size),
        "sigma_axis_feature_names": list(SIGMA_FEATURE_NAMES),
        "equivalent_isotropic_sigma_formula": "(sigma_small*sigma_middle*sigma_large)^(1/3)",
        "rms_sigma_formula": "sqrt(mean(sigma_axis^2))",
        "diameter_proxy_multiplier": 2.0,
        "median_sigma_proxies_um": proxy_medians,
        "calibration_nearest_neighbor_q25_um": q25,
        "calibration_nearest_neighbor_events": int(len(nearest)),
    }
    return definitions, metadata


def tracked_density_block(
    recording: WormIDRecording,
    sequence: IdentityBlindSequence,
    baseline_result: TrackResult,
    extent_result: TrackResult,
    *,
    stride: int,
) -> TrackedDensityBlock:
    """Join hidden truth and neighbor distances only after both trackers finish."""

    canonical = np.asarray(sequence.truth_indices[0], dtype=int)

    def correctness(result: TrackResult) -> np.ndarray:
        identities = [
            np.asarray(hidden_order, dtype=int)[np.asarray(assignment, dtype=int)]
            for hidden_order, assignment in zip(
                sequence.truth_indices,
                result.assignments,
                strict=True,
            )
        ]
        return np.stack(identities) == canonical[None, :]

    baseline_correct = correctness(baseline_result)
    extent_correct = correctness(extent_result)
    nearest = np.stack(
        [
            _nearest_neighbor_distances(recording.positions_zyx_um[frame_index])[canonical]
            for frame_index in sequence.source_frame_indices
        ]
    )
    # Initialization is seeded and is excluded from every tracked-event score,
    # exactly as in the primary dense-stratum analysis.
    return TrackedDensityBlock(
        recording_id=recording.recording_id,
        stride=int(stride),
        nearest_neighbor_um=nearest[1:].reshape(-1),
        baseline_correct=baseline_correct[1:].reshape(-1),
        extent_correct=extent_correct[1:].reshape(-1),
    )


def _paired_recording_bootstrap(
    differences: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("bootstrap requires finite complete-recording differences")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "recordings": int(len(values)),
        "mean_difference": float(np.mean(values)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "samples": int(samples),
        "per_recording_differences": values.tolist(),
    }


def score_density_definitions(
    blocks: Sequence[TrackedDensityBlock],
    definitions: Sequence[DensityDefinition],
    *,
    recording_ids: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    """Rescore fixed assignments under each distance cutoff; never retrack."""

    block_values = tuple(blocks)
    if not block_values:
        raise ValueError("at least one tracked block is required")
    result_rows: list[dict[str, object]] = []
    for definition in definitions:
        threshold = definition.threshold_um
        recording_summaries: dict[str, object] = {}
        differences: list[float] = []
        pooled_dense = 0
        pooled_tracked = 0
        pooled_baseline_correct = 0
        pooled_extent_correct = 0
        for recording_id in recording_ids:
            selected = [block for block in block_values if block.recording_id == recording_id]
            if not selected:
                raise ValueError(f"no tracked blocks for recording {recording_id}")
            dense_total = 0
            tracked_total = 0
            baseline_correct = 0
            extent_correct = 0
            for block in selected:
                dense = block.nearest_neighbor_um <= threshold
                dense_total += int(np.sum(dense))
                tracked_total += int(len(dense))
                baseline_correct += int(np.sum(block.baseline_correct & dense))
                extent_correct += int(np.sum(block.extent_correct & dense))
            if dense_total == 0:
                raise ValueError(
                    f"definition {definition.definition_id} has no dense events in {recording_id}"
                )
            baseline_accuracy = baseline_correct / dense_total
            extent_accuracy = extent_correct / dense_total
            difference = extent_accuracy - baseline_accuracy
            differences.append(difference)
            recording_summaries[recording_id] = {
                "analysis_role": ANALYSIS_ROLE,
                "dense_events": dense_total,
                "tracked_events": tracked_total,
                "dense_prevalence": dense_total / tracked_total,
                "dual_anchor": {
                    "correct": baseline_correct,
                    "total": dense_total,
                    "identity_accuracy": baseline_accuracy,
                },
                "dual_anchor_extent": {
                    "correct": extent_correct,
                    "total": dense_total,
                    "identity_accuracy": extent_accuracy,
                },
                "identity_accuracy_difference": difference,
            }
            pooled_dense += dense_total
            pooled_tracked += tracked_total
            pooled_baseline_correct += baseline_correct
            pooled_extent_correct += extent_correct

        pooled_baseline_accuracy = pooled_baseline_correct / pooled_dense
        pooled_extent_accuracy = pooled_extent_correct / pooled_dense
        result_rows.append(
            {
                "analysis_role": ANALYSIS_ROLE,
                "protocol_deviation_statement": PROTOCOL_DEVIATION,
                **asdict(definition),
                "dense_prevalence": pooled_dense / pooled_tracked,
                "pooled": {
                    "dense_events": pooled_dense,
                    "tracked_events": pooled_tracked,
                    "dual_anchor": {
                        "correct": pooled_baseline_correct,
                        "total": pooled_dense,
                        "identity_accuracy": pooled_baseline_accuracy,
                    },
                    "dual_anchor_extent": {
                        "correct": pooled_extent_correct,
                        "total": pooled_dense,
                        "identity_accuracy": pooled_extent_accuracy,
                    },
                    "identity_accuracy_difference": (
                        pooled_extent_accuracy - pooled_baseline_accuracy
                    ),
                },
                "recording_summaries": recording_summaries,
                "complete_recording_paired_bootstrap": _paired_recording_bootstrap(
                    differences,
                    samples=bootstrap_samples,
                    seed=seed,
                ),
            }
        )
    return result_rows


def _full_cohort_summary(blocks: Sequence[TrackedDensityBlock]) -> dict[str, object]:
    total = sum(len(block.nearest_neighbor_um) for block in blocks)
    baseline_correct = sum(int(np.sum(block.baseline_correct)) for block in blocks)
    extent_correct = sum(int(np.sum(block.extent_correct)) for block in blocks)
    baseline_accuracy = baseline_correct / total
    extent_accuracy = extent_correct / total
    return {
        "analysis_role": ANALYSIS_ROLE,
        "tracked_events": total,
        "dual_anchor": {
            "correct": baseline_correct,
            "total": total,
            "identity_accuracy": baseline_accuracy,
        },
        "dual_anchor_extent": {
            "correct": extent_correct,
            "total": total,
            "identity_accuracy": extent_accuracy,
        },
        "identity_accuracy_difference": extent_accuracy - baseline_accuracy,
    }


def verify_q25_reproduction(
    sensitivity_rows: Sequence[dict[str, object]],
    reference_payload: dict[str, object],
    *,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Verify that the Q25 rescore reproduces the reported primary dense result."""

    q25_rows = [row for row in sensitivity_rows if row["definition_id"] == "calibration_nn_q25"]
    if len(q25_rows) != 1:
        raise ValueError("sensitivity output must contain exactly one calibration Q25 row")
    q25_row = q25_rows[0]
    try:
        expected_threshold = float(reference_payload["evaluation_strata"]["dense_threshold_um"])
        expected = reference_payload["confirmatory_summary"][
            "complete_recording_paired_bootstrap"
        ]["dense_identity_accuracy_difference"]
    except (KeyError, TypeError) as error:
        raise ValueError("reference payload lacks the primary dense-stratum result") from error

    actual_bootstrap = q25_row["complete_recording_paired_bootstrap"]
    comparisons = {
        "threshold_um": (
            float(q25_row["threshold_um"]),
            expected_threshold,
        ),
        "mean_difference": (
            float(actual_bootstrap["mean_difference"]),
            float(expected["mean_difference"]),
        ),
        "ci95_low": (
            float(actual_bootstrap["ci95_low"]),
            float(expected["ci95_low"]),
        ),
        "ci95_high": (
            float(actual_bootstrap["ci95_high"]),
            float(expected["ci95_high"]),
        ),
    }
    actual_recordings = np.asarray(actual_bootstrap["per_recording_differences"], dtype=float)
    expected_recordings = np.asarray(expected["per_recording_differences"], dtype=float)
    scalar_pass = all(abs(actual - target) <= tolerance for actual, target in comparisons.values())
    recording_pass = actual_recordings.shape == expected_recordings.shape and bool(
        np.all(np.abs(actual_recordings - expected_recordings) <= tolerance)
    )
    passed = scalar_pass and recording_pass
    result = {
        "analysis_role": ANALYSIS_ROLE,
        "passed": passed,
        "absolute_tolerance": tolerance,
        "expected": {
            name: target for name, (_, target) in comparisons.items()
        }
        | {"per_recording_differences": expected_recordings.tolist()},
        "actual": {
            name: actual for name, (actual, _) in comparisons.items()
        }
        | {"per_recording_differences": actual_recordings.tolist()},
    }
    if not passed:
        raise AssertionError("calibration-Q25 row does not reproduce the primary dense result")
    return result


def run_wormid_density_sensitivity(
    calibration_path: str | Path,
    held_out_paths: Sequence[str | Path],
    *,
    feature_cache_dir: str | Path,
    output_path: str | Path,
    reference_benchmark_path: str | Path | None = None,
    strides: Sequence[int] = DEFAULT_STRIDES,
    seed: int = 20260720,
    bootstrap_samples: int = 10_000,
    geometry_config: ReferenceTrackerConfig = ReferenceTrackerConfig(
        reference_weight=6.0,
        descriptor_weight=0.0,
    ),
    extent_config: DualAnchorExtentConfig = DualAnchorExtentConfig(
        extent_weight=8.0,
        extent_update=0.25,
        feature_indices=tuple(range(15)),
        use_reliability_gate=True,
    ),
) -> dict[str, object]:
    """Run the frozen trackers once and rescore all density definitions."""

    started = perf_counter()
    held_sources = tuple(Path(path).resolve() for path in held_out_paths)
    if len(held_sources) != 4:
        raise ValueError("the frozen sensitivity design requires exactly four held-out recordings")
    calibration_source = Path(calibration_path).resolve()
    all_sources = (calibration_source, *held_sources)
    if len(set(all_sources)) != len(all_sources):
        raise ValueError("calibration and held-out paths must be unique")
    stride_values = tuple(int(stride) for stride in strides)
    if not stride_values or any(stride < 1 for stride in stride_values):
        raise ValueError("strides must be positive")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    recordings = tuple(load_wormid_recording(path) for path in all_sources)
    cache_dir = Path(feature_cache_dir)
    features = tuple(
        load_patch_feature_cache(cache_dir / f"{recording.recording_id}.npz")
        for recording in recordings
    )
    for recording, measured in zip(recordings, features, strict=True):
        if measured.descriptors.shape[:2] != (recording.frames, recording.neurons):
            raise ValueError(f"feature cache dimensions do not match {recording.recording_id}")
    calibration = recordings[0]
    calibration_features = features[0]
    definitions, proxy_metadata = calibration_density_definitions(
        calibration,
        calibration_features,
    )
    if tuple(extent_config.feature_indices or ()) != calibration_features.extent_indices:
        raise ValueError("frozen extent feature indices do not match the calibration cache")
    scaler = fit_robust_feature_scaler(
        calibration_features.descriptors,
        reliability=calibration_features.reliability,
    )

    blocks: list[TrackedDensityBlock] = []
    baseline_runs = 0
    extent_runs = 0
    for path_index, (recording, measured) in enumerate(
        zip(recordings[1:], features[1:], strict=True),
        start=1,
    ):
        for stride in stride_values:
            selected = benchmark_frame_indices(recording.frames, stride)
            sequence = identity_blind_sequence(
                recording,
                frame_indices=selected,
                seed=seed + 1000 * path_index + stride,
            )
            subset = subset_patch_features(measured, selected)
            anonymous = pair_patch_features_to_anonymous_detections(sequence, subset)
            baseline = track_sequence(sequence, "dual_anchor", config=geometry_config)
            baseline_runs += 1
            extent = track_dual_anchor_extent(
                sequence,
                anonymous,
                scaler,
                geometry_config=geometry_config,
                extent_config=extent_config,
            )
            extent_runs += 1
            blocks.append(
                tracked_density_block(
                    recording,
                    sequence,
                    baseline,
                    extent,
                    stride=stride,
                )
            )

    held_ids = tuple(recording.recording_id for recording in recordings[1:])
    sensitivity = score_density_definitions(
        blocks,
        definitions,
        recording_ids=held_ids,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    q25_check: dict[str, object] = {
        "analysis_role": ANALYSIS_ROLE,
        "performed": False,
        "reason": "no reference benchmark supplied",
    }
    reference_source: str | None = None
    if reference_benchmark_path is not None:
        reference_path = Path(reference_benchmark_path).resolve()
        reference_payload = json.loads(reference_path.read_text())
        q25_check = {
            "performed": True,
            **verify_q25_reproduction(sensitivity, reference_payload),
        }
        reference_source = str(reference_path)

    payload: dict[str, object] = {
        "schema_version": 1,
        "analysis_role": ANALYSIS_ROLE,
        "protocol_deviation_statement": PROTOCOL_DEVIATION,
        "interpretation_boundary": {
            "can_retroactively_satisfy_preregistered_density_target": False,
            "reason": (
                "nucleus radius was undefined in the preregistration and calibration "
                "nearest-neighbor Q25 was a different operational definition"
            ),
            "association_decisions_changed_across_density_definitions": False,
        },
        "seed": seed,
        "seed_rule": "seed + 1000*path_index + stride; calibration path_index=0",
        "calibration_recording_id": calibration.recording_id,
        "held_out_recording_ids": list(held_ids),
        "strides": list(stride_values),
        "sources": [
            {
                "recording_id": recording.recording_id,
                "split": "calibration" if index == 0 else "held_out",
                "path": recording.path,
                "feature_cache": str(
                    (cache_dir / f"{recording.recording_id}.npz").resolve()
                ),
            }
            for index, recording in enumerate(recordings)
        ],
        "frozen_tracking_config": {
            "geometry": asdict(geometry_config),
            "extent": asdict(extent_config),
            "feature_scaler": "calibration-only median/IQR over reliability > 0 patches",
        },
        "density_grid": [asdict(definition) for definition in definitions],
        "calibration_proxy_derivation": proxy_metadata,
        "tracking_execution": {
            "baseline_runs": baseline_runs,
            "extent_runs": extent_runs,
            "tracker_runs_total": baseline_runs + extent_runs,
            "expected_blocks_per_method": len(held_ids) * len(stride_values),
            "thresholds_rescored": len(definitions),
            "assignments_reused_across_all_thresholds": True,
        },
        "full_cohort_tracking": _full_cohort_summary(blocks),
        "sensitivity_results": sensitivity,
        "q25_reproduction_check": q25_check,
        "reference_benchmark": reference_source,
        "runtime_seconds": perf_counter() - started,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, help="calibration EY NWB")
    parser.add_argument("--held-out", nargs="+", required=True, help="four held-out EY NWBs")
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--reference-benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strides", nargs="+", type=int, default=list(DEFAULT_STRIDES))
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the density-definition sensitivity audit."""

    args = _parse_args(argv)
    payload = run_wormid_density_sensitivity(
        args.calibration,
        args.held_out,
        feature_cache_dir=args.feature_cache_dir,
        reference_benchmark_path=args.reference_benchmark,
        output_path=args.output,
        strides=args.strides,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    rows = payload["sensitivity_results"]
    print(
        json.dumps(
            {
                "analysis_role": payload["analysis_role"],
                "output": str(Path(args.output).resolve()),
                "q25_reproduction_check": payload["q25_reproduction_check"],
                "threshold_summary": [
                    {
                        "definition_id": row["definition_id"],
                        "threshold_um": row["threshold_um"],
                        "dense_prevalence": row["dense_prevalence"],
                        "mean_recording_difference": row[
                            "complete_recording_paired_bootstrap"
                        ]["mean_difference"],
                        "ci95": [
                            row["complete_recording_paired_bootstrap"]["ci95_low"],
                            row["complete_recording_paired_bootstrap"]["ci95_high"],
                        ],
                    }
                    for row in rows
                ],
                "runtime_seconds": payload["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
