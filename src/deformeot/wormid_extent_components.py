"""Post-hoc mechanism ablation for the WormID local spatial cue.

The confirmatory WormID experiment used the first 15 local-patch features as a
single frozen association cue.  Those features contain three weighted image
centroid offsets and twelve covariance/axis/eigenvalue features.  This module
separates those components without tuning on held-out identities.  It is a
post-hoc mechanism audit: it cannot make the original full-cue result a
confirmatory pure-extent result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .wormid import identity_blind_sequence, load_wormid_recording
from .wormid_benchmark import ReferenceTrackerConfig, track_sequence
from .wormid_extent import (
    AnonymousPatchSequence,
    DualAnchorExtentConfig,
    PatchFeatureSet,
    RobustFeatureScaler,
    _benchmark_frame_indices,
    _paired_recording_bootstrap,
    _reference_metrics_with_strata,
    _subset_patch_features,
    calibration_thresholds,
    fit_robust_feature_scaler,
    load_patch_feature_cache,
    pair_patch_features_to_anonymous_detections,
    track_dual_anchor_extent,
)


SEED = 20260720
STRIDES = (1, 5, 10, 20, 40)
FROZEN_CUE_WEIGHT = 8.0
FROZEN_TEMPLATE_UPDATE = 0.25


@dataclass(frozen=True)
class CueComponent:
    """A prespecified slice of the already-used 15D local spatial cue."""

    method: str
    label: str
    indices: tuple[int, ...]


CUE_COMPONENTS = (
    CueComponent(
        method="dual_anchor_full_spatial_15d",
        label="full 15D local spatial cue",
        indices=tuple(range(15)),
    ),
    CueComponent(
        method="dual_anchor_centroid_offset_3d",
        label="weighted image-centroid offsets only",
        indices=tuple(range(3)),
    ),
    CueComponent(
        method="dual_anchor_extent_only_12d",
        label="covariance/axis/eigenvalue extent only",
        indices=tuple(range(3, 15)),
    ),
)

EXPECTED_SPATIAL_FEATURE_NAMES = (
    "centroid_dx_um",
    "centroid_dy_um",
    "centroid_dz_um",
    "cov_xx_um2",
    "cov_yy_um2",
    "cov_zz_um2",
    "cov_xy_um2",
    "cov_xz_um2",
    "cov_yz_um2",
    "log_std_small",
    "log_std_middle",
    "log_std_large",
    "eig_fraction_small",
    "eig_fraction_middle",
    "eig_fraction_large",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_features(
    features: PatchFeatureSet,
    indices: Sequence[int],
) -> PatchFeatureSet:
    selected = tuple(int(value) for value in indices)
    return PatchFeatureSet(
        descriptors=features.descriptors[..., selected],
        reliability=features.reliability,
        feature_names=tuple(features.feature_names[index] for index in selected),
        extent_indices=tuple(range(len(selected))),
        source_frame_indices=features.source_frame_indices,
        extraction_seconds=0.0,
    )


def _component_anonymous(
    anonymous: AnonymousPatchSequence,
    indices: Sequence[int],
) -> AnonymousPatchSequence:
    selected = np.asarray(tuple(indices), dtype=int)
    return AnonymousPatchSequence(
        descriptors=tuple(values[:, selected] for values in anonymous.descriptors),
        reliability=anonymous.reliability,
    )


def fit_component_scalers(
    calibration_features: PatchFeatureSet,
) -> dict[str, RobustFeatureScaler]:
    """Fit each component scaler on reliable calibration patches only."""

    if tuple(calibration_features.feature_names[:15]) != EXPECTED_SPATIAL_FEATURE_NAMES:
        raise ValueError("feature cache does not contain the expected 15D cue schema")
    return {
        component.method: fit_robust_feature_scaler(
            calibration_features.descriptors[..., component.indices],
            reliability=calibration_features.reliability,
        )
        for component in CUE_COMPONENTS
    }


def _aggregate_rows(rows: Sequence[dict[str, object]]) -> dict[str, float | int]:
    tracked_total = sum(int(row["tracked_total"]) for row in rows)
    dense_total = sum(int(row["dense_total"]) for row in rows)
    switches = sum(int(row["identity_switches"]) for row in rows)
    opportunities = sum(int(row["switch_opportunities"]) for row in rows)
    if not tracked_total or not dense_total or not opportunities:
        raise ValueError("cannot aggregate an empty evaluation stratum")
    return {
        "tracked_correct": sum(int(row["tracked_correct"]) for row in rows),
        "tracked_total": tracked_total,
        "tracked_identity_accuracy": sum(int(row["tracked_correct"]) for row in rows)
        / tracked_total,
        "dense_correct": sum(int(row["dense_correct"]) for row in rows),
        "dense_total": dense_total,
        "dense_identity_accuracy": sum(int(row["dense_correct"]) for row in rows)
        / dense_total,
        "dense_prevalence": dense_total / tracked_total,
        "identity_switches": switches,
        "switch_opportunities": opportunities,
        "identity_switch_rate": switches / opportunities,
    }


def _summarize_methods(
    rows: Sequence[dict[str, object]],
    *,
    recording_ids: Sequence[str],
    strides: Sequence[int],
    methods: Sequence[str],
) -> tuple[
    dict[str, dict[str, float | int]],
    dict[str, dict[str, dict[str, float | int]]],
    dict[str, dict[str, dict[str, float | int]]],
]:
    overall: dict[str, dict[str, float | int]] = {}
    by_recording: dict[str, dict[str, dict[str, float | int]]] = {}
    by_stride: dict[str, dict[str, dict[str, float | int]]] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != len(recording_ids) * len(strides):
            raise ValueError("method lacks a complete held-out recording/stride block")
        overall[method] = _aggregate_rows(selected)
    for recording_id in recording_ids:
        by_recording[recording_id] = {}
        for method in methods:
            selected = [
                row
                for row in rows
                if row["recording_id"] == recording_id and row["method"] == method
            ]
            if len(selected) != len(strides):
                raise ValueError("recording lacks a complete method/stride block")
            by_recording[recording_id][method] = _aggregate_rows(selected)
    for stride in strides:
        by_stride[str(stride)] = {}
        for method in methods:
            selected = [
                row for row in rows if row["stride"] == stride and row["method"] == method
            ]
            if len(selected) != len(recording_ids):
                raise ValueError("stride lacks a complete recording/method block")
            by_stride[str(stride)][method] = _aggregate_rows(selected)
    return overall, by_recording, by_stride


def _component_comparisons(
    by_recording: dict[str, dict[str, dict[str, float | int]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    baseline_method = "dual_anchor_point_only"
    comparisons: dict[str, object] = {}
    for component in CUE_COMPONENTS:
        differences = {
            "tracked_identity_accuracy_difference": [],
            "dense_identity_accuracy_difference": [],
            "identity_switch_rate_reduction": [],
        }
        per_recording: dict[str, dict[str, float]] = {}
        for recording_id, summaries in by_recording.items():
            baseline = summaries[baseline_method]
            candidate = summaries[component.method]
            current = {
                "tracked_identity_accuracy_difference": float(
                    candidate["tracked_identity_accuracy"]
                )
                - float(baseline["tracked_identity_accuracy"]),
                "dense_identity_accuracy_difference": float(
                    candidate["dense_identity_accuracy"]
                )
                - float(baseline["dense_identity_accuracy"]),
                "identity_switch_rate_reduction": float(
                    baseline["identity_switch_rate"]
                )
                - float(candidate["identity_switch_rate"]),
            }
            per_recording[recording_id] = current
            for metric, value in current.items():
                differences[metric].append(value)
        comparisons[component.method] = {
            "baseline": baseline_method,
            "candidate": component.method,
            "per_recording_paired_differences": per_recording,
            "complete_recording_paired_bootstrap": {
                metric: _paired_recording_bootstrap(
                    values,
                    samples=bootstrap_samples,
                    seed=seed,
                )
                for metric, values in differences.items()
            },
        }
    return comparisons


def _metric_signature(row: dict[str, object]) -> dict[str, object]:
    excluded = {
        "method",
        "runtime_seconds",
        "first_source_frame",
        "last_source_frame",
    }
    return {key: value for key, value in row.items() if key not in excluded}


def _validate_full_cue_reproduction(
    reference_path: Path,
    *,
    rows: Sequence[dict[str, object]],
    comparisons: dict[str, object],
    calibration_recording_id: str,
    held_out_recording_ids: Sequence[str],
    strides: Sequence[int],
    seed: int,
) -> dict[str, object]:
    reference = json.loads(reference_path.read_text())
    prerequisites = {
        "calibration_recording_id": reference.get("calibration_recording_id")
        == calibration_recording_id,
        "held_out_recording_ids": reference.get("held_out_recording_ids")
        == list(held_out_recording_ids),
        "strides": reference.get("strides") == list(strides),
        "seed": reference.get("seed") == seed,
        "extent_weight": reference.get("calibration_grid", {})
        .get("selected_config", {})
        .get("extent_weight")
        == FROZEN_CUE_WEIGHT,
        "extent_update": reference.get("calibration_grid", {})
        .get("selected_config", {})
        .get("extent_update")
        == FROZEN_TEMPLATE_UPDATE,
        "feature_indices": reference.get("calibration_grid", {})
        .get("selected_config", {})
        .get("feature_indices")
        == list(range(15)),
    }
    if not all(prerequisites.values()):
        failed = [name for name, passed in prerequisites.items() if not passed]
        raise ValueError(f"reference result is incompatible: {failed}")

    reference_rows = {
        (row["recording_id"], int(row["stride"]), row["method"]): row
        for row in reference["confirmatory_held_out_results"]
    }
    mismatches: list[str] = []
    method_map = {
        "dual_anchor_point_only": "dual_anchor",
        "dual_anchor_full_spatial_15d": "dual_anchor_extent",
    }
    for row in rows:
        if row["method"] not in method_map:
            continue
        key = (
            row["recording_id"],
            int(row["stride"]),
            method_map[str(row["method"])],
        )
        expected = reference_rows.get(key)
        if expected is None or _metric_signature(row) != _metric_signature(expected):
            mismatches.append("/".join(str(value) for value in key))

    actual_bootstrap = comparisons["dual_anchor_full_spatial_15d"][
        "complete_recording_paired_bootstrap"
    ]
    expected_bootstrap = reference["confirmatory_summary"][
        "complete_recording_paired_bootstrap"
    ]
    for metric in (
        "tracked_identity_accuracy_difference",
        "dense_identity_accuracy_difference",
        "identity_switch_rate_reduction",
    ):
        if actual_bootstrap[metric] != expected_bootstrap[metric]:
            mismatches.append(f"bootstrap/{metric}")

    exact = not mismatches
    if not exact:
        raise AssertionError(
            "full 15D arm did not exactly reproduce the reference result: "
            + ", ".join(mismatches)
        )
    return {
        "checked": True,
        "exact": True,
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": _sha256(reference_path),
        "row_metric_blocks_compared": 2 * len(held_out_recording_ids) * len(strides),
        "bootstrap_contrasts_compared": 3,
        "mismatches": [],
    }


def run_extent_component_ablation(
    calibration_path: str | Path,
    held_out_paths: Sequence[str | Path],
    *,
    feature_cache_dir: str | Path,
    output_path: str | Path,
    reference_result_path: str | Path | None = None,
    strides: Sequence[int] = STRIDES,
    seed: int = SEED,
    bootstrap_samples: int = 10_000,
    geometry_config: ReferenceTrackerConfig = ReferenceTrackerConfig(
        reference_weight=6.0,
        descriptor_weight=0.0,
    ),
) -> dict[str, object]:
    """Run the frozen, held-out cue-component mechanism audit."""

    calibration_source = Path(calibration_path).resolve()
    held_sources = tuple(Path(path).resolve() for path in held_out_paths)
    if not held_sources:
        raise ValueError("at least one held-out recording is required")
    sources = (calibration_source, *held_sources)
    if len(set(sources)) != len(sources):
        raise ValueError("calibration and held-out paths must be unique")
    stride_values = tuple(int(value) for value in strides)
    if not stride_values or any(value < 1 for value in stride_values):
        raise ValueError("strides must contain positive integers")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    recordings = tuple(load_wormid_recording(path) for path in sources)
    cache_dir = Path(feature_cache_dir)
    features: list[PatchFeatureSet] = []
    cache_paths: list[Path] = []
    for recording in recordings:
        cache_path = cache_dir / f"{recording.recording_id}.npz"
        measured = load_patch_feature_cache(cache_path)
        if measured.descriptors.shape[:2] != (recording.frames, recording.neurons):
            raise ValueError(f"{cache_path} dimensions do not match {recording.recording_id}")
        if tuple(measured.feature_names[:15]) != EXPECTED_SPATIAL_FEATURE_NAMES:
            raise ValueError(f"{cache_path} does not contain the expected 15D cue schema")
        features.append(measured)
        cache_paths.append(cache_path.resolve())

    calibration = recordings[0]
    scalers = fit_component_scalers(features[0])
    dense_threshold, _ = calibration_thresholds(calibration, strides=stride_values)
    severe_thresholds: dict[int, float] = {}
    for stride in stride_values:
        selected = _benchmark_frame_indices(calibration.frames, stride)
        sequence = identity_blind_sequence(
            calibration,
            frame_indices=selected,
            seed=seed + stride,
        )
        point_result = track_sequence(sequence, "dual_anchor", config=geometry_config)
        severe_thresholds[stride] = float(
            np.quantile(point_result.cloud_motion_um[1:], 0.75)
        )

    rows: list[dict[str, object]] = []
    for path_index, (recording, measured) in enumerate(
        zip(recordings[1:], features[1:], strict=True),
        start=1,
    ):
        for stride in stride_values:
            selected = _benchmark_frame_indices(recording.frames, stride)
            sequence = identity_blind_sequence(
                recording,
                frame_indices=selected,
                seed=seed + 1000 * path_index + stride,
            )
            subset = _subset_patch_features(measured, selected)
            anonymous = pair_patch_features_to_anonymous_detections(sequence, subset)
            results = {
                "dual_anchor_point_only": track_sequence(
                    sequence,
                    "dual_anchor",
                    config=geometry_config,
                )
            }
            for component in CUE_COMPONENTS:
                results[component.method] = track_dual_anchor_extent(
                    sequence,
                    _component_anonymous(anonymous, component.indices),
                    scalers[component.method],
                    geometry_config=geometry_config,
                    extent_config=DualAnchorExtentConfig(
                        extent_weight=FROZEN_CUE_WEIGHT,
                        extent_update=FROZEN_TEMPLATE_UPDATE,
                        feature_indices=tuple(range(len(component.indices))),
                        use_reliability_gate=True,
                    ),
                )
            for method, result in results.items():
                row = _reference_metrics_with_strata(
                    sequence,
                    result,
                    recording,
                    dense_threshold_um=dense_threshold,
                    severe_threshold_um=severe_thresholds[stride],
                )
                row.pop("runtime_seconds", None)
                row.update(
                    recording_id=recording.recording_id,
                    split="held_out",
                    method=method,
                    stride=stride,
                    frames_evaluated=len(sequence.frames),
                    first_source_frame=selected[0],
                    last_source_frame=selected[-1],
                )
                rows.append(row)

    held_ids = tuple(recording.recording_id for recording in recordings[1:])
    methods = ("dual_anchor_point_only", *(item.method for item in CUE_COMPONENTS))
    overall, by_recording, by_stride = _summarize_methods(
        rows,
        recording_ids=held_ids,
        strides=stride_values,
        methods=methods,
    )
    comparisons = _component_comparisons(
        by_recording,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    if reference_result_path is None:
        reproduction: dict[str, object] = {
            "checked": False,
            "exact": None,
            "reason": "no reference result supplied",
        }
    else:
        reproduction = _validate_full_cue_reproduction(
            Path(reference_result_path),
            rows=rows,
            comparisons=comparisons,
            calibration_recording_id=calibration.recording_id,
            held_out_recording_ids=held_ids,
            strides=stride_values,
            seed=seed,
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "analysis_role": (
            "post-hoc mechanism ablation; this analysis cannot convert the original "
            "full-cue result into a confirmatory pure-extent claim"
        ),
        "seed": seed,
        "seed_rule": "seed + 1000*path_index + stride; calibration path_index=0",
        "calibration_recording_id": calibration.recording_id,
        "held_out_recording_ids": list(held_ids),
        "strides": list(stride_values),
        "frozen_association_config": {
            "cue_weight": FROZEN_CUE_WEIGHT,
            "template_update": FROZEN_TEMPLATE_UPDATE,
            "use_reliability_gate": True,
            "geometry": asdict(geometry_config),
            "held_out_identity_retuning": False,
        },
        "cue_components": {
            component.method: {
                "label": component.label,
                "indices": list(component.indices),
                "feature_names": [
                    EXPECTED_SPATIAL_FEATURE_NAMES[index]
                    for index in component.indices
                ],
                "scaler": (
                    "per-feature median/IQR fit on calibration patches with reliability > 0"
                ),
            }
            for component in CUE_COMPONENTS
        },
        "measurement_boundary": {
            "stored_trace_arrays_read": False,
            "identity_labels_used_for_association": False,
            "total_intensity_is_descriptor_feature": False,
            "reliability_gate_uses_signal_to_background": True,
            "local_patches_may_contain_neighboring_signal": True,
        },
        "sources": [
            {
                "recording_id": recording.recording_id,
                "split": "calibration" if index == 0 else "held_out",
                "path": recording.path,
                "frames": recording.frames,
                "neurons": recording.neurons,
                "feature_cache": str(cache_paths[index]),
                "feature_cache_sha256": _sha256(cache_paths[index]),
            }
            for index, recording in enumerate(recordings)
        ],
        "evaluation_strata": {
            "dense_threshold_um": dense_threshold,
            "dense_source": "calibration-animal pooled nearest-neighbor Q25",
            "severe_threshold_um_by_stride": {
                str(key): value for key, value in severe_thresholds.items()
            },
        },
        "held_out_results": rows,
        "overall_pooled_by_method": overall,
        "recording_summaries": by_recording,
        "micro_by_stride": by_stride,
        "comparisons_to_point_only": comparisons,
        "full_15d_reference_reproduction": reproduction,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--held-out", nargs="+", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--reference-result")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strides", nargs="+", type=int, default=list(STRIDES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_extent_component_ablation(
        args.calibration,
        args.held_out,
        feature_cache_dir=args.feature_cache_dir,
        reference_result_path=args.reference_result,
        output_path=args.output,
        strides=args.strides,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
