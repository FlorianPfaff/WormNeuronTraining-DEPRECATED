"""Execute the frozen untouched-cohort WormID geometry replication.

This module deliberately reuses :func:`identity_blind_sequence` and
:func:`track_sequence` without changing either implementation.  It applies the
frozen full-inventory seed rule, evaluates all four centre-only methods, and
reports recording-level and acquisition-date uncertainty for persistent
initialized ROI/neuron-instance identities.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

from .wormid import WormIDRecording, identity_blind_sequence
from .wormid_benchmark import METHODS, ReferenceTrackerConfig, score_tracks, track_sequence
from .wormid_dandi import (
    DandiAsset,
    asset_inventory_row,
    extract_coordinate_cache,
    fetch_dandi_inventory,
    sha256_file,
)
from .wormid_switch_strata import (
    benchmark_frame_indices,
    exact_high_motion_mask,
    score_switch_stratum,
)


PRIMARY_BASELINE = "first_affine"
CANDIDATE = "dual_anchor"
IDENTITY_SEMANTICS = "persistent initialized ROI/neuron-instance identity"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_frozen_protocol(
    repository_root: str | Path,
    *,
    freeze_path: str | Path = "configs/wormid_prospective.freeze.sha256",
) -> dict[str, object]:
    """Verify every frozen file before configuration or outcomes are read."""

    root = Path(repository_root).resolve()
    manifest_path = root / freeze_path
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("prospective freeze manifest is empty")
    files: dict[str, dict[str, object]] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if match is None:
            raise ValueError(f"malformed freeze line: {line!r}")
        expected, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("freeze manifest paths must stay inside the repository")
        source = root / relative_path
        actual = sha256_file(source)
        if actual != expected:
            raise RuntimeError(
                f"frozen protocol integrity failure for {relative}: "
                f"expected {expected}, observed {actual}"
            )
        files[relative] = {
            "expected_sha256": expected,
            "observed_sha256": actual,
            "size": source.stat().st_size,
            "verified": True,
        }
    return {
        "freeze_manifest": str(Path(freeze_path)),
        "freeze_manifest_sha256": sha256_file(manifest_path),
        "files": files,
        "all_verified": True,
    }


def load_frozen_config(path: str | Path) -> dict[str, object]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prospective configuration must be a mapping")
    return payload


def _recording_date(recording_id: str) -> str:
    match = re.search(r"_ses-(\d{8})_", recording_id)
    if match is None:
        raise ValueError(f"cannot parse acquisition date from {recording_id}")
    return match.group(1)


def _validate_inventory_and_config(
    config: Mapping[str, object], assets: Sequence[DandiAsset]
) -> dict[str, object]:
    dataset = config.get("dataset")
    association = config.get("association")
    primary = config.get("primary_analysis")
    secondary = config.get("secondary_analysis")
    if not all(isinstance(value, dict) for value in (dataset, association, primary, secondary)):
        raise ValueError("frozen configuration sections are missing")
    assert isinstance(dataset, dict)
    assert isinstance(association, dict)
    assert isinstance(primary, dict)
    assert isinstance(secondary, dict)

    inventory = [asset.recording_id for asset in assets]
    if inventory != sorted(inventory) or len(inventory) != 21:
        raise ValueError("frozen DANDI inventory must contain 21 lexicographically sorted assets")
    tuning = str(dataset["tuning_recording"])
    legacy = tuple(str(value) for value in dataset["legacy_evaluation_recordings"])
    untouched = tuple(str(value) for value in dataset["untouched_recordings"])
    expected = {tuning, *legacy, *untouched}
    if len(legacy) != 4 or len(untouched) != 16 or set(inventory) != expected:
        raise ValueError("DANDI inventory does not exactly match the frozen 1+4+16 split")
    methods = tuple(str(value) for value in association["methods"])
    if methods != METHODS:
        raise ValueError("frozen methods do not match the four implemented centre-only methods")
    if str(association["primary_baseline"]) != PRIMARY_BASELINE:
        raise ValueError("the frozen primary baseline changed")
    if str(association["candidate"]) != CANDIDATE:
        raise ValueError("the frozen candidate changed")
    strides = tuple(int(value) for value in association["strides"])
    if strides != (1, 5, 10, 20, 40):
        raise ValueError("the five frozen temporal strides changed")
    if str(association["seed_rule"]) != (
        "base_seed + 1000*lexicographic_full_inventory_index + stride"
    ):
        raise ValueError("the frozen seed rule changed")
    if int(primary["bootstrap_samples"]) != 20_000:
        raise ValueError("primary bootstrap sample count changed")
    if int(secondary["hierarchical_bootstrap_samples"]) != 20_000:
        raise ValueError("hierarchical bootstrap sample count changed")
    return {
        "tuning": tuning,
        "legacy": legacy,
        "untouched": untouched,
        "strides": strides,
        "methods": methods,
        "base_seed": int(association["base_seed"]),
        "primary_bootstrap_samples": int(primary["bootstrap_samples"]),
        "primary_bootstrap_seed": int(primary["bootstrap_seed"]),
        "hierarchical_bootstrap_samples": int(
            secondary["hierarchical_bootstrap_samples"]
        ),
        "hierarchical_bootstrap_seed": int(secondary["hierarchical_bootstrap_seed"]),
        "same_date_exclusion": str(secondary["acquisition_date_sensitivity_exclusion"]),
    }


def _tracker_config(config: Mapping[str, object]) -> ReferenceTrackerConfig:
    association = config["association"]
    assert isinstance(association, dict)
    geometry = association["geometry"]
    assert isinstance(geometry, dict)
    return ReferenceTrackerConfig(
        reference_weight=float(geometry["reference_weight"]),
        descriptor_weight=float(geometry["descriptor_weight"]),
        affine_iterations=int(geometry["affine_iterations"]),
        affine_ridge=float(geometry["affine_ridge"]),
        affine_huber_delta=float(geometry["affine_huber_delta"]),
        affine_min_scale=float(geometry["affine_min_scale"]),
        affine_max_scale=float(geometry["affine_max_scale"]),
        spatial_scale_floor_um=float(geometry["spatial_scale_floor_um"]),
    )


def _benchmark_recording(
    task: tuple[
        WormIDRecording,
        str,
        int,
        tuple[int, ...],
        int,
        ReferenceTrackerConfig,
    ],
) -> list[dict[str, object]]:
    recording, cohort, inventory_index, strides, base_seed, tracker_config = task
    rows: list[dict[str, object]] = []
    for stride in strides:
        selected = benchmark_frame_indices(recording.frames, stride)
        permutation_seed = base_seed + 1000 * inventory_index + stride
        sequence = identity_blind_sequence(
            recording,
            frame_indices=selected,
            seed=permutation_seed,
        )
        results = {
            method: track_sequence(sequence, method, config=tracker_config)
            for method in METHODS
        }
        masks = {
            method: exact_high_motion_mask(result, selected)
            for method, result in results.items()
        }
        reference_mask = masks[METHODS[0]]
        if any(not np.array_equal(reference_mask, mask) for mask in masks.values()):
            raise AssertionError("identity-blind high-motion strata differed by method")

        for method in METHODS:
            result = results[method]
            metrics = score_tracks(sequence, result)
            high_motion = score_switch_stratum(sequence, result, reference_mask)
            opportunities = (metrics.frames - 1) * metrics.neurons
            row: dict[str, object] = {
                "recording_id": recording.recording_id,
                "cohort": cohort,
                "acquisition_date": _recording_date(recording.recording_id),
                "full_inventory_index": inventory_index,
                "stride": stride,
                "permutation_seed": permutation_seed,
                "method": method,
                "frames": metrics.frames,
                "neurons": metrics.neurons,
                "first_source_frame": selected[0],
                "last_source_frame": selected[-1],
                "correct": metrics.correct,
                "total": metrics.total,
                "identity_accuracy": metrics.identity_accuracy,
                "identity_switches": metrics.identity_switches,
                "identity_switch_opportunities": opportunities,
                "identity_switch_rate": metrics.identity_switches / opportunities,
                "high_motion_correct": metrics.severe_correct,
                "high_motion_total": metrics.severe_total,
                "high_motion_identity_accuracy": metrics.severe_identity_accuracy,
                "high_motion_identity_switches": int(high_motion["switches"]),
                "high_motion_identity_switch_opportunities": int(
                    high_motion["opportunities"]
                ),
                "high_motion_identity_switch_rate": float(high_motion["switch_rate"]),
                "high_motion_selected_transitions": int(
                    high_motion["selected_transitions"]
                ),
                "high_motion_threshold_um": float(
                    np.min(result.cloud_motion_um[1:][reference_mask])
                ),
                "high_motion_selection": (
                    "exact top quartile of identity-blind symmetric cloud motion; "
                    "earlier source-frame tie break"
                ),
                "high_motion_ties_at_boundary": metrics.severe_ties_at_boundary,
                "identity_semantics": IDENTITY_SEMANTICS,
                "semantic_labels_used_for_association": False,
                "fluorescence_used_for_association": False,
            }
            rows.append(row)
    return rows


def _bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    value_name: str,
) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or np.any(~np.isfinite(array)):
        raise ValueError(f"finite one-dimensional {value_name} values are required")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return {
        "estimand": value_name,
        "recordings": int(len(array)),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "per_recording_values": array.tolist(),
    }


def _method_index(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, int, str], Mapping[str, object]]:
    indexed: dict[tuple[str, int, str], Mapping[str, object]] = {}
    for row in rows:
        key = (str(row["recording_id"]), int(row["stride"]), str(row["method"]))
        if key in indexed:
            raise ValueError(f"duplicate benchmark row {key}")
        indexed[key] = row
    return indexed


def _recording_effects(
    rows: Sequence[Mapping[str, object]], recording_ids: Sequence[str]
) -> list[dict[str, object]]:
    indexed = _method_index(rows)
    output: list[dict[str, object]] = []
    for recording_id in sorted(recording_ids):
        recording_rows = [row for row in rows if row["recording_id"] == recording_id]
        strides = sorted({int(row["stride"]) for row in recording_rows})
        accuracy: list[float] = []
        high_accuracy: list[float] = []
        switch_counts = {PRIMARY_BASELINE: 0, CANDIDATE: 0}
        switch_opportunities = {PRIMARY_BASELINE: 0, CANDIDATE: 0}
        high_switch_counts = {PRIMARY_BASELINE: 0, CANDIDATE: 0}
        high_switch_opportunities = {PRIMARY_BASELINE: 0, CANDIDATE: 0}
        for stride in strides:
            baseline = indexed[(recording_id, stride, PRIMARY_BASELINE)]
            candidate = indexed[(recording_id, stride, CANDIDATE)]
            accuracy.append(
                float(candidate["identity_accuracy"])
                - float(baseline["identity_accuracy"])
            )
            high_accuracy.append(
                float(candidate["high_motion_identity_accuracy"])
                - float(baseline["high_motion_identity_accuracy"])
            )
            for method, row in ((PRIMARY_BASELINE, baseline), (CANDIDATE, candidate)):
                switch_counts[method] += int(row["identity_switches"])
                switch_opportunities[method] += int(row["identity_switch_opportunities"])
                high_switch_counts[method] += int(row["high_motion_identity_switches"])
                high_switch_opportunities[method] += int(
                    row["high_motion_identity_switch_opportunities"]
                )
        baseline_rate = switch_counts[PRIMARY_BASELINE] / switch_opportunities[
            PRIMARY_BASELINE
        ]
        candidate_rate = switch_counts[CANDIDATE] / switch_opportunities[CANDIDATE]
        baseline_high_rate = high_switch_counts[PRIMARY_BASELINE] / high_switch_opportunities[
            PRIMARY_BASELINE
        ]
        candidate_high_rate = high_switch_counts[CANDIDATE] / high_switch_opportunities[
            CANDIDATE
        ]
        first = recording_rows[0]
        output.append(
            {
                "recording_id": recording_id,
                "cohort": str(first["cohort"]),
                "acquisition_date": str(first["acquisition_date"]),
                "strides": strides,
                "mean_accuracy_difference": float(np.mean(accuracy)),
                "mean_high_motion_accuracy_difference": float(np.mean(high_accuracy)),
                "first_affine_switches": switch_counts[PRIMARY_BASELINE],
                "dual_anchor_switches": switch_counts[CANDIDATE],
                "switch_opportunities": switch_opportunities[PRIMARY_BASELINE],
                "first_affine_switch_rate": baseline_rate,
                "dual_anchor_switch_rate": candidate_rate,
                "switch_rate_reduction": baseline_rate - candidate_rate,
                "relative_switch_reduction": (
                    (switch_counts[PRIMARY_BASELINE] - switch_counts[CANDIDATE])
                    / switch_counts[PRIMARY_BASELINE]
                    if switch_counts[PRIMARY_BASELINE]
                    else None
                ),
                "first_affine_high_motion_switches": high_switch_counts[
                    PRIMARY_BASELINE
                ],
                "dual_anchor_high_motion_switches": high_switch_counts[CANDIDATE],
                "high_motion_switch_opportunities": high_switch_opportunities[
                    PRIMARY_BASELINE
                ],
                "first_affine_high_motion_switch_rate": baseline_high_rate,
                "dual_anchor_high_motion_switch_rate": candidate_high_rate,
                "high_motion_switch_rate_reduction": baseline_high_rate
                - candidate_high_rate,
                "relative_high_motion_switch_reduction": (
                    (
                        high_switch_counts[PRIMARY_BASELINE]
                        - high_switch_counts[CANDIDATE]
                    )
                    / high_switch_counts[PRIMARY_BASELINE]
                    if high_switch_counts[PRIMARY_BASELINE]
                    else None
                ),
            }
        )
    return output


def _pooled_switch_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    high_motion: bool,
) -> dict[str, object]:
    prefix = "high_motion_identity_" if high_motion else "identity_"
    counts = {
        method: sum(
            int(row[f"{prefix}switches"])
            for row in rows
            if row["method"] == method
        )
        for method in (PRIMARY_BASELINE, CANDIDATE)
    }
    opportunities = sum(
        int(row[f"{prefix}switch_opportunities"])
        for row in rows
        if row["method"] == PRIMARY_BASELINE
    )
    baseline = counts[PRIMARY_BASELINE]
    candidate = counts[CANDIDATE]
    return {
        "first_affine_switches": baseline,
        "dual_anchor_switches": candidate,
        "opportunities": opportunities,
        "first_affine_switch_rate": baseline / opportunities,
        "dual_anchor_switch_rate": candidate / opportunities,
        "switch_rate_reduction": (baseline - candidate) / opportunities,
        "relative_switch_reduction": (baseline - candidate) / baseline
        if baseline
        else None,
    }


def _by_stride_summary(
    rows: Sequence[Mapping[str, object]],
    recording_ids: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    indexed = _method_index(rows)
    strides = sorted({int(row["stride"]) for row in rows})
    output: dict[str, object] = {}
    for stride in strides:
        selected = [
            row
            for row in rows
            if int(row["stride"]) == stride and str(row["recording_id"]) in recording_ids
        ]
        accuracy = []
        high_accuracy = []
        switch_rate_reduction = []
        high_switch_rate_reduction = []
        for recording_id in sorted(recording_ids):
            baseline = indexed[(recording_id, stride, PRIMARY_BASELINE)]
            candidate = indexed[(recording_id, stride, CANDIDATE)]
            accuracy.append(
                float(candidate["identity_accuracy"])
                - float(baseline["identity_accuracy"])
            )
            high_accuracy.append(
                float(candidate["high_motion_identity_accuracy"])
                - float(baseline["high_motion_identity_accuracy"])
            )
            switch_rate_reduction.append(
                float(baseline["identity_switch_rate"])
                - float(candidate["identity_switch_rate"])
            )
            high_switch_rate_reduction.append(
                float(baseline["high_motion_identity_switch_rate"])
                - float(candidate["high_motion_identity_switch_rate"])
            )
        output[str(stride)] = {
            "accuracy_difference": _bootstrap_mean(
                accuracy,
                samples=samples,
                seed=seed,
                value_name="dual_anchor minus first_affine seeded identity accuracy",
            ),
            "all_transition_switches": _pooled_switch_summary(
                selected, high_motion=False
            ),
            "switch_rate_reduction": _bootstrap_mean(
                switch_rate_reduction,
                samples=samples,
                seed=seed,
                value_name="first_affine minus dual_anchor identity-switch rate",
            ),
            "high_motion_accuracy_difference": _bootstrap_mean(
                high_accuracy,
                samples=samples,
                seed=seed,
                value_name=(
                    "dual_anchor minus first_affine high-motion seeded identity accuracy"
                ),
            ),
            "high_motion_switches": _pooled_switch_summary(selected, high_motion=True),
            "high_motion_switch_rate_reduction": _bootstrap_mean(
                high_switch_rate_reduction,
                samples=samples,
                seed=seed,
                value_name=(
                    "first_affine minus dual_anchor high-motion identity-switch rate"
                ),
            ),
        }
    return output


def _hierarchical_bootstrap(
    effects: Sequence[Mapping[str, object]],
    *,
    field: str,
    samples: int,
    seed: int,
) -> dict[str, object]:
    by_date: dict[str, np.ndarray] = {}
    for date in sorted({str(row["acquisition_date"]) for row in effects}):
        by_date[date] = np.asarray(
            [float(row[field]) for row in effects if row["acquisition_date"] == date],
            dtype=float,
        )
    dates = tuple(sorted(by_date))
    if len(dates) < 2:
        raise ValueError("hierarchical bootstrap needs at least two acquisition dates")
    date_means = {date: float(values.mean()) for date, values in by_date.items()}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        selected_dates = rng.choice(dates, size=len(dates), replace=True)
        sampled_date_means = []
        for date in selected_dates:
            values = by_date[str(date)]
            sampled_date_means.append(
                float(rng.choice(values, size=len(values), replace=True).mean())
            )
        draws[sample_index] = float(np.mean(sampled_date_means))
    return {
        "field": field,
        "estimand": "equal acquisition-date mean of within-date equal-recording means",
        "dates": len(dates),
        "recordings": len(effects),
        "point_estimate": float(np.mean(list(date_means.values()))),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "date_means": date_means,
    }


def _method_descriptives(
    rows: Sequence[Mapping[str, object]], recording_ids: Sequence[str]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for method in METHODS:
        recording_values = []
        for recording_id in sorted(recording_ids):
            values = [
                float(row["identity_accuracy"])
                for row in rows
                if row["recording_id"] == recording_id and row["method"] == method
            ]
            recording_values.append(float(np.mean(values)))
        output[method] = {
            "recording_macro_mean_accuracy": float(np.mean(recording_values)),
            "per_recording_mean_accuracy": recording_values,
        }
    return output


def analyze_prospective_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    untouched_ids: Sequence[str],
    legacy_ids: Sequence[str],
    primary_samples: int,
    primary_seed: int,
    hierarchical_samples: int,
    hierarchical_seed: int,
    same_date_exclusion: str,
) -> dict[str, object]:
    """Apply the frozen primary and secondary estimands to benchmark rows."""

    expected_ids = set(untouched_ids) | set(legacy_ids)
    observed_ids = {str(row["recording_id"]) for row in rows}
    if observed_ids != expected_ids:
        raise ValueError("benchmark rows do not contain the exact 20 nontuning recordings")
    expected_rows = len(expected_ids) * 5 * len(METHODS)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} method/stride rows, observed {len(rows)}")

    untouched_rows = [row for row in rows if row["recording_id"] in untouched_ids]
    untouched_effects = _recording_effects(untouched_rows, untouched_ids)
    primary = _bootstrap_mean(
        [float(row["mean_accuracy_difference"]) for row in untouched_effects],
        samples=primary_samples,
        seed=primary_seed,
        value_name=(
            "equal-recording mean of within-recording equal-stride dual_anchor minus "
            "first_affine seeded identity accuracy"
        ),
    )
    primary["criterion"] = "95% percentile interval lower endpoint greater than zero"
    primary["replication_criterion_met"] = bool(float(primary["ci95_low"]) > 0)

    switch_rate = _bootstrap_mean(
        [float(row["switch_rate_reduction"]) for row in untouched_effects],
        samples=primary_samples,
        seed=primary_seed,
        value_name=(
            "equal-recording first_affine minus dual_anchor switch-rate reduction after "
            "pooling switches and opportunities across strides within recording"
        ),
    )
    high_accuracy = _bootstrap_mean(
        [float(row["mean_high_motion_accuracy_difference"]) for row in untouched_effects],
        samples=primary_samples,
        seed=primary_seed,
        value_name=(
            "equal-recording mean of within-recording equal-stride dual_anchor minus "
            "first_affine high-motion seeded identity accuracy"
        ),
    )
    high_switch_rate = _bootstrap_mean(
        [float(row["high_motion_switch_rate_reduction"]) for row in untouched_effects],
        samples=primary_samples,
        seed=primary_seed,
        value_name=(
            "equal-recording first_affine minus dual_anchor high-motion switch-rate "
            "reduction after pooling across strides within recording"
        ),
    )

    if same_date_exclusion not in untouched_ids:
        raise ValueError("same-date sensitivity exclusion is not in the untouched cohort")
    sensitivity_effects = [
        row for row in untouched_effects if row["recording_id"] != same_date_exclusion
    ]
    sensitivity = _bootstrap_mean(
        [float(row["mean_accuracy_difference"]) for row in sensitivity_effects],
        samples=primary_samples,
        seed=primary_seed,
        value_name=(
            "equal-recording dual_anchor minus first_affine accuracy after excluding the "
            "only untouched recording acquired on the tuning date"
        ),
    )
    sensitivity.update(
        {
            "excluded_recording": same_date_exclusion,
            "declared_role": "sensitivity analysis, not replacement primary analysis",
            "lower_endpoint_greater_than_zero": bool(
                float(sensitivity["ci95_low"]) > 0
            ),
        }
    )

    all_effects = _recording_effects(rows, sorted(expected_ids))
    strata: dict[str, object] = {}
    for name, identifiers in (
        ("legacy_evaluation", tuple(legacy_ids)),
        ("untouched_expansion", tuple(untouched_ids)),
        ("all_nontuning", tuple(sorted(expected_ids))),
    ):
        selected_effects = [row for row in all_effects if row["recording_id"] in identifiers]
        selected_rows = [row for row in rows if row["recording_id"] in identifiers]
        strata[name] = {
            "recording_ids": list(identifiers),
            "recordings": len(identifiers),
            "accuracy_difference": _bootstrap_mean(
                [float(row["mean_accuracy_difference"]) for row in selected_effects],
                samples=primary_samples,
                seed=primary_seed,
                value_name=(
                    "equal-recording mean within-recording equal-stride dual_anchor minus "
                    "first_affine seeded identity accuracy"
                ),
            ),
            "all_transition_switches": _pooled_switch_summary(
                selected_rows, high_motion=False
            ),
            "high_motion_accuracy_difference": _bootstrap_mean(
                [
                    float(row["mean_high_motion_accuracy_difference"])
                    for row in selected_effects
                ],
                samples=primary_samples,
                seed=primary_seed,
                value_name=(
                    "equal-recording mean within-recording equal-stride dual_anchor minus "
                    "first_affine high-motion seeded identity accuracy"
                ),
            ),
            "high_motion_switches": _pooled_switch_summary(
                selected_rows, high_motion=True
            ),
            "method_descriptives": _method_descriptives(selected_rows, identifiers),
        }

    hierarchical = {
        field: _hierarchical_bootstrap(
            untouched_effects,
            field=field,
            samples=hierarchical_samples,
            seed=hierarchical_seed,
        )
        for field in (
            "mean_accuracy_difference",
            "switch_rate_reduction",
            "mean_high_motion_accuracy_difference",
            "high_motion_switch_rate_reduction",
        )
    }
    return {
        "primary_untouched_accuracy_effect": primary,
        "untouched_by_stride": _by_stride_summary(
            untouched_rows,
            untouched_ids,
            samples=primary_samples,
            seed=primary_seed,
        ),
        "untouched_pooled_switch_rate_effect": switch_rate,
        "untouched_high_motion_accuracy_effect": high_accuracy,
        "untouched_high_motion_switch_rate_effect": high_switch_rate,
        "untouched_recording_effects": untouched_effects,
        "untouched_acquisition_date_hierarchical_bootstrap": hierarchical,
        "same_tuning_date_exclusion_sensitivity": sensitivity,
        "nontuning_20_stratified_summary": strata,
        "nontuning_20_recording_effects": all_effects,
    }


def _load_assets_with_retries(
    dandiset: str, version: str, *, attempts: int = 3
) -> tuple[tuple[DandiAsset, ...], int]:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            return fetch_dandi_inventory(dandiset, version), attempt
        except Exception as error:  # pragma: no cover - network contingency
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if attempt < attempts:
                print(errors[-1], file=sys.stderr, flush=True)
    raise RuntimeError("DANDI inventory failed after retries: " + " | ".join(errors))


def _extract_with_retries(
    asset: DandiAsset,
    cache_dir: Path,
    *,
    attempts: int = 3,
) -> tuple[WormIDRecording, dict[str, object], int]:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            recording, audit = extract_coordinate_cache(asset, cache_dir)
            return recording, audit, attempt
        except Exception as error:  # pragma: no cover - network contingency
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if attempt < attempts:
                print(
                    f"{asset.recording_id} {errors[-1]}", file=sys.stderr, flush=True
                )
    raise RuntimeError(
        f"coordinate extraction failed for {asset.recording_id}: " + " | ".join(errors)
    )


def run_prospective_expansion(
    *,
    repository_root: str | Path,
    cache_dir: str | Path,
    output_path: str | Path,
    workers: int = 1,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Verify the freeze, extract coordinates, benchmark, analyze, and write JSON."""

    root = Path(repository_root).resolve()
    freeze = verify_frozen_protocol(root)
    config_path = root / "configs/wormid_prospective.yaml"
    config = load_frozen_config(config_path)
    dataset = config["dataset"]
    assert isinstance(dataset, dict)
    assets, _ = _load_assets_with_retries(
        str(dataset["dandiset"]), str(dataset["version"])
    )
    validated = _validate_inventory_and_config(config, assets)
    tracker_config = _tracker_config(config)
    asset_by_id = {asset.recording_id: asset for asset in assets}
    full_index = {asset.recording_id: index for index, asset in enumerate(assets)}
    legacy = tuple(str(value) for value in validated["legacy"])
    untouched = tuple(str(value) for value in validated["untouched"])
    nontuning = tuple(sorted((*legacy, *untouched)))

    recordings: dict[str, WormIDRecording] = {}
    cache_audits: dict[str, dict[str, object]] = {}
    for number, recording_id in enumerate(nontuning, start=1):
        if progress is not None:
            progress(f"coordinate cache {number:02d}/{len(nontuning)} {recording_id}")
        recording, audit, extraction_attempts = _extract_with_retries(
            asset_by_id[recording_id], Path(cache_dir)
        )
        del extraction_attempts  # retries are operational, not a scientific result field
        recordings[recording_id] = recording
        cache_audits[recording_id] = audit

    tasks = []
    for recording_id in nontuning:
        cohort = "legacy_evaluation" if recording_id in legacy else "untouched_expansion"
        tasks.append(
            (
                recordings[recording_id],
                cohort,
                full_index[recording_id],
                tuple(int(value) for value in validated["strides"]),
                int(validated["base_seed"]),
                tracker_config,
            )
        )
    if progress is not None:
        progress(
            f"tracking {len(tasks)} recordings x 5 strides x 4 methods "
            f"with {workers} worker(s)"
        )
    if workers == 1:
        grouped_rows = [_benchmark_recording(task) for task in tasks]
    else:
        if workers < 1:
            raise ValueError("workers must be positive")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            grouped_rows = list(executor.map(_benchmark_recording, tasks))
    rows = [row for group in grouped_rows for row in group]
    rows.sort(
        key=lambda row: (
            int(row["full_inventory_index"]),
            int(row["stride"]),
            METHODS.index(str(row["method"])),
        )
    )
    analysis = analyze_prospective_rows(
        rows,
        untouched_ids=untouched,
        legacy_ids=legacy,
        primary_samples=int(validated["primary_bootstrap_samples"]),
        primary_seed=int(validated["primary_bootstrap_seed"]),
        hierarchical_samples=int(validated["hierarchical_bootstrap_samples"]),
        hierarchical_seed=int(validated["hierarchical_bootstrap_seed"]),
        same_date_exclusion=str(validated["same_date_exclusion"]),
    )

    inventory_rows = []
    for index, asset in enumerate(assets):
        if asset.recording_id == validated["tuning"]:
            cohort = "tuning"
        elif asset.recording_id in legacy:
            cohort = "legacy_evaluation"
        else:
            cohort = "untouched_expansion"
        inventory_rows.append(
            {
                **asset_inventory_row(asset),
                "full_inventory_index": index,
                "cohort": cohort,
                "coordinate_cache": (
                    {
                        "cache_schema_version": cache_audits[asset.recording_id][
                            "cache_schema_version"
                        ],
                        "semantic_coordinate_sha256": cache_audits[asset.recording_id][
                            "semantic_coordinate_sha256"
                        ],
                        "derived_cache_byte_sha256": cache_audits[asset.recording_id][
                            "cache_sha256"
                        ],
                        "derived_cache_size": cache_audits[asset.recording_id][
                            "cache_size"
                        ],
                        "range_block_size_bytes": cache_audits[asset.recording_id][
                            "range_block_size_bytes"
                        ],
                        "range_requests_at_cache_creation": cache_audits[
                            asset.recording_id
                        ]["range_requests"],
                        "range_requested_bytes_at_cache_creation": cache_audits[
                            asset.recording_id
                        ]["range_requested_bytes"],
                        "range_requested_fraction_of_asset_at_cache_creation": cache_audits[
                            asset.recording_id
                        ]["range_requested_fraction_of_asset"],
                        "source_full_digest_locally_verified": False,
                    }
                    if asset.recording_id in cache_audits
                    else None
                ),
            }
        )
    primary_result = analysis["primary_untouched_accuracy_effect"]
    assert isinstance(primary_result, dict)
    expected_seeds = {
        (recording_id, stride): int(validated["base_seed"])
        + 1000 * full_index[recording_id]
        + stride
        for recording_id in nontuning
        for stride in validated["strides"]
    }
    seeds_match = all(
        int(row["permutation_seed"])
        == expected_seeds[(str(row["recording_id"]), int(row["stride"]))]
        for row in rows
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "analysis_name": "frozen untouched-cohort WormID geometry replication",
        "identity_semantics": IDENTITY_SEMANTICS,
        "interpretation_boundary": (
            "transfer across animals and six dates within one EY/DANDI acquisition source; "
            "not semantic-name, detector-error, merge/split, cross-device, or "
            "cross-laboratory performance"
        ),
        "freeze_integrity": freeze,
        "protocol_sha256": freeze["files"][
            "docs/prospective_expansion_protocol.md"
        ]["observed_sha256"],
        "config_sha256": freeze["files"]["configs/wormid_prospective.yaml"][
            "observed_sha256"
        ],
        "code_sha256": {
            "identity_blind_sequence_source": sha256_file(root / "src/deformeot/wormid.py"),
            "track_sequence_source": sha256_file(
                root / "src/deformeot/wormid_benchmark.py"
            ),
            "range_extractor_source": sha256_file(
                root / "src/deformeot/wormid_dandi.py"
            ),
            "prospective_analysis_source": sha256_file(
                root / "src/deformeot/wormid_prospective.py"
            ),
        },
        "dandiset": {
            "identifier": str(dataset["dandiset"]),
            "immutable_version": str(dataset["version"]),
            "assets": inventory_rows,
        },
        "frozen_association": {
            "methods": list(validated["methods"]),
            "primary_baseline": PRIMARY_BASELINE,
            "candidate": CANDIDATE,
            "strides": list(validated["strides"]),
            "base_seed": validated["base_seed"],
            "seed_rule": (
                "base_seed + 1000*lexicographic_full_inventory_index + stride"
            ),
            "geometry": asdict(tracker_config),
            "identity_blind_sequence_modified": False,
            "track_sequence_modified": False,
        },
        "cohorts": {
            "tuning_recording": validated["tuning"],
            "legacy_evaluation_recordings": list(legacy),
            "untouched_expansion_recordings": list(untouched),
        },
        "results": rows,
        **analysis,
        "outcome_audit": {
            "freeze_verified_before_reading_config_or_computing_outcomes": True,
            "internally_frozen_not_externally_timestamped": True,
            "all_21_published_assets_in_inventory": len(assets) == 21,
            "all_16_untouched_recordings_evaluated": len(untouched) == 16
            and {str(row["recording_id"]) for row in rows if row["cohort"] == "untouched_expansion"}
            == set(untouched),
            "all_4_legacy_recordings_rerun_for_stratified_summary": len(legacy) == 4
            and {str(row["recording_id"]) for row in rows if row["cohort"] == "legacy_evaluation"}
            == set(legacy),
            "outcome_based_exclusions": [],
            "structural_or_network_failures": [],
            "permutation_seeds_match_frozen_full_inventory_rule": seeds_match,
            "semantic_labels_used_for_association": False,
            "fluorescence_used_for_association": False,
            "source_full_sha256_locally_verified": False,
            "source_digest_basis": (
                "SHA-256 and DANDI etag declared by immutable DANDI asset metadata; "
                "coordinate caches locally hashed; full movies intentionally not downloaded"
            ),
            "primary_replication_criterion_met": bool(
                primary_result["replication_criterion_met"]
            ),
            "primary_ci95_low": float(primary_result["ci95_low"]),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--cache-dir",
        default="/mnt/d/CodexData/wha_neuron_data/wormid_ey_geometry",
    )
    parser.add_argument(
        "--output", default="work/results/wormid_prospective_expansion.json"
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_prospective_expansion(
        repository_root=args.repository_root,
        cache_dir=args.cache_dir,
        output_path=args.output,
        workers=args.workers,
    )
    print(json.dumps(payload["primary_untouched_accuracy_effect"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
