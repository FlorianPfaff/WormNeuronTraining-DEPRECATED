"""Leakage-controlled neural-signal consequences of WormID tracking.

The association methods in :mod:`deformeot.wormid_benchmark` consume geometry
only.  This module opens fluorescence *after* an association has been produced,
permutes each frame's fluorescence into the same anonymous detection order, and
then reconstructs one time series per initialized track.  This separation is
important: fluorescence is an outcome of tracking here, never an association
feature or a tuning signal.

Raw-fluorescence correlation is the primary downstream measure.  Correlation of
the first temporal difference is a stricter trace-mixing measure that discounts
static offsets.  A robust positive-derivative event score is included as an
explicitly exploratory timing proxy; it is not presented as spike inference.
Uncertainty intervals resample complete recordings, never neurons or frames.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np
from scipy.stats import rankdata

from .wormid import IdentityBlindSequence, identity_blind_sequence, load_wormid_recording
from .wormid_benchmark import (
    METHODS,
    ReferenceTrackerConfig,
    score_tracks,
    track_sequence,
)


RAW_FLUORESCENCE_SERIES = (
    "processing/CalciumActivity/SignalRawFluor/SignalCalciumImResponseSeries"
)
NEURON_LABELS = "processing/CalciumActivity/NeuronIDs/labels"


@dataclass(frozen=True)
class RawFluorescence:
    """Raw WormID fluorescence in canonical ROI-index order.

    ``values`` has shape ``(samples, neurons)``.  Non-finite values are retained
    deliberately: downstream metrics use pairwise-finite samples and report all
    exclusions rather than silently imputing a neural signal.
    """

    path: str
    values: np.ndarray
    times_seconds: np.ndarray
    roi_indices: tuple[int, ...]
    conversion: float
    offset: float

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        times = np.asarray(self.times_seconds, dtype=float)
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("values must have shape (samples, neurons>=1)")
        if times.shape != (values.shape[0],):
            raise ValueError("times_seconds must have one value per sample")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("times_seconds must be finite and strictly increasing")
        if tuple(self.roi_indices) != tuple(range(values.shape[1])):
            raise ValueError("values must be canonicalized to ROI indices 0..N-1")
        if not np.isfinite(self.conversion) or not np.isfinite(self.offset):
            raise ValueError("conversion and offset must be finite")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "times_seconds", times.copy())

    @property
    def samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def neurons(self) -> int:
        return int(self.values.shape[1])

    @property
    def nonfinite_values(self) -> int:
        return int(np.count_nonzero(~np.isfinite(self.values)))


def load_wormid_raw_fluorescence(
    path: str | Path,
    *,
    series_path: str = RAW_FLUORESCENCE_SERIES,
) -> RawFluorescence:
    """Read a WormID ``RoiResponseSeries`` and canonicalize its ROI columns."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with h5py.File(source_path, "r") as nwb:
        for required in (series_path, NEURON_LABELS):
            if required not in nwb:
                raise ValueError(f"{source_path} is missing NWB path {required}")
        series = nwb[series_path]
        for field in ("data", "rois", "starting_time"):
            if field not in series:
                raise ValueError(f"{series.name} is missing {field}")
        dataset = series["data"]
        values = np.asarray(dataset[:], dtype=float)
        if values.ndim != 2:
            raise ValueError(f"{dataset.name} must have time-by-ROI shape")
        conversion = float(dataset.attrs.get("conversion", 1.0))
        offset = float(dataset.attrs.get("offset", 0.0))
        if not np.isfinite(conversion) or not np.isfinite(offset):
            raise ValueError("raw-fluorescence conversion and offset must be finite")
        values = values * conversion + offset

        roi_ids = np.asarray(series["rois"][:], dtype=np.int64)
        neuron_count = int(len(nwb[NEURON_LABELS]))
        if roi_ids.shape != (values.shape[1],):
            raise ValueError("RoiResponseSeries rois must identify every data column")
        if not np.array_equal(np.sort(roi_ids), np.arange(neuron_count)):
            raise ValueError("raw fluorescence must contain each canonical ROI exactly once")
        canonical = np.empty((values.shape[0], neuron_count), dtype=float)
        canonical[:, roi_ids] = values

        starting_time = series["starting_time"]
        start = float(starting_time[()])
        rate = float(starting_time.attrs.get("rate", np.nan))
        if not np.isfinite(start) or not np.isfinite(rate) or rate <= 0:
            raise ValueError("raw-fluorescence starting_time must define a positive rate")
        times = start + np.arange(len(canonical), dtype=float) / rate

    return RawFluorescence(
        path=str(source_path.resolve()),
        values=canonical,
        times_seconds=times,
        roi_indices=tuple(range(neuron_count)),
        conversion=conversion,
        offset=offset,
    )


def anonymous_detection_fluorescence(
    raw: RawFluorescence,
    sequence: IdentityBlindSequence,
) -> tuple[np.ndarray, ...]:
    """Materialize fluorescence in anonymous detection-row order.

    This post-association adapter uses hidden permutations solely because the NWB
    stores fluorescence by canonical ROI column.  The returned arrays contain no
    identity fields: element ``d`` is simply the signal measured at anonymous
    detection row ``d`` in that frame.  Trackers must run before this function is
    called and must never receive its return value.
    """

    if len(sequence.source_frame_indices) != len(sequence.truth_indices):
        raise ValueError("sequence frame metadata have inconsistent lengths")
    signals: list[np.ndarray] = []
    for source_frame, hidden_order in zip(
        sequence.source_frame_indices, sequence.truth_indices
    ):
        if source_frame < 0 or source_frame >= raw.samples:
            raise IndexError("sequence source frame is outside fluorescence samples")
        hidden_order = np.asarray(hidden_order, dtype=int)
        if hidden_order.ndim != 1 or np.any(hidden_order < 0) or np.any(
            hidden_order >= raw.neurons
        ):
            raise IndexError("hidden detection permutation contains an invalid ROI")
        if len(np.unique(hidden_order)) != len(hidden_order):
            raise ValueError("hidden detection permutation contains duplicate ROIs")
        signals.append(raw.values[source_frame, hidden_order].copy())
    return tuple(signals)


def reconstruct_track_traces(
    detection_fluorescence: Sequence[Sequence[float] | np.ndarray],
    assignments: Sequence[Sequence[int] | np.ndarray],
    *,
    tracks: int | None = None,
) -> np.ndarray:
    """Gather anonymous detection signals into persistent track rows.

    Each assignment maps ``canonical track row -> detection row``.  ``-1`` is a
    missing observation and becomes ``NaN``; other negative values, duplicate
    detections, and out-of-range rows fail loudly.  The output shape is
    ``(tracks, frames)``.
    """

    if len(detection_fluorescence) != len(assignments) or not assignments:
        raise ValueError("signals and assignments need equal nonzero frame counts")
    if tracks is None:
        first = np.asarray(assignments[0])
        if first.ndim != 1:
            raise ValueError("assignment arrays must be one-dimensional")
        tracks = int(len(first))
    if tracks < 1:
        raise ValueError("tracks must be positive")

    reconstructed = np.full((tracks, len(assignments)), np.nan, dtype=float)
    for frame_index, (signals, assignment) in enumerate(
        zip(detection_fluorescence, assignments)
    ):
        signals = np.asarray(signals, dtype=float)
        assignment = np.asarray(assignment)
        if signals.ndim != 1 or assignment.shape != (tracks,):
            raise ValueError("each signal/assignment frame must be one-dimensional")
        if not np.issubdtype(assignment.dtype, np.integer):
            if not np.all(np.isfinite(assignment)) or not np.all(
                assignment == np.floor(assignment)
            ):
                raise ValueError("assignment rows must be integers")
        assignment = assignment.astype(int)
        if np.any(assignment < -1) or np.any(assignment >= len(signals)):
            raise IndexError("assignment refers to an invalid detection row")
        observed = assignment >= 0
        if len(np.unique(assignment[observed])) != int(np.count_nonzero(observed)):
            raise ValueError("a detection cannot supply multiple tracks in one frame")
        reconstructed[observed, frame_index] = signals[assignment[observed]]
    return reconstructed


def ground_truth_track_traces(
    raw: RawFluorescence,
    sequence: IdentityBlindSequence,
) -> np.ndarray:
    """Return held-out truth traces in initialized track-row order."""

    canonical_truth = np.asarray(sequence.truth_indices[0], dtype=int)
    if canonical_truth.ndim != 1 or len(canonical_truth) != raw.neurons:
        raise ValueError("initialized WormID cohort must contain every raw-fluorescence ROI")
    source_frames = np.asarray(sequence.source_frame_indices, dtype=int)
    if np.any(source_frames < 0) or np.any(source_frames >= raw.samples):
        raise IndexError("sequence source frame is outside fluorescence samples")
    return raw.values[source_frames][:, canonical_truth].T.copy()


def trace_source_contamination(
    sequence: IdentityBlindSequence,
    assignments: Sequence[Sequence[int] | np.ndarray],
) -> dict[str, float | int | None]:
    """Count trace samples sourced from a neuron other than the seeded identity."""

    if len(assignments) != len(sequence.truth_indices):
        raise ValueError("assignments must have one row per sequence frame")
    canonical = np.asarray(sequence.truth_indices[0], dtype=int)
    contaminated_by_track = np.zeros(len(canonical), dtype=int)
    observed_by_track = np.zeros(len(canonical), dtype=int)
    missing = 0
    for hidden_order, assignment in zip(sequence.truth_indices, assignments):
        hidden_order = np.asarray(hidden_order, dtype=int)
        assignment = np.asarray(assignment, dtype=int)
        if assignment.shape != canonical.shape:
            raise ValueError("assignment shape does not match initialized cohort")
        if np.any(assignment < -1) or np.any(assignment >= len(hidden_order)):
            raise IndexError("assignment refers to an invalid detection row")
        observed = assignment >= 0
        missing += int(np.count_nonzero(~observed))
        sourced = np.full(len(canonical), -1, dtype=int)
        sourced[observed] = hidden_order[assignment[observed]]
        contaminated_by_track += observed & (sourced != canonical)
        observed_by_track += observed
    fractions = np.divide(
        contaminated_by_track,
        observed_by_track,
        out=np.full(len(canonical), np.nan),
        where=observed_by_track > 0,
    )
    valid = fractions[np.isfinite(fractions)]
    observed_total = int(observed_by_track.sum())
    contaminated_total = int(contaminated_by_track.sum())
    return {
        "samples_total": int(len(canonical) * len(assignments)),
        "samples_observed": observed_total,
        "samples_missing": int(missing),
        "samples_from_wrong_neuron": contaminated_total,
        "foreign_source_fraction": float(contaminated_total / observed_total)
        if observed_total
        else None,
        "median_track_foreign_source_fraction": float(np.median(valid))
        if len(valid)
        else None,
    }


def _as_trace_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2:
        raise ValueError("traces must have shape (tracks, samples)")
    return matrix


def _is_constant(values: np.ndarray, tolerance: float) -> bool:
    scale = max(1.0, float(np.max(np.abs(values))))
    return bool(float(np.ptp(values)) <= tolerance * scale)


def trace_correlation_metrics(
    predicted: Sequence[Sequence[float]] | np.ndarray,
    truth: Sequence[Sequence[float]] | np.ndarray,
    *,
    min_valid_samples: int = 3,
    constant_tolerance: float = 1e-12,
    include_per_track: bool = True,
) -> dict[str, object]:
    """Compute per-track Pearson and Spearman correlation with audited exclusions.

    Samples are pairwise-finite; no imputation is performed.  A track is excluded
    from both correlation aggregates when fewer than ``min_valid_samples`` pairs
    remain or either trace is numerically constant.  Counts for every exclusion
    reason are returned.
    """

    predicted = _as_trace_matrix(predicted)
    truth = _as_trace_matrix(truth)
    if predicted.shape != truth.shape:
        raise ValueError("predicted and truth traces must have identical shape")
    if min_valid_samples < 2 or constant_tolerance < 0:
        raise ValueError("invalid sample-count or constant tolerance")

    rows: list[dict[str, object]] = []
    pearson_values: list[float] = []
    spearman_values: list[float] = []
    exclusion_counts: dict[str, int] = {
        "insufficient_valid_samples": 0,
        "constant_ground_truth": 0,
        "constant_prediction": 0,
        "constant_ground_truth_and_prediction": 0,
    }
    finite_pairs = 0
    for track_index, (prediction, target) in enumerate(zip(predicted, truth)):
        valid = np.isfinite(prediction) & np.isfinite(target)
        valid_count = int(np.count_nonzero(valid))
        finite_pairs += valid_count
        x = prediction[valid]
        y = target[valid]
        status = "ok"
        pearson: float | None = None
        spearman: float | None = None
        if valid_count < min_valid_samples:
            status = "insufficient_valid_samples"
        else:
            constant_prediction = _is_constant(x, constant_tolerance)
            constant_truth = _is_constant(y, constant_tolerance)
            if constant_prediction and constant_truth:
                status = "constant_ground_truth_and_prediction"
            elif constant_truth:
                status = "constant_ground_truth"
            elif constant_prediction:
                status = "constant_prediction"
            else:
                x_centered = x - np.mean(x)
                y_centered = y - np.mean(y)
                pearson = float(
                    np.dot(x_centered, y_centered)
                    / np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
                )
                x_rank = rankdata(x, method="average")
                y_rank = rankdata(y, method="average")
                x_rank -= np.mean(x_rank)
                y_rank -= np.mean(y_rank)
                spearman = float(
                    np.dot(x_rank, y_rank)
                    / np.sqrt(np.dot(x_rank, x_rank) * np.dot(y_rank, y_rank))
                )
                pearson = float(np.clip(pearson, -1.0, 1.0))
                spearman = float(np.clip(spearman, -1.0, 1.0))
                pearson_values.append(pearson)
                spearman_values.append(spearman)
        if status != "ok":
            exclusion_counts[status] += 1
        rows.append(
            {
                "track_index": int(track_index),
                "valid_samples": valid_count,
                "nonfinite_pairs": int(len(prediction) - valid_count),
                "status": status,
                "pearson": pearson,
                "spearman": spearman,
            }
        )

    def aggregate(values: list[float], operation: str) -> float | None:
        if not values:
            return None
        return float(np.median(values) if operation == "median" else np.mean(values))

    summary: dict[str, object] = {
        "tracks_total": int(predicted.shape[0]),
        "samples_per_track": int(predicted.shape[1]),
        "tracks_valid": int(len(pearson_values)),
        "pairwise_finite_samples": int(finite_pairs),
        "pairwise_finite_fraction": float(finite_pairs / predicted.size)
        if predicted.size
        else None,
        "pearson_median": aggregate(pearson_values, "median"),
        "pearson_mean": aggregate(pearson_values, "mean"),
        "spearman_median": aggregate(spearman_values, "median"),
        "spearman_mean": aggregate(spearman_values, "mean"),
        "exclusions": exclusion_counts,
        "nan_policy": "pairwise finite; no imputation",
        "constant_policy": "exclude track from correlation aggregate",
    }
    if include_per_track:
        summary["per_track"] = rows
    return summary


def first_difference_rate(traces: np.ndarray, times_seconds: np.ndarray) -> np.ndarray:
    """First temporal difference divided by elapsed seconds."""

    traces = _as_trace_matrix(traces)
    times = np.asarray(times_seconds, dtype=float)
    if times.shape != (traces.shape[1],):
        raise ValueError("times_seconds must match the trace sample axis")
    intervals = np.diff(times)
    if not np.all(np.isfinite(times)) or np.any(intervals <= 0):
        raise ValueError("times_seconds must be finite and strictly increasing")
    return np.diff(traces, axis=1) / intervals[None, :]


def _positive_derivative_events(
    trace: np.ndarray,
    times_seconds: np.ndarray,
    *,
    threshold_z: float,
    minimum_separation_seconds: float,
    constant_tolerance: float,
) -> tuple[np.ndarray, str]:
    derivative = np.diff(trace) / np.diff(times_seconds)
    derivative_times = times_seconds[1:]
    finite = np.isfinite(derivative)
    values = derivative[finite]
    if len(values) < 3:
        return np.empty(0, dtype=float), "insufficient_valid_derivatives"
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    magnitude = max(1.0, float(np.max(np.abs(values))))
    if scale <= constant_tolerance * magnitude:
        return np.empty(0, dtype=float), "constant_derivative"
    threshold = center + threshold_z * scale
    candidates: list[int] = []
    for index in np.flatnonzero(finite & (derivative > threshold)):
        left = derivative[index - 1] if index > 0 else -np.inf
        right = derivative[index + 1] if index + 1 < len(derivative) else -np.inf
        if (not np.isfinite(left) or derivative[index] >= left) and (
            not np.isfinite(right) or derivative[index] >= right
        ):
            candidates.append(int(index))
    # Retain the strongest candidate in each temporal refractory neighbourhood.
    accepted: list[int] = []
    for index in sorted(candidates, key=lambda value: derivative[value], reverse=True):
        event_time = derivative_times[index]
        if all(
            abs(event_time - derivative_times[other]) >= minimum_separation_seconds
            for other in accepted
        ):
            accepted.append(index)
    accepted.sort()
    return derivative_times[np.asarray(accepted, dtype=int)], "ok"


def _event_matches(predicted: np.ndarray, truth: np.ndarray, tolerance: float) -> int:
    predicted = np.sort(np.asarray(predicted, dtype=float))
    truth = np.sort(np.asarray(truth, dtype=float))
    predicted_index = 0
    truth_index = 0
    matches = 0
    while predicted_index < len(predicted) and truth_index < len(truth):
        difference = predicted[predicted_index] - truth[truth_index]
        if abs(difference) <= tolerance:
            matches += 1
            predicted_index += 1
            truth_index += 1
        elif difference < -tolerance:
            predicted_index += 1
        else:
            truth_index += 1
    return matches


def event_timing_metrics(
    predicted: Sequence[Sequence[float]] | np.ndarray,
    truth: Sequence[Sequence[float]] | np.ndarray,
    times_seconds: Sequence[float] | np.ndarray,
    *,
    threshold_z: float = 3.0,
    tolerance_seconds: float | None = None,
    minimum_separation_seconds: float | None = None,
    constant_tolerance: float = 1e-12,
    include_per_track: bool = True,
) -> dict[str, object]:
    """Score robust positive-derivative event timing as an exploratory proxy."""

    predicted = _as_trace_matrix(predicted)
    truth = _as_trace_matrix(truth)
    times = np.asarray(times_seconds, dtype=float)
    if predicted.shape != truth.shape or times.shape != (predicted.shape[1],):
        raise ValueError("predicted, truth, and times have incompatible shapes")
    intervals = np.diff(times)
    if len(intervals) < 1 or np.any(~np.isfinite(times)) or np.any(intervals <= 0):
        raise ValueError("at least two strictly increasing finite times are required")
    cadence = float(np.median(intervals))
    if tolerance_seconds is None:
        tolerance_seconds = 1.01 * cadence
    if minimum_separation_seconds is None:
        minimum_separation_seconds = 2.0 * cadence
    if threshold_z <= 0 or tolerance_seconds < 0 or minimum_separation_seconds < 0:
        raise ValueError("event parameters must be nonnegative and threshold positive")

    rows: list[dict[str, object]] = []
    total_matches = 0
    total_predicted = 0
    total_truth = 0
    eventful_tracks = 0
    for track_index, (prediction, target) in enumerate(zip(predicted, truth)):
        prediction_events, prediction_status = _positive_derivative_events(
            prediction,
            times,
            threshold_z=threshold_z,
            minimum_separation_seconds=minimum_separation_seconds,
            constant_tolerance=constant_tolerance,
        )
        truth_events, truth_status = _positive_derivative_events(
            target,
            times,
            threshold_z=threshold_z,
            minimum_separation_seconds=minimum_separation_seconds,
            constant_tolerance=constant_tolerance,
        )
        matches = _event_matches(prediction_events, truth_events, tolerance_seconds)
        predicted_count = int(len(prediction_events))
        truth_count = int(len(truth_events))
        total_matches += matches
        total_predicted += predicted_count
        total_truth += truth_count
        if truth_count:
            eventful_tracks += 1
            precision = float(matches / predicted_count) if predicted_count else 0.0
            recall = float(matches / truth_count)
            f1 = float(2 * precision * recall / (precision + recall)) if matches else 0.0
            status = "ok"
        else:
            precision = None
            recall = None
            f1 = None
            status = "no_reference_events"
        rows.append(
            {
                "track_index": int(track_index),
                "reference_events": truth_count,
                "predicted_events": predicted_count,
                "matched_events": int(matches),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "status": status,
                "prediction_detection_status": prediction_status,
                "reference_detection_status": truth_status,
            }
        )
    precision = float(total_matches / total_predicted) if total_predicted else None
    recall = float(total_matches / total_truth) if total_truth else None
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    summary: dict[str, object] = {
        "tracks_total": int(predicted.shape[0]),
        "tracks_with_reference_events": int(eventful_tracks),
        "reference_events": int(total_truth),
        "predicted_events": int(total_predicted),
        "matched_events": int(total_matches),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "threshold": f"median derivative + {threshold_z:g} * 1.4826*MAD",
        "tolerance_seconds": float(tolerance_seconds),
        "minimum_separation_seconds": float(minimum_separation_seconds),
        "interpretation": "exploratory positive-derivative timing proxy; not spike inference",
    }
    if include_per_track:
        summary["per_track"] = rows
    return summary


def evaluate_wormid_assignments(
    raw: RawFluorescence,
    sequence: IdentityBlindSequence,
    assignments: Sequence[Sequence[int] | np.ndarray],
    *,
    include_per_track: bool = False,
) -> dict[str, object]:
    """Evaluate neural-signal fidelity after a geometry-only tracker has run."""

    detection_signals = anonymous_detection_fluorescence(raw, sequence)
    predicted = reconstruct_track_traces(
        detection_signals,
        assignments,
        tracks=len(sequence.initial_track_ids),
    )
    truth = ground_truth_track_traces(raw, sequence)
    source_frames = np.asarray(sequence.source_frame_indices, dtype=int)
    times = raw.times_seconds[source_frames]
    return {
        "raw": trace_correlation_metrics(
            predicted, truth, include_per_track=include_per_track
        ),
        "first_difference_per_second": trace_correlation_metrics(
            first_difference_rate(predicted, times),
            first_difference_rate(truth, times),
            include_per_track=include_per_track,
        ),
        "positive_derivative_events": event_timing_metrics(
            predicted, truth, times, include_per_track=include_per_track
        ),
        "source_contamination": trace_source_contamination(sequence, assignments),
        "leakage_control": (
            "tracking assignments were completed from anonymous geometry before raw "
            "fluorescence was materialized; fluorescence was not used for association "
            "or tuning"
        ),
    }


def _nested_metric(row: Mapping[str, object], path: str) -> float | None:
    value: object = row
    for field in path.split("."):
        if not isinstance(value, Mapping) or field not in value:
            raise KeyError(f"metric {path!r} is absent")
        value = value[field]
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def complete_recording_paired_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    improved: str,
    baseline: str,
    metric: str,
    strata: Sequence[str] = (),
    samples: int = 10_000,
    seed: int = 20260720,
) -> list[dict[str, object]]:
    """Bootstrap paired method differences by resampling complete recordings.

    A separate bootstrap is returned for each requested stratum (typically each
    temporal stride).  Recordings lacking either finite method value are reported
    and excluded as incomplete; neurons and frames are never resampled.
    """

    if samples < 1:
        raise ValueError("samples must be positive")
    grouped: dict[tuple[object, ...], dict[str, dict[str, float | None]]] = {}
    for row in rows:
        stratum = tuple(row[field] for field in strata)
        recording_id = str(row["recording_id"])
        method = str(row["method"])
        value = _nested_metric(row, metric)
        methods = grouped.setdefault(stratum, {}).setdefault(recording_id, {})
        if method in methods:
            raise ValueError("duplicate method/recording/stratum row")
        methods[method] = value
    output: list[dict[str, object]] = []
    for group_index, (stratum, recordings) in enumerate(sorted(grouped.items())):
        differences: list[float] = []
        incomplete: list[str] = []
        for recording_id, methods in sorted(recordings.items()):
            left = methods.get(improved)
            right = methods.get(baseline)
            if left is None or right is None:
                incomplete.append(recording_id)
            else:
                differences.append(float(left - right))
        if not differences:
            output.append(
                {
                    "stratum": dict(zip(strata, stratum)),
                    "improved": improved,
                    "baseline": baseline,
                    "metric": metric,
                    "recordings_available": int(len(recordings)),
                    "complete_recordings": 0,
                    "incomplete_recording_ids": incomplete,
                    "mean_paired_difference": None,
                    "median_paired_difference": None,
                    "ci95_low": None,
                    "ci95_high": None,
                    "bootstrap_samples": int(samples),
                    "resampling_unit": "complete recording",
                    "status": "unavailable: no finite complete pairs",
                }
            )
            continue
        difference_array = np.asarray(differences, dtype=float)
        rng = np.random.default_rng(seed + group_index)
        draws = rng.choice(
            difference_array,
            size=(samples, len(difference_array)),
            replace=True,
        ).mean(axis=1)
        output.append(
            {
                "stratum": dict(zip(strata, stratum)),
                "improved": improved,
                "baseline": baseline,
                "metric": metric,
                "recordings_available": int(len(recordings)),
                "complete_recordings": int(len(difference_array)),
                "incomplete_recording_ids": incomplete,
                "mean_paired_difference": float(np.mean(difference_array)),
                "median_paired_difference": float(np.median(difference_array)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "bootstrap_samples": int(samples),
                "resampling_unit": "complete recording",
            }
        )
    return output


def _frame_indices(frame_count: int, stride: int) -> tuple[int, ...]:
    if stride < 1:
        raise ValueError("stride must be positive")
    selected = tuple(range(0, frame_count, stride))
    return selected if selected[-1] == frame_count - 1 else (*selected, frame_count - 1)


def run_wormid_downstream_benchmark(
    paths: Iterable[str | Path],
    *,
    strides: Sequence[int],
    output_path: str | Path,
    config: ReferenceTrackerConfig = ReferenceTrackerConfig(reference_weight=6.0),
    methods: Sequence[str] = METHODS,
    seed: int = 20260720,
    bootstrap_samples: int = 10_000,
    tuning_recording_id: str | None = None,
) -> dict[str, object]:
    """Run geometry-only tracking, then evaluate held-out raw fluorescence."""

    paths = tuple(paths)
    if not paths:
        raise ValueError("at least one WormID recording is required")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for path_index, path in enumerate(paths):
        recording = load_wormid_recording(path)
        raw = load_wormid_raw_fluorescence(path)
        if raw.samples != recording.frames or raw.neurons != recording.neurons:
            raise ValueError("trajectory and raw-fluorescence arrays are not aligned")
        if not np.allclose(raw.times_seconds, recording.times_seconds, atol=1e-9, rtol=0):
            raise ValueError("trajectory and raw-fluorescence timestamps are not aligned")
        sources.append(
            {
                "recording_id": recording.recording_id,
                "path": recording.path,
                "frames": recording.frames,
                "neurons": recording.neurons,
                "raw_signal_path": RAW_FLUORESCENCE_SERIES + "/data",
                "raw_nonfinite_values": raw.nonfinite_values,
            }
        )
        for stride in strides:
            selected = _frame_indices(recording.frames, int(stride))
            sequence = identity_blind_sequence(
                recording,
                frame_indices=selected,
                seed=seed + 1000 * path_index + int(stride),
            )
            for method in methods:
                # Leakage boundary: this call completes before fluorescence is
                # passed into any downstream function.
                tracked = track_sequence(sequence, method, config=config)
                identity = asdict(score_tracks(sequence, tracked))
                downstream = evaluate_wormid_assignments(
                    raw,
                    sequence,
                    tracked.assignments,
                    include_per_track=False,
                )
                rows.append(
                    {
                        "recording_id": recording.recording_id,
                        "stride": int(stride),
                        "method": method,
                        "source_frames": len(selected),
                        "association_inputs": "anonymous 3D coordinates only",
                        "fluorescence_used_for_association_or_tuning": False,
                        "identity": identity,
                        **downstream,
                    }
                )

    if tuning_recording_id is None:
        tuning_recording_id = str(sources[0]["recording_id"])
    recording_ids = {str(source["recording_id"]) for source in sources}
    if tuning_recording_id not in recording_ids:
        raise ValueError("tuning_recording_id is not one of the input recordings")
    heldout_rows = [row for row in rows if row["recording_id"] != tuning_recording_id]

    metric_roles = {
        "raw.pearson_median": "preregistered primary",
        "raw.pearson_mean": "post-hoc descriptive (median ceiling diagnostic)",
        "first_difference_per_second.pearson_median": "post-hoc secondary",
        "first_difference_per_second.pearson_mean": "post-hoc descriptive",
        "positive_derivative_events.micro_f1": "post-hoc exploratory timing proxy",
        "source_contamination.foreign_source_fraction": "mechanistic tracking diagnostic",
    }

    def comparison_set(
        selected_rows: Sequence[Mapping[str, object]], analysis_set: str
    ) -> list[dict[str, object]]:
        comparisons: list[dict[str, object]] = []
        if "dual_anchor" not in methods or "first_affine" not in methods:
            return comparisons
        for metric, metric_role in metric_roles.items():
            items = complete_recording_paired_bootstrap(
                selected_rows,
                improved="dual_anchor",
                baseline="first_affine",
                metric=metric,
                strata=("stride",),
                samples=bootstrap_samples,
                seed=seed,
            )
            for item in items:
                item["analysis_set"] = analysis_set
                item["metric_role"] = metric_role
            comparisons.extend(items)
        return comparisons

    heldout_comparisons = comparison_set(
        heldout_rows, "confirmatory four held-out animals"
    )
    all_recording_comparisons = comparison_set(rows, "all-five sensitivity analysis")
    primary_stride_one = next(
        (
            item
            for item in heldout_comparisons
            if item["metric"] == "raw.pearson_median"
            and item["stratum"] == {"stride": 1}
        ),
        None,
    )
    primary_gain = (
        primary_stride_one["mean_paired_difference"] if primary_stride_one else None
    )
    primary_passed = primary_gain is not None and float(primary_gain) >= 0.03
    payload: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "association_config_frozen": asdict(config),
        "methods": list(methods),
        "sources": sources,
        "results": rows,
        "tuning_recording_id": tuning_recording_id,
        "heldout_recording_ids": sorted(recording_ids - {tuning_recording_id}),
        "confirmatory_heldout_bootstrap": heldout_comparisons,
        "all_five_sensitivity_bootstrap": all_recording_comparisons,
        "primary_target_assessment": {
            "metric": "recording median per-neuron raw Pearson correlation",
            "comparison": "dual_anchor minus first_affine",
            "temporal_stride": 1,
            "target_difference": 0.03,
            "observed_mean_paired_difference": primary_gain,
            "ci95_low": primary_stride_one["ci95_low"] if primary_stride_one else None,
            "ci95_high": primary_stride_one["ci95_high"] if primary_stride_one else None,
            "complete_heldout_recordings": (
                primary_stride_one["complete_recordings"] if primary_stride_one else 0
            ),
            "status": "PASSED" if primary_passed else "FAILED",
            "interpretation": (
                "The median has a ceiling near 1 because most neuron traces are never "
                "mixed. Post-hoc mean, first-difference, event, or contamination metrics "
                "must not be used to rescue a failed primary endpoint."
            ),
        },
        "primary_downstream_metric": "recording median per-neuron raw Pearson correlation",
        "secondary_metrics": [
            "recording mean per-neuron raw Pearson correlation",
            "median per-neuron Pearson correlation of first-difference-per-second traces",
            "mean per-neuron Pearson correlation of first-difference-per-second traces",
            "fraction of observed samples sourced from the wrong neuron",
        ],
        "exploratory_metric": (
            "robust positive-derivative event timing F1; this is not spike inference"
        ),
        "leakage_control": (
            "all associations used anonymous coordinates only and were completed before "
            "raw fluorescence was accessed for outcome evaluation"
        ),
        "missing_and_constant_policy": (
            "missing assignments and non-finite values become missing samples; correlations "
            "use pairwise-finite samples with no imputation; constant or undersampled tracks "
            "are excluded and counted"
        ),
        "caveats": [
            "Raw fluorescence is an ROI-table value, not a fresh pixel-level re-extraction.",
            "The event score is a cadence-dependent derivative proxy, not deconvolved spikes.",
            "Five animals give coarse animal-level confidence intervals.",
            "All methods share first-frame initialization and an exhaustive fixed cohort.",
        ],
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
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--reference-weight", type=float, default=6.0)
    parser.add_argument("--descriptor-weight", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--tuning-recording-id",
        help="animal used to freeze settings; defaults to the first input path",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_wormid_downstream_benchmark(
        args.paths,
        strides=args.strides,
        methods=args.methods,
        output_path=args.output,
        config=ReferenceTrackerConfig(
            reference_weight=args.reference_weight,
            descriptor_weight=args.descriptor_weight,
        ),
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        tuning_recording_id=args.tuning_recording_id,
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
