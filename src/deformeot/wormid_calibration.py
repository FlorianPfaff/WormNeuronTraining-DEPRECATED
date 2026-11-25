"""Frozen-split confidence calibration for dual-anchor extent tracking.

The association algorithm in this module mirrors
:func:`deformeot.wormid_extent.track_dual_anchor_extent`, while retaining only
identity-blind diagnostics from each solved cost matrix.  Correct/incorrect
labels are joined *after* tracking and are used to fit a confidence model on one
declared calibration animal.  No identity label, neural trace, or held-out
outcome enters association or confidence-feature construction.

The command-line benchmark uses blocked out-of-fold predictions on the
calibration recording to select one L2 regularization value.  It then fits the
frozen logistic model on that recording and evaluates four complete held-out
recordings.  Confidence intervals resample complete recordings, never neurons,
frames, or associations independently.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import (
    adaptive_bin_ece,
    brier_score,
    fixed_bin_ece,
    grouped_bootstrap,
    negative_log_likelihood,
    risk_coverage_curve,
)
from .wormid import IdentityBlindSequence, identity_blind_sequence, load_wormid_recording
from .wormid_benchmark import ReferenceTrackerConfig, TrackResult
from .wormid_extent import (
    AnonymousPatchSequence,
    DualAnchorExtentConfig,
    PatchFeatureSet,
    RobustFeatureScaler,
    fit_robust_feature_scaler,
    pair_patch_features_to_anonymous_detections,
)
from .wormid_extent import _reference_affine as reference_affine
from .wormid_extent import _reference_spatial_cost as reference_spatial_cost


FEATURE_NAMES = (
    "selected_total_cost",
    "row_alternative_margin",
    "column_alternative_margin",
    "selected_temporal_cost",
    "selected_reference_cost",
    "selected_extent_cost",
    "selected_pair_reliability",
    "temporal_selected_is_row_best",
    "reference_selected_is_row_best",
    "extent_selected_is_row_best",
    "temporal_reference_row_best_agree",
    "selected_row_rank_fraction",
    "selected_column_rank_fraction",
    "selected_row_softmax_mass",
    "selected_column_softmax_mass",
    "temporal_reference_prediction_distance",
    "cloud_motion_um",
    "source_frame_gap",
    "reference_agreement_streak_fraction",
    "reference_disagreement_fraction",
)


@dataclass(frozen=True)
class AssociationDiagnosticEvents:
    """Identity-blind diagnostics aligned to track rows after initialization."""

    features: np.ndarray
    source_frame_indices: np.ndarray
    track_rows: np.ndarray
    selected_detection_rows: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=float)
        source = np.asarray(self.source_frame_indices, dtype=np.int64)
        tracks = np.asarray(self.track_rows, dtype=np.int64)
        detections = np.asarray(self.selected_detection_rows, dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("diagnostic feature matrix has the wrong shape")
        if any(values.shape != (len(features),) for values in (source, tracks, detections)):
            raise ValueError("diagnostic index vectors must align to feature rows")
        if not np.all(np.isfinite(features)):
            raise ValueError("diagnostic features must be finite")
        object.__setattr__(self, "features", features.copy())
        object.__setattr__(self, "source_frame_indices", source.copy())
        object.__setattr__(self, "track_rows", tracks.copy())
        object.__setattr__(self, "selected_detection_rows", detections.copy())


@dataclass(frozen=True)
class LabeledDiagnosticEvents:
    """Diagnostics joined to hidden truth strictly after association."""

    features: np.ndarray
    correctness: np.ndarray
    recording_ids: np.ndarray
    strides: np.ndarray
    source_frame_indices: np.ndarray


def _alternative_margin(cost: np.ndarray, selected: np.ndarray, *, axis: int) -> np.ndarray:
    """Difference between the cheapest unselected alternative and selected cost."""

    count = len(selected)
    masked = np.asarray(cost, dtype=float).copy()
    if axis == 1:
        selected_cost = masked[np.arange(count), selected]
        masked[np.arange(count), selected] = np.inf
        return np.min(masked, axis=1) - selected_cost
    if axis == 0:
        selected_cost = masked[np.arange(count), selected]
        masked[np.arange(count), selected] = np.inf
        # Row i selected column selected[i].  Read the best competing row for
        # that column and return results in track-row order.
        return np.min(masked, axis=0)[selected] - selected_cost
    raise ValueError("axis must be zero or one")


def _selected_softmax_mass(cost: np.ndarray, selected: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = np.asarray(cost, dtype=float)
    if axis == 1:
        shifted = shifted - np.min(shifted, axis=1, keepdims=True)
        weights = np.exp(-np.clip(shifted, 0.0, 700.0))
        return weights[np.arange(len(selected)), selected] / np.sum(weights, axis=1)
    if axis == 0:
        shifted = shifted - np.min(shifted, axis=0, keepdims=True)
        weights = np.exp(-np.clip(shifted, 0.0, 700.0))
        return weights[np.arange(len(selected)), selected] / np.sum(weights, axis=0)[selected]
    raise ValueError("axis must be zero or one")


def track_dual_anchor_extent_with_diagnostics(
    sequence: IdentityBlindSequence,
    anonymous_features: AnonymousPatchSequence,
    scaler: RobustFeatureScaler,
    *,
    geometry_config: ReferenceTrackerConfig = ReferenceTrackerConfig(),
    extent_config: DualAnchorExtentConfig = DualAnchorExtentConfig(),
) -> tuple[TrackResult, AssociationDiagnosticEvents]:
    """Track exactly as the production method and retain label-free diagnostics."""

    if not sequence.frames:
        raise ValueError("sequence is empty")
    if len(anonymous_features.descriptors) != len(sequence.frames) or len(
        anonymous_features.reliability
    ) != len(sequence.frames):
        raise ValueError("anonymous features must contain one array per frame")
    reference = np.asarray(sequence.frames[0].positions, dtype=float)
    count = len(reference)
    if extent_config.feature_indices is None:
        indices = np.arange(len(scaler.center), dtype=int)
    else:
        indices = np.asarray(extent_config.feature_indices, dtype=int)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("feature_indices must select at least one feature")

    scaled = tuple(
        scaler.transform(np.asarray(values, dtype=float))[:, indices]
        for values in anonymous_features.descriptors
    )
    qualities = tuple(
        np.asarray(values, dtype=float) for values in anonymous_features.reliability
    )
    for frame, values, quality in zip(sequence.frames, scaled, qualities, strict=True):
        if values.shape[0] != len(frame.positions) or quality.shape != (len(frame.positions),):
            raise ValueError("anonymous feature rows must match detection rows")

    started = perf_counter()
    assignments: list[np.ndarray] = [np.arange(count, dtype=np.int64)]
    previous = reference.copy()
    template = scaled[0].copy()
    template_quality = qualities[0].copy()
    cloud_motion: list[float] = [0.0]
    feature_blocks: list[np.ndarray] = []
    source_blocks: list[np.ndarray] = []
    track_blocks: list[np.ndarray] = []
    detection_blocks: list[np.ndarray] = []
    reference_agreement_streak = np.zeros(count, dtype=float)
    reference_disagreement_count = np.zeros(count, dtype=float)

    for local_frame, frame in enumerate(sequence.frames[1:], start=1):
        current = np.asarray(frame.positions, dtype=float)
        forward = np.min(cdist(previous, current), axis=1)
        backward = np.min(cdist(current, previous), axis=1)
        motion = float(0.5 * (np.median(forward) + np.median(backward)))
        cloud_motion.append(motion)

        temporal_prediction = reference_affine(previous, current, geometry_config).apply(previous)
        temporal = reference_spatial_cost(
            temporal_prediction,
            current,
            geometry_config.spatial_scale_floor_um,
        )
        reference_prediction = reference_affine(reference, current, geometry_config).apply(
            reference
        )
        anchored = reference_spatial_cost(
            reference_prediction,
            current,
            geometry_config.spatial_scale_floor_um,
        )
        cost = temporal + geometry_config.reference_weight * anchored

        extent_distance = cdist(template, scaled[local_frame]) / np.sqrt(len(indices))
        pair_quality = np.sqrt(
            template_quality[:, None] * qualities[local_frame][None, :]
        )
        if not extent_config.use_reliability_gate:
            pair_quality = np.ones_like(extent_distance)
        extent_cost = extent_config.extent_weight * pair_quality * extent_distance
        if extent_config.extent_weight:
            cost = cost + extent_cost

        track_rows, detection_rows = linear_sum_assignment(cost)
        assignment = np.full(count, -1, dtype=np.int64)
        assignment[track_rows] = detection_rows
        assignments.append(assignment)

        rows = np.arange(count)
        temporal_best = np.argmin(temporal, axis=1)
        reference_best = np.argmin(anchored, axis=1)
        extent_best = np.argmin(extent_cost, axis=1)
        selected_temporal = temporal[rows, assignment]
        selected_reference = anchored[rows, assignment]
        selected_extent = extent_cost[rows, assignment]
        selected_total = cost[rows, assignment]
        temporal_selected = temporal_best == assignment
        reference_selected = reference_best == assignment
        extent_selected = extent_best == assignment
        reference_agreement_streak = np.where(
            reference_selected, reference_agreement_streak + 1.0, 0.0
        )
        reference_disagreement_count += ~reference_selected

        row_rank = np.sum(cost < selected_total[:, None], axis=1) / max(count - 1, 1)
        selected_cost_by_column = np.empty(count, dtype=float)
        selected_cost_by_column[assignment] = selected_total
        column_rank = np.sum(
            cost < selected_cost_by_column[None, :], axis=0
        )[assignment] / max(count - 1, 1)
        frame_gap = int(
            sequence.source_frame_indices[local_frame]
            - sequence.source_frame_indices[local_frame - 1]
        )
        diagnostics = np.column_stack(
            [
                selected_total,
                _alternative_margin(cost, assignment, axis=1),
                _alternative_margin(cost, assignment, axis=0),
                selected_temporal,
                selected_reference,
                selected_extent,
                pair_quality[rows, assignment],
                temporal_selected.astype(float),
                reference_selected.astype(float),
                extent_selected.astype(float),
                (temporal_best == reference_best).astype(float),
                row_rank,
                column_rank,
                _selected_softmax_mass(cost, assignment, axis=1),
                _selected_softmax_mass(cost, assignment, axis=0),
                np.linalg.norm(temporal_prediction - reference_prediction, axis=1),
                np.full(count, motion),
                np.full(count, frame_gap, dtype=float),
                reference_agreement_streak / local_frame,
                reference_disagreement_count / local_frame,
            ]
        )
        feature_blocks.append(diagnostics)
        source_blocks.append(
            np.full(count, sequence.source_frame_indices[local_frame], dtype=np.int64)
        )
        track_blocks.append(rows.astype(np.int64))
        detection_blocks.append(assignment.copy())

        previous = current[assignment]
        update = extent_config.extent_update
        template = (1 - update) * template + update * scaled[local_frame][assignment]
        template_quality = (
            (1 - update) * template_quality + update * qualities[local_frame][assignment]
        )

    result = TrackResult(
        method="dual_anchor_extent" if extent_config.extent_weight else "dual_anchor",
        assignments=tuple(assignments),
        runtime_seconds=float(perf_counter() - started),
        cloud_motion_um=np.asarray(cloud_motion, dtype=float),
    )
    events = AssociationDiagnosticEvents(
        features=np.vstack(feature_blocks),
        source_frame_indices=np.concatenate(source_blocks),
        track_rows=np.concatenate(track_blocks),
        selected_detection_rows=np.concatenate(detection_blocks),
    )
    return result, events


def label_diagnostic_events(
    sequence: IdentityBlindSequence,
    result: TrackResult,
    diagnostics: AssociationDiagnosticEvents,
    *,
    recording_id: str,
    stride: int,
) -> LabeledDiagnosticEvents:
    """Join tracker outputs to hidden identities after all associations are fixed."""

    canonical = np.asarray(sequence.truth_indices[0], dtype=np.int64)
    correctness: list[np.ndarray] = []
    for hidden_order, assignment in zip(
        sequence.truth_indices[1:], result.assignments[1:], strict=True
    ):
        identities = np.asarray(hidden_order, dtype=np.int64)[np.asarray(assignment, dtype=int)]
        correctness.append(identities == canonical)
    labels = np.concatenate(correctness)
    if len(labels) != len(diagnostics.features):
        raise AssertionError("truth rows and diagnostic rows became misaligned")
    return LabeledDiagnosticEvents(
        features=diagnostics.features,
        correctness=labels,
        recording_ids=np.full(len(labels), recording_id, dtype=object),
        strides=np.full(len(labels), int(stride), dtype=np.int64),
        source_frame_indices=diagnostics.source_frame_indices,
    )


def load_patch_feature_cache(path: str | Path) -> PatchFeatureSet:
    """Load an auditable feature cache written by the extent benchmark."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        return PatchFeatureSet(
            descriptors=payload["descriptors"],
            reliability=payload["reliability"],
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            extent_indices=tuple(int(value) for value in payload["extent_indices"]),
            source_frame_indices=tuple(int(value) for value in payload["source_frame_indices"]),
            extraction_seconds=float(payload["extraction_seconds"]),
        )


def _selected_frames(frame_count: int, stride: int) -> tuple[int, ...]:
    selected = tuple(range(0, frame_count, stride))
    return selected if selected[-1] == frame_count - 1 else (*selected, frame_count - 1)


def _subset_features(features: PatchFeatureSet, selected: Sequence[int]) -> PatchFeatureSet:
    lookup = {source: index for index, source in enumerate(features.source_frame_indices)}
    rows = np.asarray([lookup[int(source)] for source in selected], dtype=int)
    return PatchFeatureSet(
        descriptors=features.descriptors[rows],
        reliability=features.reliability[rows],
        feature_names=features.feature_names,
        extent_indices=features.extent_indices,
        source_frame_indices=tuple(int(value) for value in selected),
        extraction_seconds=features.extraction_seconds,
    )


def _combine(events: Sequence[LabeledDiagnosticEvents]) -> LabeledDiagnosticEvents:
    if not events:
        raise ValueError("at least one event block is required")
    return LabeledDiagnosticEvents(
        features=np.vstack([event.features for event in events]),
        correctness=np.concatenate([event.correctness for event in events]),
        recording_ids=np.concatenate([event.recording_ids for event in events]),
        strides=np.concatenate([event.strides for event in events]),
        source_frame_indices=np.concatenate([event.source_frame_indices for event in events]),
    )


def _confidence_pipeline(regularization: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(regularization),
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=20260720,
                ),
            ),
        ]
    )


def _equal_stride_weights(strides: np.ndarray) -> np.ndarray:
    values, counts = np.unique(strides, return_counts=True)
    weight_by_stride = {int(value): len(strides) / (len(values) * count) for value, count in zip(values, counts, strict=True)}
    return np.asarray([weight_by_stride[int(value)] for value in strides], dtype=float)


def fit_blocked_confidence_model(
    calibration: LabeledDiagnosticEvents,
    *,
    regularization_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    temporal_blocks: int = 5,
) -> tuple[Pipeline, np.ndarray, dict[str, object]]:
    """Select L2 strength by temporal-block OOF NLL on the calibration animal."""

    if len(np.unique(calibration.recording_ids)) != 1:
        raise ValueError("confidence fitting requires exactly one calibration recording")
    if temporal_blocks < 2:
        raise ValueError("temporal_blocks must be at least two")
    maximum_frame = int(np.max(calibration.source_frame_indices))
    groups = np.minimum(
        temporal_blocks - 1,
        calibration.source_frame_indices * temporal_blocks // (maximum_frame + 1),
    )
    if len(np.unique(groups)) != temporal_blocks:
        raise ValueError("not all requested temporal blocks contain events")
    splitter = GroupKFold(n_splits=temporal_blocks)
    sample_weights = _equal_stride_weights(calibration.strides)
    candidates: list[dict[str, object]] = []
    candidate_predictions: list[np.ndarray] = []
    labels = calibration.correctness.astype(int)
    for regularization in regularization_grid:
        predictions = np.empty(len(labels), dtype=float)
        fold_nll: list[float] = []
        for train, test in splitter.split(calibration.features, labels, groups=groups):
            model = _confidence_pipeline(float(regularization))
            model.fit(
                calibration.features[train],
                labels[train],
                logistic__sample_weight=sample_weights[train],
            )
            predictions[test] = model.predict_proba(calibration.features[test])[:, 1]
            fold_nll.append(
                float(
                    log_loss(
                        labels[test],
                        predictions[test],
                        sample_weight=sample_weights[test],
                        labels=[0, 1],
                    )
                )
            )
        objective = float(
            log_loss(labels, predictions, sample_weight=sample_weights, labels=[0, 1])
        )
        candidates.append(
            {
                "regularization_C": float(regularization),
                "equal_stride_weighted_oof_nll": objective,
                "fold_nll": fold_nll,
            }
        )
        candidate_predictions.append(predictions)
    objectives = np.asarray(
        [candidate["equal_stride_weighted_oof_nll"] for candidate in candidates], dtype=float
    )
    selected_index = int(np.argmin(objectives))
    selected_regularization = float(regularization_grid[selected_index])
    model = _confidence_pipeline(selected_regularization)
    model.fit(
        calibration.features,
        labels,
        logistic__sample_weight=sample_weights,
    )
    selection = {
        "procedure": "five contiguous source-time blocks; each stride has equal total weight",
        "temporal_blocks": temporal_blocks,
        "regularization_grid": [float(value) for value in regularization_grid],
        "candidates": candidates,
        "selected_regularization_C": selected_regularization,
        "selected_by": "minimum equal-stride-weighted out-of-fold Bernoulli NLL",
    }
    return model, candidate_predictions[selected_index], selection


def _risk_at_coverage(
    probabilities: np.ndarray,
    correctness: np.ndarray,
    coverage: float,
) -> dict[str, float | int]:
    """Select the top round(coverage*n) events with a deterministic stable order."""

    if not 0 < coverage <= 1:
        raise ValueError("coverage must lie in (0, 1]")
    probabilities = np.asarray(probabilities, dtype=float)
    correctness = np.asarray(correctness, dtype=bool)
    count = max(1, int(round(coverage * len(probabilities))))
    order = np.argsort(-probabilities, kind="stable")
    chosen = order[:count]
    threshold = float(probabilities[order[count - 1]])
    return {
        "nominal_coverage": float(coverage),
        "selected_events": int(count),
        "total_events": int(len(probabilities)),
        "realized_coverage": float(count / len(probabilities)),
        "selective_accuracy": float(np.mean(correctness[chosen])),
        "selective_risk": float(1 - np.mean(correctness[chosen])),
        "probability_threshold": threshold,
        "events_tied_at_boundary": int(np.count_nonzero(probabilities == threshold)),
    }


def _maximum_coverage_at_accuracy(
    probabilities: np.ndarray,
    correctness: np.ndarray,
    target_accuracy: float,
) -> dict[str, float | int | None]:
    """Retrospective prefix diagnostic; labels make this unsuitable for deployment."""

    order = np.argsort(-np.asarray(probabilities, dtype=float), kind="stable")
    sorted_correct = np.asarray(correctness, dtype=bool)[order]
    prefix_accuracy = np.cumsum(sorted_correct) / np.arange(1, len(order) + 1)
    eligible = np.flatnonzero(prefix_accuracy >= target_accuracy)
    if len(eligible) == 0:
        return {
            "target_accuracy": float(target_accuracy),
            "selected_events": 0,
            "total_events": int(len(order)),
            "coverage": 0.0,
            "accuracy": None,
            "probability_threshold": None,
        }
    count = int(eligible[-1] + 1)
    return {
        "target_accuracy": float(target_accuracy),
        "selected_events": count,
        "total_events": int(len(order)),
        "coverage": float(count / len(order)),
        "accuracy": float(prefix_accuracy[count - 1]),
        "probability_threshold": float(np.asarray(probabilities)[order[count - 1]]),
    }


def _apply_threshold(
    probabilities: np.ndarray,
    correctness: np.ndarray,
    threshold: float | None,
) -> dict[str, float | int | None]:
    if threshold is None:
        return {
            "probability_threshold": None,
            "selected_events": 0,
            "total_events": int(len(probabilities)),
            "coverage": 0.0,
            "accuracy": None,
        }
    selected = np.asarray(probabilities) >= threshold
    return {
        "probability_threshold": float(threshold),
        "selected_events": int(np.count_nonzero(selected)),
        "total_events": int(len(probabilities)),
        "coverage": float(np.mean(selected)),
        "accuracy": float(np.mean(np.asarray(correctness)[selected]))
        if np.any(selected)
        else None,
    }


def _risk_coverage_grid(
    probabilities: np.ndarray,
    correctness: np.ndarray,
) -> dict[str, object]:
    levels = np.linspace(0.01, 1.0, 100)
    probabilities = np.asarray(probabilities, dtype=float)
    correctness = np.asarray(correctness, dtype=bool)
    order = np.argsort(-probabilities, kind="stable")
    sorted_correct = correctness[order]
    cumulative_correct = np.cumsum(sorted_correct)
    points: list[dict[str, float | int]] = []
    for level in levels:
        count = max(1, int(round(float(level) * len(probabilities))))
        accuracy = float(cumulative_correct[count - 1] / count)
        threshold = float(probabilities[order[count - 1]])
        points.append(
            {
                "nominal_coverage": float(level),
                "selected_events": count,
                "total_events": int(len(probabilities)),
                "realized_coverage": float(count / len(probabilities)),
                "selective_accuracy": accuracy,
                "selective_risk": 1.0 - accuracy,
                "probability_threshold": threshold,
                "events_tied_at_boundary": int(np.count_nonzero(probabilities == threshold)),
            }
        )
    full = risk_coverage_curve(probabilities, correctness)
    return {
        "grid": points,
        "aurc": float(full.aurc),
        "events": int(full.total_events),
        "errors": int(full.total_errors),
        "note": "100 fixed coverage levels; exact curve used internally for AURC",
    }


def _calibration_report(probabilities: np.ndarray, correctness: np.ndarray) -> dict[str, object]:
    probabilities = np.asarray(probabilities, dtype=float)
    correctness = np.asarray(correctness, dtype=bool)
    fixed = fixed_bin_ece(probabilities, correctness, n_bins=15)
    adaptive = adaptive_bin_ece(probabilities, correctness, n_bins=15)
    return {
        "events": int(len(probabilities)),
        "empirical_accuracy": float(np.mean(correctness)),
        "mean_probability": float(np.mean(probabilities)),
        "negative_log_likelihood": float(
            negative_log_likelihood(probabilities, correctness)
        ),
        "brier": float(brier_score(probabilities, correctness)),
        "fixed_ece": {
            **{key: value for key, value in asdict(fixed).items() if key != "bins"},
            "bins": [asdict(value) for value in fixed.bins],
        },
        "adaptive_ece": {
            **{key: value for key, value in asdict(adaptive).items() if key != "bins"},
            "bins": [asdict(value) for value in adaptive.bins],
        },
        "risk_coverage": _risk_coverage_grid(probabilities, correctness),
        "selective_accuracy_at_80_percent_coverage": _risk_at_coverage(
            probabilities, correctness, 0.8
        ),
        "retrospective_maximum_coverage_at_99_5_percent_accuracy": _maximum_coverage_at_accuracy(
            probabilities, correctness, 0.995
        ),
    }


def _bootstrap_summary(
    probabilities: np.ndarray,
    correctness: np.ndarray,
    recording_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    statistics = {
        "empirical_accuracy": lambda value, target: float(np.mean(target)),
        "negative_log_likelihood": negative_log_likelihood,
        "brier": brier_score,
        "fixed_ece": lambda value, target: fixed_bin_ece(value, target, n_bins=15).ece,
        "adaptive_ece": lambda value, target: adaptive_bin_ece(value, target, n_bins=15).ece,
        "aurc": lambda value, target: risk_coverage_curve(value, target).aurc,
        "selective_accuracy_at_80_percent_coverage": lambda value, target: float(
            _risk_at_coverage(value, target, 0.8)["selective_accuracy"]
        ),
        "retrospective_maximum_coverage_at_99_5_percent_accuracy": lambda value, target: float(
            _maximum_coverage_at_accuracy(value, target, 0.995)["coverage"]
        ),
    }
    output: dict[str, object] = {}
    for metric_index, (name, statistic) in enumerate(statistics.items()):
        interval = grouped_bootstrap(
            probabilities,
            correctness,
            recording_ids,
            statistic,
            n_resamples=samples,
            confidence_level=0.95,
            seed=seed + metric_index,
            aggregation="macro",
        )
        output[name] = {
            "point_estimate": interval.point_estimate,
            "ci95_low": interval.lower,
            "ci95_high": interval.upper,
            "standard_error": interval.standard_error,
            "recordings": interval.n_groups,
            "bootstrap_samples": interval.n_resamples,
            "aggregation": "mean of complete-recording statistics",
        }
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_artifact(model: Pipeline) -> dict[str, object]:
    scaler = model.named_steps["scale"]
    logistic = model.named_steps["logistic"]
    return {
        "feature_names": list(FEATURE_NAMES),
        "standardization_mean": scaler.mean_.tolist(),
        "standardization_scale": scaler.scale_.tolist(),
        "logistic_coefficients": logistic.coef_[0].tolist(),
        "logistic_intercept": float(logistic.intercept_[0]),
        "positive_class": "selected assignment preserves the initialized neuron identity",
    }


def run_frozen_calibration_benchmark(
    nwb_paths: Sequence[str | Path],
    feature_cache_paths: Sequence[str | Path],
    *,
    output_path: str | Path,
    calibration_recording_id: str,
    strides: Sequence[int] = (1, 5, 10, 20, 40),
    geometry_config: ReferenceTrackerConfig = ReferenceTrackerConfig(reference_weight=6.0),
    extent_config: DualAnchorExtentConfig = DualAnchorExtentConfig(
        extent_weight=8.0,
        extent_update=0.25,
        feature_indices=tuple(range(15)),
    ),
    seed: int = 20260720,
    bootstrap_samples: int = 10_000,
) -> dict[str, object]:
    """Run the calibration animal first, freeze the model, then open held-out outcomes."""

    if len(nwb_paths) != len(feature_cache_paths):
        raise ValueError("NWB and feature-cache paths must pair one-to-one")
    loaded: list[tuple[object, PatchFeatureSet, Path]] = []
    for nwb_path, cache_path in zip(nwb_paths, feature_cache_paths, strict=True):
        recording = load_wormid_recording(nwb_path)
        cache = load_patch_feature_cache(cache_path)
        if cache.source_frame_indices != tuple(range(recording.frames)):
            raise ValueError(f"feature cache is not complete for {recording.recording_id}")
        loaded.append((recording, cache, Path(cache_path)))
    calibration_matches = [item for item in loaded if item[0].recording_id == calibration_recording_id]
    if len(calibration_matches) != 1:
        raise ValueError("calibration_recording_id must identify exactly one input")
    calibration_recording, calibration_cache, calibration_cache_path = calibration_matches[0]
    held_out_loaded = [item for item in loaded if item[0].recording_id != calibration_recording_id]
    if len(held_out_loaded) < 2:
        raise ValueError("confirmatory evaluation requires at least two held-out recordings")

    extent_indices = np.asarray(calibration_cache.extent_indices, dtype=int)
    if tuple(extent_indices) != tuple(extent_config.feature_indices or ()): 
        raise ValueError("extent configuration must select the strict cached extent indices")
    scaler = fit_robust_feature_scaler(
        calibration_cache.descriptors[..., extent_indices],
        reliability=calibration_cache.reliability,
    )

    started = perf_counter()

    def collect(
        recording,
        cache: PatchFeatureSet,
        *,
        path_index: int,
    ) -> list[LabeledDiagnosticEvents]:
        event_blocks: list[LabeledDiagnosticEvents] = []
        strict_cache = PatchFeatureSet(
            descriptors=cache.descriptors[..., extent_indices],
            reliability=cache.reliability,
            feature_names=tuple(cache.feature_names[index] for index in extent_indices),
            extent_indices=tuple(range(len(extent_indices))),
            source_frame_indices=cache.source_frame_indices,
            extraction_seconds=cache.extraction_seconds,
        )
        for stride in strides:
            selected = _selected_frames(recording.frames, int(stride))
            sequence = identity_blind_sequence(
                recording,
                frame_indices=selected,
                seed=seed + 1000 * path_index + int(stride),
            )
            selected_features = _subset_features(strict_cache, selected)
            anonymous = pair_patch_features_to_anonymous_detections(sequence, selected_features)
            result, diagnostics = track_dual_anchor_extent_with_diagnostics(
                sequence,
                anonymous,
                scaler,
                geometry_config=geometry_config,
                extent_config=extent_config,
            )
            event_blocks.append(
                label_diagnostic_events(
                    sequence,
                    result,
                    diagnostics,
                    recording_id=recording.recording_id,
                    stride=int(stride),
                )
            )
        return event_blocks

    # Calibration animal is evaluated and the model is completely frozen before
    # held-out labels are collected or passed to any model-selection function.
    calibration_path_index = next(
        index for index, item in enumerate(loaded) if item[0].recording_id == calibration_recording_id
    )
    calibration = _combine(
        collect(
            calibration_recording,
            calibration_cache,
            path_index=calibration_path_index,
        )
    )
    model, oof_probabilities, selection = fit_blocked_confidence_model(calibration)
    calibration_report = _calibration_report(oof_probabilities, calibration.correctness)
    frozen_coverage = calibration_report[
        "retrospective_maximum_coverage_at_99_5_percent_accuracy"
    ]
    frozen_threshold = frozen_coverage["probability_threshold"]

    held_out_blocks: list[LabeledDiagnosticEvents] = []
    source_rows: list[dict[str, object]] = []
    for path_index, (recording, cache, cache_path) in enumerate(loaded):
        source_rows.append(
            {
                "recording_id": recording.recording_id,
                "split": "calibration"
                if recording.recording_id == calibration_recording_id
                else "held_out",
                "nwb_path": recording.path,
                "feature_cache_path": str(cache_path.resolve()),
                "feature_cache_sha256": _sha256(cache_path),
                "frames": recording.frames,
                "neurons": recording.neurons,
            }
        )
        if recording.recording_id != calibration_recording_id:
            held_out_blocks.extend(collect(recording, cache, path_index=path_index))
    held_out = _combine(held_out_blocks)
    held_out_probabilities = model.predict_proba(held_out.features)[:, 1]

    per_recording: list[dict[str, object]] = []
    for recording_id in sorted(set(held_out.recording_ids)):
        selected = held_out.recording_ids == recording_id
        report = _calibration_report(
            held_out_probabilities[selected], held_out.correctness[selected]
        )
        report["recording_id"] = str(recording_id)
        report["frozen_calibration_threshold_at_99_5_percent_accuracy"] = _apply_threshold(
            held_out_probabilities[selected], held_out.correctness[selected], frozen_threshold
        )
        per_recording.append(report)

    held_out_report = _calibration_report(held_out_probabilities, held_out.correctness)
    held_out_report["frozen_calibration_threshold_at_99_5_percent_accuracy"] = _apply_threshold(
        held_out_probabilities, held_out.correctness, frozen_threshold
    )
    held_out_report["complete_recording_bootstrap"] = _bootstrap_summary(
        held_out_probabilities,
        held_out.correctness,
        held_out.recording_ids,
        samples=bootstrap_samples,
        seed=seed,
    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "confidence feature map and model selection use only the calibration animal; "
            "the fitted model is frozen for four held-out animals"
        ),
        "calibration_recording_id": calibration_recording_id,
        "held_out_recording_ids": sorted(str(value) for value in set(held_out.recording_ids)),
        "strides": [int(value) for value in strides],
        "seed_rule": f"{seed} + 1000*input_path_index + stride",
        "association_label_boundary": {
            "labels_used_for_association": False,
            "stored_neural_traces_read": False,
            "diagnostics_created_before_truth_join": True,
            "confidence_response": "whether the selected row preserves initialized identity",
        },
        "geometry_config": asdict(geometry_config),
        "extent_config": asdict(extent_config),
        "feature_scaler": {
            "fit_recordings": [calibration_recording_id],
            "method": "per-feature median/IQR over reliable strict-extent patches",
            "center": scaler.center.tolist(),
            "scale": scaler.scale.tolist(),
        },
        "confidence_model_selection": selection,
        "confidence_model": _model_artifact(model),
        "sources": source_rows,
        "calibration_out_of_fold": {
            "boundary": (
                "five source-time-block OOF predictions on the calibration animal; "
                "not independent-animal evidence"
            ),
            **calibration_report,
        },
        "frozen_99_5_percent_policy": {
            "selection_source": "calibration-animal blocked out-of-fold risk-coverage curve",
            "probability_threshold": frozen_threshold,
            "calibration_oof_result": frozen_coverage,
            "note": (
                "Threshold is frozen before held-out scoring; the held-out retrospective maximum "
                "coverage is diagnostic only and is not used to alter this policy."
            ),
        },
        "held_out_pooled": held_out_report,
        "held_out_per_recording": per_recording,
        "preregistered_target": {
            "criterion": "at 80% coverage, precision >= 99.5% and ECE < 0.05",
            "coverage_result": held_out_report[
                "selective_accuracy_at_80_percent_coverage"
            ],
            "fixed_ece": held_out_report["fixed_ece"]["ece"],
            "adaptive_ece": held_out_report["adaptive_ece"]["ece"],
            "passed": bool(
                held_out_report["selective_accuracy_at_80_percent_coverage"][
                    "selective_accuracy"
                ]
                >= 0.995
                and held_out_report["fixed_ece"]["ece"] < 0.05
            ),
            "assessment": (
                "Both requirements must pass on the frozen four-animal evaluation; no post-hoc "
                "threshold or refit is substituted."
            ),
        },
        "runtime_seconds": float(perf_counter() - started),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nwb_paths", nargs="+")
    parser.add_argument("--feature-caches", nargs="+", required=True)
    parser.add_argument("--calibration-recording-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_frozen_calibration_benchmark(
        args.nwb_paths,
        args.feature_caches,
        output_path=args.output,
        calibration_recording_id=args.calibration_recording_id,
        strides=args.strides,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(payload["preregistered_target"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
