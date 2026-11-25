"""Identity-blind reference-anchored tracking benchmarks for WormID-Bench.

The tracker receives only an anonymous point cloud at each selected time point.
The first frame seeds a persistent cohort, as in a manual initialization
workflow.  Stable NWB ROI order and semantic neuron labels are deliberately kept
outside every association routine and are consumed only by :func:`score_tracks`.

Four deliberately interpretable methods are provided:

``previous_hungarian``
    Euclidean Hungarian matching from the last assigned positions.
``pairwise_affine``
    The same association after identity-blind robust affine cloud transport.
``first_affine``
    Direct robust affine registration of the first-frame reference to every
    later frame.  It cannot accumulate temporal identity errors.
``dual_anchor``
    Pairwise affine motion fused with a direct, immutable first-frame affine
    reference.  An optional distance-profile term is rigid/scale invariant.

The module is intentionally independent of the NWB label strings.  Tests use
adversarial row permutations to make accidental reliance on ROI order visible.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .tracker import estimate_shared_affine
from .wormid import IdentityBlindSequence, identity_blind_sequence, load_wormid_recording


METHODS = (
    "previous_hungarian",
    "pairwise_affine",
    "first_affine",
    "dual_anchor",
)


@dataclass(frozen=True)
class ReferenceTrackerConfig:
    """Geometry-only settings, frozen before held-out animal evaluation."""

    reference_weight: float = 6.0
    descriptor_weight: float = 0.0
    descriptor_quantiles: int = 32
    affine_iterations: int = 6
    affine_ridge: float = 0.2
    affine_huber_delta: float = 2.5
    affine_min_scale: float = 0.70
    affine_max_scale: float = 1.40
    spatial_scale_floor_um: float = 0.10

    def __post_init__(self) -> None:
        if self.reference_weight < 0 or self.descriptor_weight < 0:
            raise ValueError("reference/descriptor weights cannot be negative")
        if self.descriptor_quantiles < 4:
            raise ValueError("descriptor_quantiles must be at least four")
        if self.affine_iterations < 1 or self.affine_ridge < 0:
            raise ValueError("affine iterations/ridge are invalid")
        if self.affine_huber_delta <= 0 or self.spatial_scale_floor_um <= 0:
            raise ValueError("positive robust and spatial scales are required")
        if not 0 < self.affine_min_scale <= self.affine_max_scale:
            raise ValueError("affine scale bounds are invalid")


@dataclass(frozen=True)
class TrackResult:
    """Persistent track-row to anonymous detection-row assignments."""

    method: str
    assignments: tuple[np.ndarray, ...]
    runtime_seconds: float
    cloud_motion_um: np.ndarray


@dataclass(frozen=True)
class TrackMetrics:
    """Seeded-cohort identity metrics; exhaustive assignment makes accuracy IDF1."""

    method: str
    frames: int
    neurons: int
    correct: int
    total: int
    identity_accuracy: float
    seeded_idf1: float
    identity_switches: int
    severe_correct: int
    severe_total: int
    severe_identity_accuracy: float
    severe_transition_threshold_um: float
    severe_frames: int
    severe_selection_mode: str
    severe_ties_at_boundary: int
    runtime_seconds: float


def _validate_positions(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("a point cloud must have shape (n>=2, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("point coordinates must be finite")
    return points


def distance_profile_descriptor(points: np.ndarray, *, quantiles: int = 32) -> np.ndarray:
    """Return a permutation-equivariant, rigid/scale-invariant shape signature.

    Each row contains quantiles of distances from one point to every other point.
    Dividing by the cloud-wide median pair distance removes a common scale.  The
    operation never uses correspondence, detection order, or semantic labels.
    """

    points = _validate_positions(points)
    if quantiles < 4:
        raise ValueError("quantiles must be at least four")
    distances = cdist(points, points)
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    scale = max(scale, np.finfo(float).eps)
    ordered = np.sort(distances, axis=1)[:, 1:]
    levels = np.linspace(0.0, 1.0, quantiles)
    # np.quantile returns (quantiles, points); transpose restores one row/point.
    descriptor = np.quantile(ordered / scale, levels, axis=1).T
    return np.asarray(descriptor, dtype=float)


def descriptor_cost(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Robustly standardize profile discrepancies without known pairs."""

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.ndim != 2 or current.ndim != 2 or reference.shape[1] != current.shape[1]:
        raise ValueError("descriptor matrices must have the same feature count")
    pooled = np.vstack([reference, current])
    feature_scale = np.median(np.abs(pooled - np.median(pooled, axis=0)), axis=0)
    feature_scale = np.maximum(1.4826 * feature_scale, 0.03)
    return cdist(reference / feature_scale, current / feature_scale) / np.sqrt(
        reference.shape[1]
    )


def _affine(source: np.ndarray, target: np.ndarray, config: ReferenceTrackerConfig):
    return estimate_shared_affine(
        source,
        target,
        iterations=config.affine_iterations,
        ridge=config.affine_ridge,
        huber_delta=config.affine_huber_delta,
        min_scale=config.affine_min_scale,
        max_scale=config.affine_max_scale,
    )


def _spatial_cost(predicted: np.ndarray, target: np.ndarray, floor_um: float) -> np.ndarray:
    distances = cdist(predicted, target)
    rows, cols = linear_sum_assignment(distances)
    residual = distances[rows, cols]
    # A robust data-dependent scale makes descriptor weighting portable across
    # animals and temporal gaps, while using coordinates only.
    scale = max(float(np.median(residual)), floor_um)
    return distances / scale


def track_sequence(
    sequence: IdentityBlindSequence,
    method: str,
    *,
    config: ReferenceTrackerConfig = ReferenceTrackerConfig(),
) -> TrackResult:
    """Track an initialized cohort using anonymous detection coordinates only."""

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if not sequence.frames:
        raise ValueError("sequence is empty")
    reference = _validate_positions(sequence.frames[0].positions)
    count = len(reference)
    if any(len(frame.positions) != count for frame in sequence.frames):
        raise ValueError("these exhaustive WormID methods require a fixed cohort size")

    started = time.perf_counter()
    assignments: list[np.ndarray] = [np.arange(count, dtype=np.int64)]
    previous = reference.copy()
    reference_descriptor = distance_profile_descriptor(
        reference, quantiles=config.descriptor_quantiles
    )
    cloud_motion: list[float] = [0.0]

    for frame in sequence.frames[1:]:
        current = _validate_positions(frame.positions)
        # This symmetric cloud score is label-free and is retained only to define
        # a severe-motion evaluation stratum.
        forward = np.min(cdist(previous, current), axis=1)
        backward = np.min(cdist(current, previous), axis=1)
        cloud_motion.append(float(0.5 * (np.median(forward) + np.median(backward))))

        if method == "previous_hungarian":
            cost = cdist(previous, current)
        elif method == "pairwise_affine":
            predicted = _affine(previous, current, config).apply(previous)
            cost = cdist(predicted, current)
        elif method == "first_affine":
            predicted = _affine(reference, current, config).apply(reference)
            cost = cdist(predicted, current)
        else:
            temporal_prediction = _affine(previous, current, config).apply(previous)
            temporal = _spatial_cost(
                temporal_prediction, current, config.spatial_scale_floor_um
            )
            reference_prediction = _affine(reference, current, config).apply(reference)
            anchored = _spatial_cost(
                reference_prediction, current, config.spatial_scale_floor_um
            )
            cost = temporal + config.reference_weight * anchored
            if config.descriptor_weight:
                current_descriptor = distance_profile_descriptor(
                    current, quantiles=config.descriptor_quantiles
                )
                shape = descriptor_cost(reference_descriptor, current_descriptor)
                cost += config.descriptor_weight * shape

        track_rows, detection_rows = linear_sum_assignment(cost)
        assignment = np.full(count, -1, dtype=np.int64)
        assignment[track_rows] = detection_rows
        assignments.append(assignment)
        previous = current[assignment]

    return TrackResult(
        method=method,
        assignments=tuple(assignments),
        runtime_seconds=float(time.perf_counter() - started),
        cloud_motion_um=np.asarray(cloud_motion, dtype=float),
    )


def score_tracks(
    sequence: IdentityBlindSequence,
    result: TrackResult,
    *,
    severe_quantile: float = 0.75,
    severe_threshold_um: float | None = None,
) -> TrackMetrics:
    """Evaluate hidden identities, optionally with a pre-frozen severe threshold."""

    if len(result.assignments) != len(sequence.frames):
        raise ValueError("result length does not match sequence")
    if not 0 < severe_quantile < 1:
        raise ValueError("severe_quantile must lie strictly between zero and one")
    canonical = np.asarray(sequence.truth_indices[0], dtype=int)
    identity_by_track: list[np.ndarray] = []
    correct_by_frame: list[np.ndarray] = []
    for hidden_order, assignment in zip(sequence.truth_indices, result.assignments):
        assignment = np.asarray(assignment, dtype=int)
        if assignment.shape != canonical.shape or np.any(assignment < 0):
            raise ValueError("WormID benchmark assignments must be exhaustive")
        identities = np.asarray(hidden_order, dtype=int)[assignment]
        identity_by_track.append(identities)
        correct_by_frame.append(identities == canonical)
    correct_matrix = np.stack(correct_by_frame)
    identities = np.stack(identity_by_track)

    transition_motion = result.cloud_motion_um[1:]
    if severe_threshold_um is None:
        # Select exactly the top fraction. Voxel quantization creates many ties,
        # so thresholding at q75 can accidentally mark most frames "severe".
        # lexsort breaks boundary ties by earlier source-frame index.
        severe_count = max(1, int(np.ceil((1.0 - severe_quantile) * len(transition_motion))))
        source_indices = np.asarray(sequence.source_frame_indices[1:], dtype=int)
        ranking = np.lexsort((source_indices, -transition_motion))
        selected = ranking[:severe_count]
        severe = np.zeros(len(result.cloud_motion_um), dtype=bool)
        severe[selected + 1] = True
        threshold = float(np.min(transition_motion[selected]))
        ties_at_boundary = int(np.count_nonzero(transition_motion == threshold))
        selection_mode = "exact_top_fraction_motion_then_earlier_source_frame_tiebreak"
    else:
        threshold = float(severe_threshold_um)
        severe = result.cloud_motion_um >= threshold
        severe[0] = False
        ties_at_boundary = int(np.count_nonzero(transition_motion == threshold))
        selection_mode = "frozen_numeric_threshold_greater_or_equal"
    severe_correct = int(correct_matrix[severe].sum())
    severe_total = int(np.count_nonzero(severe) * len(canonical))
    switches = int(np.count_nonzero(np.diff(identities, axis=0)))
    correct = int(correct_matrix.sum())
    total = int(correct_matrix.size)
    accuracy = float(correct / total)
    return TrackMetrics(
        method=result.method,
        frames=len(sequence.frames),
        neurons=len(canonical),
        correct=correct,
        total=total,
        identity_accuracy=accuracy,
        seeded_idf1=accuracy,
        identity_switches=switches,
        severe_correct=severe_correct,
        severe_total=severe_total,
        severe_identity_accuracy=float(severe_correct / severe_total)
        if severe_total
        else float("nan"),
        severe_transition_threshold_um=threshold,
        severe_frames=int(np.count_nonzero(severe)),
        severe_selection_mode=selection_mode,
        severe_ties_at_boundary=ties_at_boundary,
        runtime_seconds=result.runtime_seconds,
    )


def bootstrap_recording_difference(
    records: Sequence[dict[str, float]],
    *,
    improved: str,
    baseline: str,
    field: str = "identity_accuracy",
    samples: int = 10_000,
    seed: int = 20260720,
) -> dict[str, float | int]:
    """Complete-recording paired bootstrap for a method difference."""

    by_recording: dict[str, dict[str, float]] = {}
    for row in records:
        by_recording.setdefault(str(row["recording_id"]), {})[str(row["method"])] = float(
            row[field]
        )
    differences = np.asarray(
        [methods[improved] - methods[baseline] for methods in by_recording.values()],
        dtype=float,
    )
    if not len(differences):
        raise ValueError("no complete paired recordings")
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, len(differences)), replace=True).mean(axis=1)
    return {
        "recordings": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": int(samples),
    }


def _indices(frame_count: int, stride: int) -> tuple[int, ...]:
    if stride < 1:
        raise ValueError("stride must be positive")
    selected = tuple(range(0, frame_count, stride))
    return selected if selected[-1] == frame_count - 1 else (*selected, frame_count - 1)


def run_benchmark(
    paths: Iterable[str | Path],
    *,
    strides: Sequence[int],
    config: ReferenceTrackerConfig,
    output_path: str | Path,
    seed: int = 20260720,
    tuning_recording_id: str | None = None,
) -> dict[str, object]:
    """Run frozen settings on recordings and atomically write JSON evidence."""

    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for path_index, path in enumerate(paths):
        recording = load_wormid_recording(path)
        sources.append(
            {
                "recording_id": recording.recording_id,
                "split": "tuning"
                if recording.recording_id == tuning_recording_id
                else "held_out",
                "path": recording.path,
                "frames": recording.frames,
                "neurons": recording.neurons,
            }
        )
        for stride in strides:
            selected = _indices(recording.frames, int(stride))
            sequence = identity_blind_sequence(
                recording,
                frame_indices=selected,
                seed=seed + 1000 * path_index + int(stride),
            )
            for method in METHODS:
                result = track_sequence(sequence, method, config=config)
                metrics = score_tracks(sequence, result)
                row = asdict(metrics)
                row.update(
                    {
                        "recording_id": recording.recording_id,
                        "split": "tuning"
                        if recording.recording_id == tuning_recording_id
                        else "held_out",
                        "stride": int(stride),
                        "first_source_frame": selected[0],
                        "last_source_frame": selected[-1],
                        "labels_used_for_association": False,
                    }
                )
                rows.append(row)

    payload: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "tuning_recording_id": tuning_recording_id,
        "config": asdict(config),
        "methods": list(METHODS),
        "sources": sources,
        "results": rows,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="WormID EY NWB recordings")
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument("--reference-weight", type=float, default=6.0)
    parser.add_argument("--descriptor-weight", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--tuning-recording-id")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ReferenceTrackerConfig(
        reference_weight=args.reference_weight,
        descriptor_weight=args.descriptor_weight,
    )
    payload = run_benchmark(
        args.paths,
        strides=args.strides,
        config=config,
        output_path=args.output,
        seed=args.seed,
        tuning_recording_id=args.tuning_recording_id,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
