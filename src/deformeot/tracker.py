"""Deformation-aware probabilistic tracking of persistent 3-D point targets.

This module is the dependency-light point/ellipsoid core of DeformEOT.  It is
intended for recordings in which a cohort of neurons is initialized in the first
frame and biological births/deaths do not occur.  Temporary invisibility is
represented explicitly; detections that do not belong to the initialized cohort
are treated as clutter.

The implementation makes four deliberately limited modelling choices:

* a robust affine map transports the complete neural configuration between frames;
* each neuron has a Gaussian local-motion state and Bernoulli visibility state;
* an optional log-Euclidean SPD extent discrepancy contributes to association;
* exact K-best linear assignments are retained in a beam and marginalized after a
  configurable fixed lag.

The resulting association probabilities are beam-normalized approximate model
posteriors (the affine transform is a plug-in estimate).  They are *not* claimed
to be calibrated on a biological data set until evaluated or temperature-scaled
on a held-out recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import math
from typing import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


_EPS = 1e-12


def _as_positions(values: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("positions must have shape (n_detections, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must be finite")
    return array


def _nearest_spd(matrix: np.ndarray, floor: float = 1e-7) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def _validate_extents(extents: np.ndarray | None, size: int) -> np.ndarray | None:
    if extents is None:
        return None
    array = np.asarray(extents, dtype=float)
    if array.shape != (size, 3, 3):
        raise ValueError("extents must have shape (n_detections, 3, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("extents must be finite")
    checked = np.empty_like(array)
    for index, extent in enumerate(array):
        values = np.linalg.eigvalsh(0.5 * (extent + extent.T))
        if values[0] <= 0:
            raise ValueError(f"extent {index} is not positive definite")
        checked[index] = _nearest_spd(extent)
    return checked


@dataclass(frozen=True)
class DetectionFrame:
    """Detections for one volume.

    Positions can use any consistent physical coordinate system.  ASCENT arrays
    can therefore be passed as ``(z, y, x)`` after multiplying by voxel spacing.
    Extents are optional 3-by-3 SPD scatter/covariance matrices in the same units.
    """

    positions: np.ndarray
    extents: np.ndarray | None = None
    confidences: np.ndarray | None = None
    object_ids: Sequence[Hashable] | None = None
    time: int | float | None = None

    def __post_init__(self) -> None:
        positions = _as_positions(self.positions)
        extents = _validate_extents(self.extents, len(positions))
        if self.confidences is None:
            confidences = np.ones(len(positions), dtype=float)
        else:
            confidences = np.asarray(self.confidences, dtype=float)
            if confidences.shape != (len(positions),):
                raise ValueError("confidences must have shape (n_detections,)")
            if not np.all(np.isfinite(confidences)) or np.any(
                (confidences <= 0) | (confidences > 1)
            ):
                raise ValueError("confidences must lie in (0, 1]")
        if self.object_ids is None:
            object_ids: tuple[Hashable, ...] | None = None
        else:
            object_ids = tuple(self.object_ids)
            if len(object_ids) != len(positions):
                raise ValueError("object_ids must have one entry per detection")
            if len(set(object_ids)) != len(object_ids):
                raise ValueError("object_ids must be unique within a frame")
        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(self, "extents", None if extents is None else extents.copy())
        object.__setattr__(self, "confidences", confidences.copy())
        object.__setattr__(self, "object_ids", object_ids)


@dataclass(frozen=True)
class TrackerConfig:
    """Numerical and probabilistic settings for :class:`DeformEOTTracker`."""

    max_hypotheses: int = 12
    assignments_per_hypothesis: int = 6
    fixed_lag: int = 3
    measurement_sigma: float = 0.6
    process_sigma: float = 0.35
    initial_position_sigma: float = 0.3
    velocity_decay: float = 0.65
    velocity_gain: float = 0.45
    detection_probability: float = 0.94
    visible_stay_probability: float = 0.96
    reappearance_probability: float = 0.45
    clutter_density: float = 1e-3
    gate_mahalanobis_sq: float = 25.0
    extent_weight: float = 1.0
    extent_log_sigma: float = 0.7
    extent_update_gain: float = 0.35
    extent_process_sigma: float = 0.03
    affine_iterations: int = 6
    affine_ridge: float = 0.2
    affine_huber_delta: float = 2.5
    affine_min_scale: float = 0.55
    affine_max_scale: float = 1.8
    association_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.max_hypotheses < 1 or self.assignments_per_hypothesis < 1:
            raise ValueError("hypothesis counts must be positive")
        if self.fixed_lag < 0:
            raise ValueError("fixed_lag must be nonnegative")
        positive = {
            "measurement_sigma": self.measurement_sigma,
            "process_sigma": self.process_sigma,
            "initial_position_sigma": self.initial_position_sigma,
            "clutter_density": self.clutter_density,
            "gate_mahalanobis_sq": self.gate_mahalanobis_sq,
            "extent_log_sigma": self.extent_log_sigma,
            "affine_huber_delta": self.affine_huber_delta,
            "affine_min_scale": self.affine_min_scale,
            "affine_max_scale": self.affine_max_scale,
            "association_temperature": self.association_temperature,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        probabilities = {
            "velocity_decay": self.velocity_decay,
            "velocity_gain": self.velocity_gain,
            "detection_probability": self.detection_probability,
            "visible_stay_probability": self.visible_stay_probability,
            "reappearance_probability": self.reappearance_probability,
            "extent_update_gain": self.extent_update_gain,
        }
        for name, value in probabilities.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.extent_weight < 0 or self.extent_process_sigma < 0:
            raise ValueError("extent noise and weight cannot be negative")
        if self.affine_iterations < 1 or self.affine_ridge < 0:
            raise ValueError("affine_iterations must be positive and ridge nonnegative")
        if self.affine_min_scale > self.affine_max_scale:
            raise ValueError("affine_min_scale cannot exceed affine_max_scale")


@dataclass(frozen=True)
class AffineTransform:
    """Shared affine transport using ``points @ matrix.T + offset``."""

    matrix: np.ndarray
    offset: np.ndarray
    matched_pairs: int
    robust_rmse: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.matrix.T + self.offset


def _fit_weighted_affine(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an affine row-vector map, regularized toward the identity."""

    if len(source) < 4:
        displacement = target - source
        offset = np.average(displacement, axis=0, weights=weights)
        return np.eye(3), offset

    design = np.column_stack([source, np.ones(len(source))])
    root_weight = np.sqrt(np.maximum(weights, _EPS))[:, None]
    lhs = design * root_weight
    rhs = target * root_weight
    gram = lhs.T @ lhs
    cross = lhs.T @ rhs
    if ridge:
        gram[:3, :3] += ridge * np.eye(3)
        cross[:3, :] += ridge * np.eye(3)
    coefficients = np.linalg.solve(gram + 1e-9 * np.eye(4), cross)
    return coefficients[:3, :].T, coefficients[3, :]


def _bounded_affine(
    matrix: np.ndarray,
    min_scale: float,
    max_scale: float,
) -> np.ndarray:
    left, values, right = np.linalg.svd(matrix)
    # A reflection is not a plausible one-frame physical deformation.
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    return left @ np.diag(np.clip(values, min_scale, max_scale)) @ right


def estimate_shared_affine(
    source: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 6,
    ridge: float = 0.2,
    huber_delta: float = 2.5,
    min_scale: float = 0.55,
    max_scale: float = 1.8,
) -> AffineTransform:
    """Estimate a robust shared deformation without known correspondences.

    Alternating Hungarian correspondence and Huber-weighted affine regression is
    used.  Singular values are bounded to prevent an unlucky clutter match from
    collapsing or exploding the configuration.  With fewer than four pairs the
    estimator intentionally falls back to translation.
    """

    source = _as_positions(source)
    target = _as_positions(target)
    if len(source) == 0 or len(target) == 0:
        return AffineTransform(np.eye(3), np.zeros(3), 0, float("nan"))

    matrix = np.eye(3)
    offset = np.median(target, axis=0) - np.median(source, axis=0)
    row_ind = np.empty(0, dtype=int)
    col_ind = np.empty(0, dtype=int)
    weights = np.empty(0, dtype=float)
    residual_norm = np.empty(0, dtype=float)

    for _ in range(iterations):
        warped = source @ matrix.T + offset
        row_ind, col_ind = linear_sum_assignment(cdist(warped, target))
        paired_source = source[row_ind]
        paired_target = target[col_ind]

        residual = paired_target - (paired_source @ matrix.T + offset)
        residual_norm = np.linalg.norm(residual, axis=1)
        scale = 1.4826 * np.median(np.abs(residual_norm - np.median(residual_norm)))
        scale = max(float(scale), 1e-3)
        cutoff = huber_delta * scale
        weights = np.minimum(1.0, cutoff / np.maximum(residual_norm, _EPS))
        next_matrix, next_offset = _fit_weighted_affine(
            paired_source, paired_target, weights, ridge
        )
        matrix = _bounded_affine(next_matrix, min_scale, max_scale)
        # Refit translation after bounding the linear component.
        offset = np.average(
            paired_target - paired_source @ matrix.T, axis=0, weights=weights
        )

    residual = target[col_ind] - (source[row_ind] @ matrix.T + offset)
    residual_norm = np.linalg.norm(residual, axis=1)
    robust_rmse = float(
        np.sqrt(np.average(residual_norm**2, weights=np.maximum(weights, _EPS)))
    )
    return AffineTransform(matrix, offset, int(len(row_ind)), robust_rmse)


@dataclass
class _TrackState:
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    visibility: float
    missed_frames: int
    extent: np.ndarray | None

    def copy(self) -> "_TrackState":
        return _TrackState(
            self.position.copy(),
            self.velocity.copy(),
            self.covariance.copy(),
            float(self.visibility),
            int(self.missed_frames),
            None if self.extent is None else self.extent.copy(),
        )


@dataclass
class _Prediction:
    position: np.ndarray
    global_position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    visibility: float
    extent: np.ndarray | None


@dataclass
class _Hypothesis:
    states: list[_TrackState]
    log_weight: float
    history: list[tuple[int, ...]]
    transforms: list[AffineTransform]


@dataclass
class FramePosterior:
    """Per-frame tracker output.

    The last column of each association matrix is the missed-detection event.
    ``association_probabilities`` uses fixed-lag evidence, while
    ``filter_association_probabilities`` uses only data up to this frame.
    """

    frame_index: int
    time: int | float
    track_ids: tuple[Hashable, ...]
    object_ids: tuple[Hashable, ...]
    positions: np.ndarray
    filter_association_probabilities: np.ndarray
    association_probabilities: np.ndarray
    association_entropy: np.ndarray
    map_assignment: np.ndarray
    posterior_positions: np.ndarray
    posterior_covariances: np.ndarray
    posterior_extents: tuple[np.ndarray | None, ...]
    visibility_probability: np.ndarray
    shared_transform: AffineTransform
    hypothesis_count: int

    @property
    def association_confidence(self) -> np.ndarray:
        """Probability of each track's fixed-lag MAP association event."""

        return np.max(self.association_probabilities, axis=1)


@dataclass
class TrackingResult:
    """A sequence of fixed-lag posteriors and convenience export methods."""

    frames: list[FramePosterior]
    track_ids: tuple[Hashable, ...]
    config: TrackerConfig
    calibration_status: str = field(
        default=(
            "beam-normalized approximate model posterior; not empirically calibrated"
        ),
        init=False,
    )

    def to_dataframe(self, *, include_misses: bool = False) -> pd.DataFrame:
        """Return ASCENT-compatible tracks plus uncertainty columns.

        Coordinate columns retain the caller's ordering but are named ``z,y,x``
        for interoperability.  Matched rows use observed positions; optional miss
        rows use the state estimate and an empty ``ObjectID``.
        """

        rows: list[dict[str, object]] = []
        for frame in self.frames:
            for track_index, track_id in enumerate(self.track_ids):
                detection_index = int(frame.map_assignment[track_index])
                if detection_index < 0 and not include_misses:
                    continue
                if detection_index >= 0:
                    position = frame.positions[detection_index]
                    object_id: object = frame.object_ids[detection_index]
                    event_index = detection_index
                else:
                    position = frame.posterior_positions[track_index]
                    object_id = ""
                    event_index = len(frame.object_ids)
                rows.append(
                    {
                        "TrackID": track_id,
                        "ObjectID": object_id,
                        "t": frame.time,
                        "z": float(position[0]),
                        "y": float(position[1]),
                        "x": float(position[2]),
                        "association_probability": float(
                            frame.association_probabilities[track_index, event_index]
                        ),
                        "association_entropy": float(frame.association_entropy[track_index]),
                        "visibility_probability": float(
                            frame.visibility_probability[track_index]
                        ),
                        "is_observed": detection_index >= 0,
                    }
                )
        return pd.DataFrame.from_records(rows)


def _log_spd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_nearest_spd(matrix))
    return (vectors * np.log(np.maximum(values, 1e-7))) @ vectors.T


def _solve_constrained_assignment(
    cost: np.ndarray,
    prefix: tuple[int, ...],
    forbidden: frozenset[tuple[int, int]],
) -> tuple[tuple[int, ...], float] | None:
    constrained = np.array(cost, copy=True)
    for row, column in forbidden:
        constrained[row, column] = np.inf
    if len(set(prefix)) != len(prefix):
        return None
    for row, column in enumerate(prefix):
        constrained[row, :] = np.inf
        constrained[row, column] = cost[row, column]
        for other_row in range(len(prefix), cost.shape[0]):
            constrained[other_row, column] = np.inf
    try:
        rows, columns = linear_sum_assignment(constrained)
    except ValueError:
        return None
    if len(rows) != cost.shape[0] or not np.all(np.isfinite(constrained[rows, columns])):
        return None
    assignment = np.empty(cost.shape[0], dtype=int)
    assignment[rows] = columns
    return tuple(int(value) for value in assignment), float(constrained[rows, columns].sum())


def k_best_assignments(cost: np.ndarray, k: int) -> list[tuple[np.ndarray, float]]:
    """Return the exact K lowest-cost one-to-one row assignments.

    This is Murty-style partitioning of the assignment solution space.  The cost
    matrix may be rectangular but must have at least as many columns as rows.
    """

    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] > cost.shape[1]:
        raise ValueError("cost must be 2-D with at least as many columns as rows")
    if k < 1:
        raise ValueError("k must be positive")
    if cost.shape[0] == 0:
        return [(np.empty(0, dtype=int), 0.0)]

    counter = itertools.count()
    root_forbidden: frozenset[tuple[int, int]] = frozenset()
    root = _solve_constrained_assignment(cost, (), root_forbidden)
    if root is None:
        return []
    root_assignment, root_cost = root
    heap: list[
        tuple[float, int, tuple[int, ...], frozenset[tuple[int, int]], tuple[int, ...]]
    ] = [(root_cost, next(counter), (), root_forbidden, root_assignment)]
    seen_nodes: set[tuple[tuple[int, ...], frozenset[tuple[int, int]]]] = {
        ((), root_forbidden)
    }
    seen_solutions: set[tuple[int, ...]] = set()
    results: list[tuple[np.ndarray, float]] = []

    while heap and len(results) < k:
        value, _, prefix, forbidden, assignment = heapq.heappop(heap)
        if assignment not in seen_solutions:
            seen_solutions.add(assignment)
            results.append((np.asarray(assignment, dtype=int), value))

        for split in range(len(prefix), cost.shape[0]):
            child_prefix = assignment[:split]
            child_forbidden = forbidden | {(split, assignment[split])}
            node_key = (child_prefix, child_forbidden)
            if node_key in seen_nodes:
                continue
            seen_nodes.add(node_key)
            solution = _solve_constrained_assignment(
                cost, child_prefix, child_forbidden
            )
            if solution is None:
                continue
            child_assignment, child_cost = solution
            heapq.heappush(
                heap,
                (
                    child_cost,
                    next(counter),
                    child_prefix,
                    child_forbidden,
                    child_assignment,
                ),
            )
    return results


def _normalize_hypotheses(hypotheses: list[_Hypothesis]) -> list[_Hypothesis]:
    hypotheses.sort(key=lambda item: item.log_weight, reverse=True)
    maximum = hypotheses[0].log_weight
    normalizer = maximum + math.log(
        sum(math.exp(item.log_weight - maximum) for item in hypotheses)
    )
    for item in hypotheses:
        item.log_weight -= normalizer
    return hypotheses


def _hypothesis_weights(hypotheses: Sequence[_Hypothesis]) -> np.ndarray:
    weights = np.exp([hypothesis.log_weight for hypothesis in hypotheses])
    return weights / weights.sum()


def _assignment_marginals(
    hypotheses: Sequence[_Hypothesis],
    history_index: int,
    n_tracks: int,
    n_detections: int,
) -> np.ndarray:
    probabilities = np.zeros((n_tracks, n_detections + 1), dtype=float)
    for weight, hypothesis in zip(_hypothesis_weights(hypotheses), hypotheses):
        assignment = hypothesis.history[history_index]
        for track_index, detection_index in enumerate(assignment):
            event_index = detection_index if detection_index >= 0 else n_detections
            probabilities[track_index, event_index] += weight
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), _EPS)
    return probabilities


class DeformEOTTracker:
    """Approximate Bayesian tracker for a persistent initialized neuron cohort."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()

    def _predict(
        self,
        states: Sequence[_TrackState],
        transform: AffineTransform,
    ) -> list[_Prediction]:
        config = self.config
        process = config.process_sigma**2 * np.eye(3)
        predictions: list[_Prediction] = []
        for state in states:
            global_position = transform.apply(state.position[None, :])[0]
            velocity = config.velocity_decay * (transform.matrix @ state.velocity)
            position = global_position + velocity
            covariance = (
                transform.matrix @ state.covariance @ transform.matrix.T + process
            )
            visibility = (
                state.visibility * config.visible_stay_probability
                + (1 - state.visibility) * config.reappearance_probability
            )
            if state.extent is None:
                extent = None
            else:
                extent = _nearest_spd(
                    transform.matrix @ state.extent @ transform.matrix.T
                    + config.extent_process_sigma**2 * np.eye(3)
                )
            predictions.append(
                _Prediction(
                    position,
                    global_position,
                    velocity,
                    covariance,
                    float(np.clip(visibility, _EPS, 1 - _EPS)),
                    extent,
                )
            )
        return predictions

    def _association_cost(
        self,
        predictions: Sequence[_Prediction],
        frame: DetectionFrame,
    ) -> np.ndarray:
        config = self.config
        n_tracks = len(predictions)
        n_detections = len(frame.positions)
        # Each row receives a private miss column, ensuring misses are independent.
        cost = np.full((n_tracks, n_detections + n_tracks), np.inf, dtype=float)
        measurement_covariance = config.measurement_sigma**2 * np.eye(3)

        for track_index, prediction in enumerate(predictions):
            innovation_covariance = prediction.covariance + measurement_covariance
            inverse = np.linalg.inv(innovation_covariance)
            _, logdet = np.linalg.slogdet(innovation_covariance)
            for detection_index, position in enumerate(frame.positions):
                residual = position - prediction.position
                mahalanobis_sq = float(residual @ inverse @ residual)
                if mahalanobis_sq > config.gate_mahalanobis_sq:
                    continue
                gaussian_nll = 0.5 * (
                    mahalanobis_sq + logdet + 3 * math.log(2 * math.pi)
                )
                extent_nll = 0.0
                if (
                    config.extent_weight > 0
                    and prediction.extent is not None
                    and frame.extents is not None
                ):
                    difference = _log_spd(frame.extents[detection_index]) - _log_spd(
                        prediction.extent
                    )
                    extent_nll = (
                        0.5
                        * config.extent_weight
                        * float(np.sum(difference**2))
                        / config.extent_log_sigma**2
                    )
                detection_prior = (
                    prediction.visibility
                    * config.detection_probability
                    * float(frame.confidences[detection_index])
                )
                # Likelihood ratio against a spatially uniform clutter process.
                cost[track_index, detection_index] = (
                    gaussian_nll
                    + extent_nll
                    - math.log(max(detection_prior, _EPS))
                    + math.log(config.clutter_density)
                ) / config.association_temperature

            miss_probability = 1 - prediction.visibility * config.detection_probability
            cost[track_index, n_detections + track_index] = -math.log(
                max(miss_probability, _EPS)
            ) / config.association_temperature
        return cost

    def _update_state(
        self,
        state: _TrackState,
        prediction: _Prediction,
        frame: DetectionFrame,
        detection_index: int,
    ) -> _TrackState:
        config = self.config
        if detection_index < 0:
            miss_likelihood = 1 - config.detection_probability
            denominator = 1 - prediction.visibility * config.detection_probability
            visibility = prediction.visibility * miss_likelihood / max(denominator, _EPS)
            return _TrackState(
                prediction.position.copy(),
                prediction.velocity.copy(),
                prediction.covariance.copy(),
                float(np.clip(visibility, _EPS, 1 - _EPS)),
                state.missed_frames + 1,
                None if prediction.extent is None else prediction.extent.copy(),
            )

        measurement_covariance = config.measurement_sigma**2 * np.eye(3)
        innovation_covariance = prediction.covariance + measurement_covariance
        gain = prediction.covariance @ np.linalg.inv(innovation_covariance)
        residual = frame.positions[detection_index] - prediction.position
        position = prediction.position + gain @ residual
        identity = np.eye(3)
        covariance = (
            (identity - gain)
            @ prediction.covariance
            @ (identity - gain).T
            + gain @ measurement_covariance @ gain.T
        )
        velocity = prediction.velocity + config.velocity_gain * residual

        observed_extent = None if frame.extents is None else frame.extents[detection_index]
        if observed_extent is None:
            extent = prediction.extent
        elif prediction.extent is None:
            extent = observed_extent.copy()
        else:
            extent = _nearest_spd(
                (1 - config.extent_update_gain) * prediction.extent
                + config.extent_update_gain * observed_extent
            )
        return _TrackState(
            position,
            velocity,
            _nearest_spd(covariance),
            1 - _EPS,
            0,
            None if extent is None else extent.copy(),
        )

    def _expand(
        self,
        hypothesis: _Hypothesis,
        frame: DetectionFrame,
    ) -> list[_Hypothesis]:
        source = np.vstack([state.position for state in hypothesis.states])
        transform = estimate_shared_affine(
            source,
            frame.positions,
            iterations=self.config.affine_iterations,
            ridge=self.config.affine_ridge,
            huber_delta=self.config.affine_huber_delta,
            min_scale=self.config.affine_min_scale,
            max_scale=self.config.affine_max_scale,
        )
        predictions = self._predict(hypothesis.states, transform)
        cost = self._association_cost(predictions, frame)
        assignments = k_best_assignments(cost, self.config.assignments_per_hypothesis)
        branches: list[_Hypothesis] = []
        n_detections = len(frame.positions)
        for column_assignment, assignment_cost in assignments:
            detection_assignment = tuple(
                int(column) if column < n_detections else -1
                for column in column_assignment
            )
            states = [
                self._update_state(state, prediction, frame, detection_index)
                for state, prediction, detection_index in zip(
                    hypothesis.states, predictions, detection_assignment
                )
            ]
            branches.append(
                _Hypothesis(
                    states,
                    hypothesis.log_weight - assignment_cost,
                    [*hypothesis.history, detection_assignment],
                    [*hypothesis.transforms, transform],
                )
            )
        return branches

    @staticmethod
    def _mixture_state(
        hypotheses: Sequence[_Hypothesis],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray | None, ...], np.ndarray]:
        weights = _hypothesis_weights(hypotheses)
        n_tracks = len(hypotheses[0].states)
        positions = np.empty((n_tracks, 3), dtype=float)
        covariances = np.empty((n_tracks, 3, 3), dtype=float)
        extents: list[np.ndarray | None] = []
        visibility = np.empty(n_tracks, dtype=float)
        for track_index in range(n_tracks):
            samples = np.vstack(
                [hypothesis.states[track_index].position for hypothesis in hypotheses]
            )
            mean = np.average(samples, axis=0, weights=weights)
            positions[track_index] = mean
            covariance = np.zeros((3, 3), dtype=float)
            extent_values: list[np.ndarray] = []
            extent_weights: list[float] = []
            for weight, hypothesis, sample in zip(weights, hypotheses, samples):
                state = hypothesis.states[track_index]
                delta = sample - mean
                covariance += weight * (state.covariance + np.outer(delta, delta))
                if state.extent is not None:
                    extent_values.append(state.extent)
                    extent_weights.append(float(weight))
            covariances[track_index] = _nearest_spd(covariance)
            if extent_values:
                normalized = np.asarray(extent_weights) / np.sum(extent_weights)
                extents.append(
                    _nearest_spd(
                        sum(
                            weight * extent
                            for weight, extent in zip(normalized, extent_values)
                        )
                    )
                )
            else:
                extents.append(None)
            visibility[track_index] = np.average(
                [hypothesis.states[track_index].visibility for hypothesis in hypotheses],
                weights=weights,
            )
        return positions, covariances, tuple(extents), visibility

    def track(
        self,
        frames: Sequence[DetectionFrame | np.ndarray],
        *,
        initial_track_ids: Sequence[Hashable] | None = None,
    ) -> TrackingResult:
        """Track detections through a sequence.

        Every first-frame detection initializes one persistent target.  This is
        suitable for fDNC/ASCENT coordinate arrays when the reference frame has a
        named or otherwise fixed identity ordering.  New biological targets are
        intentionally not spawned in later frames.
        """

        if not frames:
            raise ValueError("at least one frame is required")
        normalized_frames = [
            frame if isinstance(frame, DetectionFrame) else DetectionFrame(frame)
            for frame in frames
        ]
        first = normalized_frames[0]
        if len(first.positions) == 0:
            raise ValueError("the initialization frame must contain detections")
        if initial_track_ids is None:
            track_ids: tuple[Hashable, ...] = tuple(range(len(first.positions)))
        else:
            track_ids = tuple(initial_track_ids)
            if len(track_ids) != len(first.positions):
                raise ValueError("initial_track_ids must match first-frame detections")
            if len(set(track_ids)) != len(track_ids):
                raise ValueError("initial_track_ids must be unique")

        initial_covariance = self.config.initial_position_sigma**2 * np.eye(3)
        states = [
            _TrackState(
                position.copy(),
                np.zeros(3),
                initial_covariance.copy(),
                1 - _EPS,
                0,
                None if first.extents is None else first.extents[index].copy(),
            )
            for index, position in enumerate(first.positions)
        ]
        identity_assignment = tuple(range(len(first.positions)))
        identity_transform = AffineTransform(np.eye(3), np.zeros(3), len(states), 0.0)
        hypotheses = [
            _Hypothesis(states, 0.0, [identity_assignment], [identity_transform])
        ]

        first_probabilities = np.eye(len(states), len(states) + 1)
        outputs = [
            FramePosterior(
                frame_index=0,
                time=0 if first.time is None else first.time,
                track_ids=track_ids,
                object_ids=(
                    tuple(range(len(first.positions)))
                    if first.object_ids is None
                    else tuple(first.object_ids)
                ),
                positions=first.positions.copy(),
                filter_association_probabilities=first_probabilities.copy(),
                association_probabilities=first_probabilities.copy(),
                association_entropy=np.zeros(len(states)),
                map_assignment=np.arange(len(states), dtype=int),
                posterior_positions=first.positions.copy(),
                posterior_covariances=np.repeat(
                    initial_covariance[None, :, :], len(states), axis=0
                ),
                posterior_extents=tuple(state.extent for state in states),
                visibility_probability=np.ones(len(states)),
                shared_transform=identity_transform,
                hypothesis_count=1,
            )
        ]
        last_finalized = 0

        for frame_index, frame in enumerate(normalized_frames[1:], start=1):
            branches: list[_Hypothesis] = []
            for hypothesis in hypotheses:
                branches.extend(self._expand(hypothesis, frame))
            if not branches:
                raise RuntimeError(f"no feasible assignment at frame {frame_index}")
            hypotheses = _normalize_hypotheses(branches)[: self.config.max_hypotheses]
            hypotheses = _normalize_hypotheses(hypotheses)

            filter_probabilities = _assignment_marginals(
                hypotheses, frame_index, len(track_ids), len(frame.positions)
            )
            positions, covariances, extents, visibility = self._mixture_state(hypotheses)
            map_assignment = np.asarray(hypotheses[0].history[frame_index], dtype=int)
            probabilities_for_entropy = np.maximum(filter_probabilities, _EPS)
            output = FramePosterior(
                frame_index=frame_index,
                time=frame_index if frame.time is None else frame.time,
                track_ids=track_ids,
                object_ids=(
                    tuple(range(len(frame.positions)))
                    if frame.object_ids is None
                    else tuple(frame.object_ids)
                ),
                positions=frame.positions.copy(),
                filter_association_probabilities=filter_probabilities.copy(),
                association_probabilities=filter_probabilities.copy(),
                association_entropy=-np.sum(
                    probabilities_for_entropy * np.log(probabilities_for_entropy), axis=1
                ),
                map_assignment=map_assignment,
                posterior_positions=positions,
                posterior_covariances=covariances,
                posterior_extents=extents,
                visibility_probability=visibility,
                shared_transform=hypotheses[0].transforms[frame_index],
                hypothesis_count=len(hypotheses),
            )
            outputs.append(output)

            smooth_index = frame_index - self.config.fixed_lag
            if smooth_index > last_finalized:
                self._finalize_frame(outputs[smooth_index], hypotheses, smooth_index)
                last_finalized = smooth_index

        for frame_index in range(last_finalized + 1, len(outputs)):
            self._finalize_frame(outputs[frame_index], hypotheses, frame_index)
        return TrackingResult(outputs, track_ids, self.config)

    @staticmethod
    def _finalize_frame(
        output: FramePosterior,
        hypotheses: Sequence[_Hypothesis],
        history_index: int,
    ) -> None:
        probabilities = _assignment_marginals(
            hypotheses,
            history_index,
            len(output.track_ids),
            len(output.object_ids),
        )
        output.association_probabilities = probabilities
        safe = np.maximum(probabilities, _EPS)
        output.association_entropy = -np.sum(safe * np.log(safe), axis=1)
        output.map_assignment = np.asarray(
            hypotheses[0].history[history_index], dtype=int
        )
