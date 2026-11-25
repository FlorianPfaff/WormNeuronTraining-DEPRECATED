"""Bounded PyRecEst MHT for the fixed, exhaustive WormID cohort.

PyRecEst supplies the actual Gaussian filter banks, alternative global histories,
measurement updates, hypothesis weights and beam pruning.  The subclass below
adapts its candidate-scoring and ranked-assignment hooks to this experiment:

* an anonymous shared affine map transports all Gaussian states;
* the immutable first-frame cost is included BEFORE ranked assignment generation;
* near-optimal candidate edges plus a complete Hungarian backbone give a sparse,
  feasible bipartite graph;
* independent graph components use PyRecEst's Murty implementation, with a
  cardinality bonus conditioning the experiment on exhaustive observations;
* delayed whole-permutation commitments condition the surviving histories and
  release snapshots older than the requested lag.

The near-best edge window, per-track cap and finite hypothesis beam are deliberate
approximations.  A correct association outside those gates cannot be recovered.
The reference term is a geometric score, not an independent Gaussian observation;
the resulting hypothesis weights must not be called calibrated probabilities.
Residual velocity is measured per SELECTED scan, so ``fixed_lag=2`` permits two
later selected scans, irrespective of the original recording stride.  The final
scans use only the shorter lookahead available at the end of the sequence.

PyRecEst is optional and imported only when tracking is requested.  Its NumPy
backend is required by this adapter and ranked-assignment implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .wormid import IdentityBlindSequence
from .wormid_benchmark import (
    ReferenceTrackerConfig,
    TrackResult,
    _affine,
    _spatial_cost,
    _validate_positions,
    descriptor_cost,
    distance_profile_descriptor,
)


@dataclass(frozen=True)
class PyRecEstMHTConfig:
    """Compute budget, scoring and selected-scan latency for native MHT."""

    max_global_hypotheses: int = 4
    hypotheses_per_parent: int = 3
    max_measurements_per_track: int = 5
    fixed_lag: int = 2
    measurement_sigma_um: float = 0.3
    process_sigma_um: float = 0.6
    initial_sigma_um: float = 0.3
    reference_cost_weight: float = 1.0
    candidate_cost_window: float = 4.0
    prune_log_weight_delta: float = 20.0
    use_gaussian_cost: bool = True
    velocity_weight: float = 0.0
    velocity_decay: float = 0.8

    def __post_init__(self) -> None:
        for name in (
            "max_global_hypotheses", "hypotheses_per_parent",
            "max_measurements_per_track", "fixed_lag",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            minimum = 0 if name == "fixed_lag" else 1
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        for name in (
            "measurement_sigma_um", "process_sigma_um", "initial_sigma_um",
            "candidate_cost_window", "prune_log_weight_delta",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("reference_cost_weight", "velocity_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not np.isfinite(self.velocity_decay) or not 0 <= self.velocity_decay <= 1:
            raise ValueError("velocity_decay must lie in [0, 1]")
        if not isinstance(self.use_gaussian_cost, bool):
            raise ValueError("use_gaussian_cost must be a boolean")


def _components(candidate_measurements):
    """Connected components of a sparse square bipartite candidate graph."""
    n = len(candidate_measurements)
    parent = list(range(2 * n))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for track, candidates in enumerate(candidate_measurements):
        for measurement, _ in candidates:
            left, right = root(track), root(n + int(measurement))
            parent[right] = left
    groups = {}
    for track in range(n):
        group = groups.setdefault(root(track), [[], set()])
        group[0].append(track)
        group[1].update(int(measurement) for measurement, _ in candidate_measurements[track])
    return [(tracks, sorted(measurements)) for tracks, measurements in groups.values()]


def _combine_component_assignments(component_solutions, n_tracks, maximum):
    """Exact top-K sum of already-ranked independent component solutions."""
    if not component_solutions:
        return [(0.0, tuple())]
    initial = (0,) * len(component_solutions)
    initial_cost = sum(solutions[0][0] for _, solutions in component_solutions)
    heap = [(initial_cost, initial)]
    seen = {initial}
    result = []
    while heap and len(result) < maximum:
        total_cost, ranks = heappop(heap)
        assignment = [-1] * n_tracks
        for rank, (tracks, solutions) in zip(ranks, component_solutions):
            for track, measurement in zip(tracks, solutions[rank][1]):
                assignment[track] = int(measurement)
        result.append((-float(total_cost), tuple(assignment)))
        for component, (_, solutions) in enumerate(component_solutions):
            next_rank = ranks[component] + 1
            if next_rank >= len(solutions):
                continue
            next_ranks = list(ranks)
            next_ranks[component] = next_rank
            next_ranks = tuple(next_ranks)
            if next_ranks in seen:
                continue
            seen.add(next_ranks)
            next_cost = (
                total_cost - solutions[ranks[component]][0] + solutions[next_rank][0]
            )
            heappush(heap, (next_cost, next_ranks))
    return result


@lru_cache(maxsize=1)
def _native_mht_type():
    """Create the small scoring adapter without making PyRecEst mandatory."""
    try:
        import pyrecest.backend as backend
        from pyrecest.filters.multi_hypothesis_tracker import MultiHypothesisTracker
        from pyrecest.utils import murty_k_best_assignments
    except ImportError as exc:
        raise ImportError(
            "PyRecEst MHT requires a separate Python >=3.11 environment with PyRecEst"
        ) from exc
    if getattr(backend, "__backend_name__", None) != "numpy":
        raise ValueError("the WormID PyRecEst MHT adapter requires PYRECEST_BACKEND=numpy")

    class ExhaustiveMHT(MultiHypothesisTracker):
        """Native MHT with sparse scores and exhaustive-assignment conditioning."""

        def __init__(self, initial_prior, *, config, geometry):
            self.config = config
            self.geometry = geometry
            self.reference_cost = None
            self.maximum_component_size = 0
            self.maximum_history_length = 0
            super().__init__(
                initial_prior=initial_prior,
                association_param={
                    "max_global_hypotheses": config.max_global_hypotheses,
                    "max_hypotheses_per_global_hypothesis": config.hypotheses_per_parent,
                    "max_measurements_per_track": config.max_measurements_per_track,
                    "prune_log_weight_delta": config.prune_log_weight_delta,
                    "pruning_strategy": "diverse_top_k",
                    "diversity_history_length": max(1, config.fixed_lag + 1),
                    "max_hypotheses_per_signature": 1,
                },
                log_prior_estimates=False,
                log_posterior_estimates=False,
            )

        def _get_base_log_score(self, n_meas):
            # Fixed-cohort conditioning removes miss, clutter and existence terms.
            return 0.0

        def _build_candidate_measurements(
            self, filter_bank, measurements, measurement_matrix, cov_mats_meas
        ):
            positions = np.stack([state.get_point_estimate() for state in filter_bank])
            predicted = positions @ np.asarray(measurement_matrix).T
            observations = np.asarray(measurements).T
            if self.config.use_gaussian_cost:
                covariances = np.stack([state.filter_state.C for state in filter_bank])
                matrix = np.asarray(measurement_matrix)
                innovations = observations[None, :, :] - predicted[:, None, :]
                innovation_covariances = (
                    matrix[None] @ covariances @ matrix.T[None]
                    + np.asarray(cov_mats_meas)[None]
                )
                inverse = np.linalg.inv(innovation_covariances)
                mahalanobis = np.einsum("nmi,nij,nmj->nm", innovations, inverse, innovations)
                signs, log_determinants = np.linalg.slogdet(innovation_covariances)
                if np.any(signs <= 0):
                    raise ValueError("innovation covariance must be positive definite")
                cost = 0.5 * (
                    mahalanobis + log_determinants[:, None]
                    + observations.shape[1] * np.log(2.0 * np.pi)
                )
            else:
                cost = _spatial_cost(
                    predicted, observations, self.geometry.spatial_scale_floor_um
                )
            if self.reference_cost is not None:
                cost += self.reference_cost
            rows, backbone = linear_sum_assignment(cost)
            if len(rows) != len(filter_bank):
                raise ValueError("MHT observations must exhaust the fixed cohort")
            candidates = []
            for track in range(len(filter_bank)):
                # The complete optimum remains feasible even if row-wise caps
                # would otherwise violate Hall's condition.
                order = np.argsort(cost[track], kind="stable")
                cutoff = cost[track, order[0]] + self.config.candidate_cost_window
                selected = [
                    int(index) for index in order
                    if cost[track, index] <= cutoff
                ][:self.config.max_measurements_per_track]
                required = int(backbone[track])
                if required not in selected:
                    selected.append(required)
                candidates.append([(index, -float(cost[track, index])) for index in selected])
            return candidates

        def _enumerate_candidate_assignments(self, candidate_measurements, base_log_score):
            component_solutions = []
            maximum = self.config.hypotheses_per_parent
            for tracks, measurements in _components(candidate_measurements):
                if len(tracks) != len(measurements):
                    raise RuntimeError("Hungarian backbone did not balance a component")
                self.maximum_component_size = max(self.maximum_component_size, len(tracks))
                lookup = {measurement: index for index, measurement in enumerate(measurements)}
                costs = np.full((len(tracks), len(measurements)), np.inf)
                for row, track in enumerate(tracks):
                    for measurement, gain in candidate_measurements[track]:
                        costs[row, lookup[int(measurement)]] = -float(gain)
                if len(tracks) == 1:
                    solutions = [(float(costs[0, 0]), (measurements[0],))]
                else:
                    # A matched edge receives a bonus greater than twice the
                    # absolute total finite cost, so exhaustive feasible solutions
                    # precede every partial solution. Strip it from reported scores.
                    bonus = 2.0 * (float(np.abs(costs[np.isfinite(costs)]).sum()) + 1.0)
                    ranked = murty_k_best_assignments(costs - bonus, k=maximum)
                    solutions = []
                    for record in ranked:
                        assignment = np.asarray(record["assignment"], dtype=int)
                        if np.any(assignment < 0):
                            continue
                        actual_cost = float(costs[np.arange(len(tracks)), assignment].sum())
                        solutions.append((actual_cost, tuple(measurements[index] for index in assignment)))
                    if not solutions:
                        raise RuntimeError("native PyRecEst Murty lost a feasible exhaustive assignment")
                component_solutions.append((tracks, solutions))
            return _combine_component_assignments(
                component_solutions, len(candidate_measurements), maximum
            )

        def commit_oldest(self):
            distribution = self.get_assignment_distribution(time_index=0)
            assignment = max(distribution, key=distribution.get)
            indices = [
                index for index, history in enumerate(self._global_hypothesis_histories)
                if tuple(history[0]) == assignment
            ]
            self._global_hypotheses = [self._global_hypotheses[index] for index in indices]
            self._global_base_log_weights = np.asarray([
                self._global_base_log_weights[index] for index in indices
            ])
            self._global_hypothesis_histories = [
                self._global_hypothesis_histories[index][1:] for index in indices
            ]
            self._global_filter_bank_histories = [
                self._global_filter_bank_histories[index][1:] for index in indices
            ]
            self._normalize_base_log_weights()
            self._refresh_effective_log_weights()
            return np.asarray(assignment, dtype=np.int64)

    return ExhaustiveMHT


def _build_mht(reference, config, geometry):
    from pyrecest.distributions import GaussianDistribution

    covariance = np.diag([config.initial_sigma_um ** 2] * 6)
    initial = [
        GaussianDistribution(np.concatenate([position, np.zeros(3)]), covariance.copy())
        for position in reference
    ]
    return _native_mht_type()(initial, config=config, geometry=geometry)


def track_pyrecest_mht(
    sequence: IdentityBlindSequence,
    *,
    config: PyRecEstMHTConfig = PyRecEstMHTConfig(),
    geometry: ReferenceTrackerConfig = ReferenceTrackerConfig(),
) -> TrackResult:
    """Run native MHT, returning exhaustive assignments after selected-scan lag.

    Only ``sequence.frames[*].positions`` is consumed.  Hidden evaluation fields,
    stable ROI ids, semantic labels, intensities and calcium traces are unused.
    """
    if not sequence.frames:
        raise ValueError("sequence is empty")
    reference = _validate_positions(sequence.frames[0].positions)
    n = len(reference)
    if any(len(frame.positions) != n for frame in sequence.frames):
        raise ValueError("PyRecEst MHT benchmark requires a fixed exhaustive cohort")
    started = time.perf_counter()
    _native_mht_type()  # Friendly dependency/backend validation before constructing states.
    tracker = _build_mht(reference, config, geometry)
    assignments = [np.arange(n, dtype=np.int64)]
    cloud_motion = [0.0]
    reference_descriptor = None
    if geometry.descriptor_weight:
        reference_descriptor = distance_profile_descriptor(
            reference, quantiles=geometry.descriptor_quantiles
        )
    previous_cloud = reference
    measurement_matrix = np.column_stack([np.eye(3), np.zeros((3, 3))])
    measurement_covariance = config.measurement_sigma_um ** 2 * np.eye(3)
    process_covariance = config.process_sigma_um ** 2 * np.eye(6)
    for frame in sequence.frames[1:]:
        current = _validate_positions(frame.positions)
        distances = cdist(previous_cloud, current)
        cloud_motion.append(float(0.5 * (
            np.median(distances.min(axis=1)) + np.median(distances.min(axis=0))
        )))
        transform = _affine(previous_cloud, current, geometry)
        system = np.zeros((6, 6))
        system[:3, :3] = transform.matrix
        system[:3, 3:] = config.velocity_weight * transform.matrix
        system[3:, 3:] = config.velocity_decay * transform.matrix
        tracker.predict_linear(
            system, process_covariance,
            np.concatenate([transform.offset, np.zeros(3)]),
        )
        reference_prediction = _affine(reference, current, geometry).apply(reference)
        tracker.reference_cost = (
            config.reference_cost_weight * geometry.reference_weight
            * _spatial_cost(reference_prediction, current, geometry.spatial_scale_floor_um)
        )
        if reference_descriptor is not None:
            descriptor = distance_profile_descriptor(current, quantiles=geometry.descriptor_quantiles)
            tracker.reference_cost += geometry.descriptor_weight * descriptor_cost(
                reference_descriptor, descriptor
            )
        tracker.update_linear(current.T, measurement_matrix, measurement_covariance)
        history_length = len(tracker.global_hypothesis_histories[0])
        tracker.maximum_history_length = max(tracker.maximum_history_length, history_length)
        if history_length > config.fixed_lag:
            assignments.append(tracker.commit_oldest())
        previous_cloud = current
    while tracker.global_hypothesis_histories and tracker.global_hypothesis_histories[0]:
        assignments.append(tracker.commit_oldest())
    if len(assignments) != len(sequence.frames):
        raise RuntimeError("MHT output length does not match selected scans")
    for assignment in assignments:
        if not np.array_equal(np.sort(assignment), np.arange(n)):
            raise RuntimeError("MHT did not emit an exhaustive permutation")
    return TrackResult(
        method="pyrecest_mht",
        assignments=tuple(assignments),
        runtime_seconds=float(time.perf_counter() - started),
        cloud_motion_um=np.asarray(cloud_motion, dtype=float),
    )
