"""Causal information-matched mechanism controls without changing frozen MHT.

The rolling control exhaustively searches the admitted top-three assignment tree
over at most three observations, commits only its oldest permutation, and starts
again from that committed Gaussian state. It is exact within that gated tree,
not over all possible permutations, and maintains no uncommitted hypothesis bank
between windows. Beam controls use native PyRecEst pruning and differ only in
whether the oldest assignment is chosen by best-path score or marginal mass.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial.distance import cdist

from .wormid import IdentityBlindSequence
from .wormid_benchmark import (
    ReferenceTrackerConfig, TrackResult, _affine, _spatial_cost, _validate_positions,
    descriptor_cost, distance_profile_descriptor,
)
from .wormid_pyrecest_mht import PyRecEstMHTConfig, _build_mht


MODES = ('rolling_best_path', 'beam_best_path', 'beam_marginal', 'online')


@dataclass(frozen=True)
class _Transition:
    observations: np.ndarray
    system: np.ndarray
    offset: np.ndarray
    reference_cost: np.ndarray


@dataclass(frozen=True)
class _Child:
    score: float
    assignment: tuple[int, ...]
    bank: Any


@dataclass(frozen=True)
class _SearchResult:
    score: float
    path: tuple[tuple[int, ...], ...]
    first_bank: Any
    leaves: int
    expanded_nodes: int


def _search_admitted_tree(bank, start, end, expand) -> _SearchResult:
    """Depth-first maximum of a finite admitted tree, retaining stable first ties.

The callback must return children in the native ranked-assignment order and must
not mutate its parent bank. Only the best root-to-leaf path is retained.
"""
    if end < start:
        raise ValueError('a search window must contain at least one observation')
    best_score, best_path, best_first = -np.inf, None, None
    leaves, expanded_nodes = 0, 0

    def visit(current, frame, score, path, first_bank):
        nonlocal best_score, best_path, best_first, leaves, expanded_nodes
        if frame > end:
            leaves += 1
            if best_path is None or score > best_score:
                best_score, best_path, best_first = score, path, first_bank
            return
        children = expand(current, frame)
        if not children:
            raise RuntimeError('the admitted assignment tree lost an exhaustive path')
        expanded_nodes += 1
        for child in children:
            if not np.isfinite(child.score):
                raise ValueError('admitted assignment scores must be finite')
            visit(child.bank, frame + 1, score + child.score, path + (child.assignment,),
                  child.bank if first_bank is None else first_bank)

    visit(bank, start, 0.0, (), None)
    return _SearchResult(float(best_score), best_path, best_first, leaves, expanded_nodes)


def _bank_sha256(bank) -> str:
    digest = hashlib.sha256()
    for state in bank:
        digest.update(np.asarray(state.get_point_estimate(), dtype='<f8').tobytes())
        digest.update(np.asarray(state.filter_state.C, dtype='<f8').tobytes())
    return digest.hexdigest()


class _MechanismContext:
    """Lazily form only the anonymous transitions requested by a causal window."""

    def __init__(self, sequence, config, geometry):
        if not sequence.frames:
            raise ValueError('sequence is empty')
        self.frames, self.config, self.geometry = sequence.frames, config, geometry
        self.reference = _validate_positions(self.frames[0].positions)
        self.count = len(self.reference)
        self.tracker = _build_mht(self.reference, config, geometry)
        self.initial_bank = deepcopy(self.tracker._global_hypotheses[0])
        self.measurement_matrix = np.column_stack([np.eye(3), np.zeros((3, 3))])
        self.measurement_covariance = config.measurement_sigma_um ** 2 * np.eye(3)
        self.process_covariance = config.process_sigma_um ** 2 * np.eye(6)
        self.reference_descriptor = (distance_profile_descriptor(self.reference, quantiles=geometry.descriptor_quantiles)
                                     if geometry.descriptor_weight else None)
        self.cloud_motion = np.zeros(len(self.frames), dtype=float)
        self.transitions = {}
        self.maximum_observed_frame = 0

    def transition(self, index):
        if index not in self.transitions:
            previous = _validate_positions(self.frames[index - 1].positions)
            current = _validate_positions(self.frames[index].positions)
            if len(previous) != self.count or len(current) != self.count:
                raise ValueError('mechanism controls require a fixed exhaustive cohort')
            distances = cdist(previous, current)
            self.cloud_motion[index] = .5 * (np.median(distances.min(axis=1)) + np.median(distances.min(axis=0)))
            transform = _affine(previous, current, self.geometry)
            system = np.zeros((6, 6))
            system[:3, :3] = transform.matrix
            system[:3, 3:] = self.config.velocity_weight * transform.matrix
            system[3:, 3:] = self.config.velocity_decay * transform.matrix
            offset = np.concatenate([transform.offset, np.zeros(3)])
            reference_prediction = _affine(self.reference, current, self.geometry).apply(self.reference)
            reference_cost = (self.config.reference_cost_weight * self.geometry.reference_weight
                              * _spatial_cost(reference_prediction, current, self.geometry.spatial_scale_floor_um))
            if self.reference_descriptor is not None:
                descriptor = distance_profile_descriptor(current, quantiles=self.geometry.descriptor_quantiles)
                reference_cost += self.geometry.descriptor_weight * descriptor_cost(self.reference_descriptor, descriptor)
            self.transitions[index] = _Transition(current, system, offset, reference_cost)
            self.maximum_observed_frame = max(self.maximum_observed_frame, index)
        return self.transitions[index]

    def predict_bank(self, bank, transition):
        predicted = deepcopy(bank)
        for state in predicted:
            # These are the same native Kalman calls used by MHT.predict_linear.
            state.predict_linear(transition.system, self.process_covariance, transition.offset)
        return predicted

    def expand(self, bank, index):
        transition = self.transition(index)
        predicted = self.predict_bank(bank, transition)
        self.tracker.reference_cost = transition.reference_cost
        candidates = self.tracker._build_candidate_measurements(
            predicted, transition.observations.T, self.measurement_matrix, self.measurement_covariance)
        ranked = self.tracker._enumerate_candidate_assignments(candidates, 0.0)
        return [_Child(float(score), tuple(map(int, assignment)), self.tracker._apply_assignment(
            predicted, assignment, transition.observations.T, self.measurement_matrix,
            self.measurement_covariance)) for score, assignment in ranked]

    def release_before(self, index):
        for old in tuple(self.transitions):
            if old < index:
                del self.transitions[old]


def _replay_admitted_path(context, bank, start, path):
    """Rebuild a path from its committed prefix, returning bank and raw score."""
    current, total = deepcopy(bank), 0.0
    for index, assignment in enumerate(path, start=start):
        selected = next((child for child in context.expand(current, index)
                         if child.assignment == tuple(assignment)), None)
        if selected is None:
            raise ValueError('replayed assignment is outside the admitted top-K tree')
        current, total = selected.bank, total + selected.score
    return current, float(total)


def _commit_best_path(tracker):
    """Commit the oldest assignment of the best native retained history.

Retain every beam history compatible with that commitment, exactly as the
frozen marginal-commit implementation does after choosing its permutation.
"""
    best = tracker.get_best_hypothesis_index()
    assignment = tuple(tracker._global_hypothesis_histories[best][0])
    indices = [index for index, history in enumerate(tracker._global_hypothesis_histories)
               if tuple(history[0]) == assignment]
    tracker._global_hypotheses = [tracker._global_hypotheses[index] for index in indices]
    tracker._global_base_log_weights = np.asarray([tracker._global_base_log_weights[index] for index in indices])
    tracker._global_hypothesis_histories = [tracker._global_hypothesis_histories[index][1:] for index in indices]
    tracker._global_filter_bank_histories = [tracker._global_filter_bank_histories[index][1:] for index in indices]
    tracker._normalize_base_log_weights()
    tracker._refresh_effective_log_weights()
    return np.asarray(assignment, dtype=np.int64)


def track_mechanism(
    sequence: IdentityBlindSequence,
    *,
    mode: str,
    config: PyRecEstMHTConfig,
    geometry: ReferenceTrackerConfig = ReferenceTrackerConfig(),
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrackResult:
    """Track using only anonymous coordinates, with selected-scan latency bounds.

``online`` fixes lag to zero; all other modes use config.fixed_lag. Rolling
search is intentionally limited to lag<=2 and <=3 children per node. Diagnostics
are emitted after each noninitial commitment. Their rolling score is an absolute
window score; beam log weights are explicitly relative and must not be compared
numerically with rolling scores. Callback execution is inside the runtime timer.
"""
    if mode not in MODES:
        raise ValueError(f'unknown mechanism mode {mode!r}')
    effective = replace(config, fixed_lag=0) if mode == 'online' else config
    if mode == 'rolling_best_path' and (effective.fixed_lag > 2 or effective.hypotheses_per_parent > 3):
        raise ValueError('rolling search is bounded to lag<=2 and at most three children per prefix')
    started = time.perf_counter()
    context = _MechanismContext(sequence, effective, geometry)
    assignments = [np.arange(context.count, dtype=np.int64)]

    if mode == 'rolling_best_path':
        committed = context.initial_bank
        for frame in range(1, len(context.frames)):
            last = min(frame + effective.fixed_lag, len(context.frames) - 1)
            result = _search_admitted_tree(committed, frame, last, context.expand)
            assignment = np.asarray(result.path[0], dtype=np.int64)
            assignments.append(assignment)
            # First-bank snapshots precede every future observation in the window.
            committed = deepcopy(result.first_bank)
            if trace_callback is not None:
                trace_callback({'mode': mode, 'committed_frame': frame,
                                'last_observed_frame': last, 'lookahead': last - frame,
                                'admitted_leaves': result.leaves, 'expanded_nodes': result.expanded_nodes,
                                'window_best_path_score': result.score,
                                'winning_path': [list(row) for row in result.path],
                                'assignment': assignment.tolist(), 'committed_state_sha256': _bank_sha256(committed),
                                'retained_uncommitted_hypotheses': 0})
            context.release_before(frame + 1)
    else:
        tracker = context.tracker

        def commit(last_observed):
            history = tracker._global_hypothesis_histories
            if mode == 'beam_best_path':
                selected = tracker.get_best_hypothesis_index()
            else:
                distribution = tracker.get_assignment_distribution(time_index=0)
                oldest = max(distribution, key=distribution.get)
                compatible = [i for i, path in enumerate(history) if tuple(path[0]) == oldest]
                selected = max(compatible, key=lambda i: float(tracker._global_log_weights[i]))
            if trace_callback is not None:
                state_hash = _bank_sha256(tracker._global_filter_bank_histories[selected][0])
                chosen_path = [list(row) for row in history[selected]]
                relative_weight = float(tracker._global_base_log_weights[selected])
                hypotheses_before = len(history)
            assignment = _commit_best_path(tracker) if mode == 'beam_best_path' else tracker.commit_oldest()
            frame = len(assignments)
            assignments.append(assignment)
            if trace_callback is not None:
                trace_callback({'mode': mode, 'committed_frame': frame,
                                'last_observed_frame': last_observed, 'lookahead': last_observed - frame,
                                'assignment': assignment.tolist(), 'winning_path': chosen_path,
                                'committed_state_sha256': state_hash,
                                'chosen_relative_base_log_weight': relative_weight,
                                'retained_hypotheses_before_commit': hypotheses_before,
                                'retained_filter_banks_after_commit': tracker.get_number_of_global_hypotheses(),
                                'retained_uncommitted_hypotheses': (tracker.get_number_of_global_hypotheses()
                                                                  if any(tracker._global_hypothesis_histories) else 0)})

        for frame in range(1, len(context.frames)):
            transition = context.transition(frame)
            tracker.predict_linear(transition.system, context.process_covariance, transition.offset)
            tracker.reference_cost = transition.reference_cost
            tracker.update_linear(transition.observations.T, context.measurement_matrix, context.measurement_covariance)
            if len(tracker._global_hypothesis_histories[0]) > effective.fixed_lag:
                commit(frame)
            context.release_before(frame + 1)
        while tracker._global_hypothesis_histories and tracker._global_hypothesis_histories[0]:
            commit(len(context.frames) - 1)

    if len(assignments) != len(context.frames):
        raise RuntimeError('mechanism control returned the wrong number of assignments')
    for assignment in assignments:
        if not np.array_equal(np.sort(assignment), np.arange(context.count)):
            raise RuntimeError('mechanism control did not produce an exhaustive permutation')
    return TrackResult(method=f'mechanism_{mode}', assignments=tuple(assignments),
                       runtime_seconds=float(time.perf_counter() - started),
                       cloud_motion_um=context.cloud_motion)
