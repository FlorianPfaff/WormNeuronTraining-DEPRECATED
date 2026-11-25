from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from deformeot.paper_phase2_mechanism import (
    _Child, _MechanismContext, _bank_sha256, _replay_admitted_path,
    _search_admitted_tree, track_mechanism,
)
from deformeot.tracker import DetectionFrame
from deformeot.wormid_benchmark import ReferenceTrackerConfig
from deformeot.wormid_pyrecest_mht import PyRecEstMHTConfig, track_pyrecest_mht


class BlindOnly:
    def __init__(self, clouds):
        self.frames = tuple(DetectionFrame(np.asarray(points, dtype=float), time=index)
                            for index, points in enumerate(clouds))

    @property
    def truth_indices(self):
        raise AssertionError('mechanism control read hidden identities')

    @property
    def initial_track_ids(self):
        raise AssertionError('mechanism control read semantic identities')


def config(**kwargs):
    return PyRecEstMHTConfig(use_gaussian_cost=False, **kwargs)


def tiny_sequence(frames=6):
    rng = np.random.default_rng(612)
    reference = np.asarray([[-2., 0., 0.], [0., 1., .4], [1.7, -.2, .8], [3.8, .4, -.6]])
    clouds = [reference]
    for index in range(1, frames):
        moved = reference + index * np.asarray([.12, -.03, .02]) + .015 * np.sin(reference * index)
        clouds.append(moved[rng.permutation(len(reference))])
    return BlindOnly(clouds)


def test_admitted_tree_search_finds_future_dependent_optimum_without_greedy_collapse():
    def step(prefix, frame):
        children = []
        for choice in range(3):
            score = -float(choice)
            if frame == 3 and prefix[0] == 2:
                score += 10
            children.append(_Child(score, (choice,), prefix + (choice,)))
        return children

    result = _search_admitted_tree((), 1, 3, step)
    exhaustive = []
    for path in product(range(3), repeat=3):
        score = -sum(path) + (10 if path[0] == 2 else 0)
        exhaustive.append((score, path))
    optimum = max(exhaustive, key=lambda row: row[0])
    assert result.score == optimum[0]
    assert tuple(row[0] for row in result.path) == optimum[1]
    assert result.first_bank == (2,)
    assert result.leaves == 27
    assert result.expanded_nodes == 13


def test_admitted_tree_ties_keep_the_first_ranked_path_and_reject_empty_children():
    result = _search_admitted_tree(None, 1, 2,
                                  lambda bank, frame: [_Child(0., (2,), 'first'), _Child(0., (1,), 'second')])
    assert result.path == ((2,), (2,))
    with pytest.raises(RuntimeError, match='lost'):
        _search_admitted_tree(None, 1, 1, lambda bank, frame: [])
    with pytest.raises(ValueError, match='finite'):
        _search_admitted_tree(None, 1, 1, lambda bank, frame: [_Child(np.nan, (0,), None)])


@pytest.mark.parametrize('mode', ['rolling_best_path', 'beam_best_path', 'beam_marginal', 'online'])
def test_lag_zero_controls_match_frozen_online_assignments_and_motion(mode):
    pytest.importorskip('pyrecest')
    sequence = tiny_sequence()
    settings = config(fixed_lag=0)
    original = track_pyrecest_mht(sequence, config=settings)
    result = track_mechanism(sequence, mode=mode, config=settings)
    assert all(np.array_equal(left, right) for left, right in zip(original.assignments, result.assignments))
    assert np.array_equal(original.cloud_motion_um, result.cloud_motion_um)


def test_marginal_wrapper_matches_frozen_lag_two_and_online_mode_forces_zero_lag():
    pytest.importorskip('pyrecest')
    sequence = tiny_sequence()
    original = track_pyrecest_mht(sequence, config=config())
    result = track_mechanism(sequence, mode='beam_marginal', config=config())
    assert all(np.array_equal(left, right) for left, right in zip(original.assignments, result.assignments))
    trace = []
    online = track_mechanism(sequence, mode='online', config=config(), trace_callback=trace.append)
    zero = track_pyrecest_mht(sequence, config=config(fixed_lag=0))
    assert all(row['lookahead'] == 0 for row in trace)
    assert all(np.array_equal(left, right) for left, right in zip(zero.assignments, online.assignments))


def test_wide_effectively_unpruned_beam_matches_exact_tree_given_same_prefix():
    pytest.importorskip('pyrecest')
    sequence = tiny_sequence(frames=4)
    # Every admitted path fits in the beam; the very large finite threshold is
    # checked against the realized window objective spread below.
    settings = config(max_global_hypotheses=27, prune_log_weight_delta=1e12,
                      candidate_cost_window=1e4)
    context = _MechanismContext(sequence, settings, ReferenceTrackerConfig())
    result = _search_admitted_tree(context.initial_bank, 1, 3, context.expand)
    tree_scores = []
    prefix_scores = {1: [], 2: [], 3: []}

    def enumerate_leaves(bank, frame, score):
        if frame == 4:
            tree_scores.append(score)
            return
        for child in context.expand(bank, frame):
            prefix_scores[frame].append(score + child.score)
            enumerate_leaves(child.bank, frame + 1, score + child.score)

    enumerate_leaves(context.initial_bank, 1, 0.)
    assert max(tree_scores) - min(tree_scores) < settings.prune_log_weight_delta
    assert all(max(scores) - min(scores) < settings.prune_log_weight_delta
               for scores in prefix_scores.values())
    tracker = context.tracker
    for index in range(1, 4):
        transition = context.transition(index)
        tracker.predict_linear(transition.system, context.process_covariance, transition.offset)
        tracker.reference_cost = transition.reference_cost
        tracker.update_linear(transition.observations.T, context.measurement_matrix, context.measurement_covariance)
    best = tracker.get_best_hypothesis_index()
    assert tuple(tuple(row) for row in tracker._global_hypothesis_histories[best]) == result.path
    assert tracker.get_number_of_global_hypotheses() == result.leaves


@pytest.mark.parametrize('mode', ['rolling_best_path', 'beam_best_path', 'beam_marginal'])
def test_tail_shortens_lookahead_and_every_commit_obeys_two_scan_bound(mode):
    pytest.importorskip('pyrecest')
    trace = []
    result = track_mechanism(tiny_sequence(frames=6), mode=mode, config=config(), trace_callback=trace.append)
    assert len(result.assignments) == 6
    assert [row['lookahead'] for row in trace] == [2, 2, 2, 1, 0]
    assert [row['committed_frame'] for row in trace] == [1, 2, 3, 4, 5]
    assert all(row['last_observed_frame'] <= row['committed_frame'] + 2 for row in trace)
    if mode == 'rolling_best_path':
        assert all(row['admitted_leaves'] <= 27 for row in trace)
        assert all(row['retained_uncommitted_hypotheses'] == 0 for row in trace)


@pytest.mark.parametrize('mode', ['rolling_best_path', 'beam_best_path', 'beam_marginal', 'online'])
def test_future_beyond_commitment_horizon_cannot_affect_assignment_or_state(mode):
    pytest.importorskip('pyrecest')
    sequence = tiny_sequence(frames=6)
    changed = BlindOnly([frame.positions for frame in sequence.frames[:4]] +
                        [frame.positions[::-1] + [25., -10., 3.] for frame in sequence.frames[4:]])
    left_trace, right_trace = [], []
    left = track_mechanism(sequence, mode=mode, config=config(), trace_callback=left_trace.append)
    right = track_mechanism(changed, mode=mode, config=config(), trace_callback=right_trace.append)
    assert np.array_equal(left.assignments[1], right.assignments[1])
    assert left_trace[0]['committed_state_sha256'] == right_trace[0]['committed_state_sha256']
    assert left_trace[0]['last_observed_frame'] <= 3


def test_rolling_does_not_even_read_beyond_the_first_commit_horizon():
    pytest.importorskip('pyrecest')
    sequence = tiny_sequence(frames=4)

    class ForbiddenFrame:
        @property
        def positions(self):
            raise AssertionError('a future observation was read before the first commitment')

    class FirstCommitComplete(Exception):
        pass

    def stop_after_first(row):
        assert row['committed_frame'] == 1 and row['last_observed_frame'] == 3
        raise FirstCommitComplete

    extended = SimpleNamespace(frames=(*sequence.frames, ForbiddenFrame()))
    with pytest.raises(FirstCommitComplete):
        track_mechanism(extended, mode='rolling_best_path', config=config(), trace_callback=stop_after_first)


def test_identical_prefix_replay_reproduces_state_without_future_contamination():
    pytest.importorskip('pyrecest')
    sequence = tiny_sequence(frames=5)
    context = _MechanismContext(sequence, config(), ReferenceTrackerConfig())
    before = _bank_sha256(context.initial_bank)
    selected = _search_admitted_tree(context.initial_bank, 1, 3, context.expand)
    replayed, score = _replay_admitted_path(context, context.initial_bank, 1, selected.path)
    again, again_score = _replay_admitted_path(context, context.initial_bank, 1, selected.path)
    first, _ = _replay_admitted_path(context, context.initial_bank, 1, selected.path[:1])
    assert _bank_sha256(replayed) == _bank_sha256(again)
    assert score == again_score == selected.score
    assert _bank_sha256(first) == _bank_sha256(selected.first_bank)
    assert _bank_sha256(context.initial_bank) == before


def test_all_rolling_committed_states_equal_independent_prefix_replay():
    pytest.importorskip('pyrecest')
    sequence, trace = tiny_sequence(), []
    result = track_mechanism(sequence, mode='rolling_best_path', config=config(), trace_callback=trace.append)
    context = _MechanismContext(sequence, config(), ReferenceTrackerConfig())
    for index, record in enumerate(trace, start=1):
        replayed, _ = _replay_admitted_path(context, context.initial_bank, 1, result.assignments[1:index + 1])
        assert _bank_sha256(replayed) == record['committed_state_sha256']


def test_invalid_modes_and_excessive_rolling_search_are_rejected_before_native_import():
    with pytest.raises(ValueError, match='unknown'):
        track_mechanism(SimpleNamespace(frames=()), mode='delay_only', config=config())
    with pytest.raises(ValueError, match='bounded'):
        track_mechanism(SimpleNamespace(frames=()), mode='rolling_best_path', config=config(fixed_lag=3))
    with pytest.raises(ValueError, match='bounded'):
        track_mechanism(SimpleNamespace(frames=()), mode='rolling_best_path', config=config(hypotheses_per_parent=4))


def test_empty_or_changing_cohort_is_rejected():
    pytest.importorskip('pyrecest')
    with pytest.raises(ValueError, match='empty'):
        track_mechanism(SimpleNamespace(frames=()), mode='online', config=config())
    sequence = tiny_sequence(frames=3)
    changed = BlindOnly([sequence.frames[0].positions, sequence.frames[1].positions[:-1]])
    with pytest.raises(ValueError, match='fixed exhaustive cohort'):
        track_mechanism(changed, mode='rolling_best_path', config=config())
