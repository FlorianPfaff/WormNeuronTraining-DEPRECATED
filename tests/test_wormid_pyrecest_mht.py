"""Adapter invariants; run in the PyRecEst environment on an SSH server."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from deformeot.tracker import DetectionFrame
from deformeot.wormid_pyrecest_mht import (
    PyRecEstMHTConfig,
    _build_mht,
    _components,
    track_pyrecest_mht,
)
from deformeot.wormid_benchmark import ReferenceTrackerConfig


class AnonymousSequence:
    """Any attempt to consume evaluation identities raises immediately."""

    def __init__(self, frames):
        self.frames = tuple(DetectionFrame(points) for points in frames)

    @property
    def truth_indices(self):
        raise AssertionError("tracker read hidden identities")

    @property
    def initial_track_ids(self):
        raise AssertionError("tracker read semantic neuron labels")


@pytest.fixture(autouse=True)
def require_native_pyrecest():
    pytest.importorskip("pyrecest")


@pytest.mark.parametrize("fixed_lag", [0, 2, 20])
@pytest.mark.parametrize("gaussian", [False, True])
def test_native_mht_anonymous_permutations_and_tail_flush(fixed_lag, gaussian):
    base = np.asarray([
        [-4, -2, -1], [-3, 2, 1], [-1, -4, 2], [0, 3, -2],
        [2, -1, 3], [3, 4, 0], [5, -3, -2], [7, 1, 1],
    ], dtype=float)
    orders = [
        np.asarray(order) for order in (
            [3, 0, 7, 2, 5, 1, 4, 6], [6, 1, 5, 2, 7, 0, 3, 4],
            [0, 3, 2, 4, 7, 6, 1, 5], [7, 5, 2, 4, 0, 1, 6, 3],
        )
    ]
    frames = [(base + step * np.asarray([0.15, -0.1, 0.08]))[order]
              for step, order in enumerate(orders)]
    result = track_pyrecest_mht(
        AnonymousSequence(frames),
        config=PyRecEstMHTConfig(fixed_lag=fixed_lag, use_gaussian_cost=gaussian),
    )
    assert result.method == "pyrecest_mht"
    assert len(result.assignments) == len(frames)
    for order, assignment in zip(orders, result.assignments):
        np.testing.assert_array_equal(order[assignment], orders[0])
        np.testing.assert_array_equal(np.sort(assignment), np.arange(len(base)))
    assert result.cloud_motion_um.shape == (len(frames),)


def test_adapter_uses_native_mht_and_native_ranked_assignment_scores():
    from pyrecest.filters.multi_hypothesis_tracker import MultiHypothesisTracker

    tracker = _build_mht(
        np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
        PyRecEstMHTConfig(hypotheses_per_parent=6), ReferenceTrackerConfig(),
    )
    assert isinstance(tracker, MultiHypothesisTracker)
    costs = np.asarray([[2.0, 3.0, 9.0], [4.0, 1.0, 3.0], [8.0, 5.0, 2.0]])
    candidates = [[(column, -float(cost)) for column, cost in enumerate(row)]
                  for row in costs]
    actual = tracker._enumerate_candidate_assignments(candidates, 0.0)
    expected = sorted(
        (float(costs[np.arange(3), assignment].sum()), assignment)
        for assignment in permutations(range(3))
    )
    assert len(actual) == 6
    assert [round(-score, 9) for score, _ in actual] == [score for score, _ in expected]
    assert {assignment for _, assignment in actual} == set(permutations(range(3)))


def test_factorized_native_murty_combines_components_in_global_score_order():
    points = np.column_stack([np.arange(5), np.zeros((5, 2))])
    tracker = _build_mht(
        points, PyRecEstMHTConfig(hypotheses_per_parent=4), ReferenceTrackerConfig(),
    )
    candidates = [
        [(0, -1.0), (1, -2.0)], [(0, -4.0), (1, -1.0)],
        [(2, -3.0)], [(3, -1.0), (4, -2.0)], [(3, -3.0), (4, -1.0)],
    ]
    assert [len(tracks) for tracks, _ in _components(candidates)] == [2, 1, 2]
    actual = tracker._enumerate_candidate_assignments(candidates, 0.0)
    assert [-score for score, _ in actual] == [7.0, 10.0, 11.0, 14.0]
    assert all(sorted(assignment) == list(range(5)) for _, assignment in actual)


def test_candidate_backbone_preserves_exhaustive_feasibility_with_row_cap():
    positions = np.asarray([[0, 0, 0], [0.1, 0, 0], [4, 0, 0]], dtype=float)
    tracker = _build_mht(
        positions, PyRecEstMHTConfig(max_measurements_per_track=1), ReferenceTrackerConfig(),
    )
    measurements = np.asarray([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=float)
    candidates = tracker._build_candidate_measurements(
        tracker.filter_bank, measurements.T,
        np.column_stack([np.eye(3), np.zeros((3, 3))]), 0.09 * np.eye(3),
    )
    ranked = tracker._enumerate_candidate_assignments(candidates, 0.0)
    assert ranked
    assert sorted(ranked[0][1]) == [0, 1, 2]


def test_commit_conditions_native_histories_and_releases_old_snapshots():
    points = np.asarray([[0, 0, 0], [0.1, 0, 0]], dtype=float)
    tracker = _build_mht(
        points, PyRecEstMHTConfig(hypotheses_per_parent=2), ReferenceTrackerConfig(reference_weight=0),
    )
    matrix = np.column_stack([np.eye(3), np.zeros((3, 3))])
    tracker.update_linear(points.T, matrix, 0.09 * np.eye(3))
    assert len(tracker.global_hypotheses) == 2
    emitted = tracker.commit_oldest()
    assert sorted(emitted) == [0, 1]
    assert len(tracker.global_hypotheses) == 1
    assert tracker.global_hypothesis_histories == [[]]
    assert tracker.global_filter_bank_histories == [[]]


def test_empty_and_changing_cohorts_raise():
    with pytest.raises(ValueError, match="empty"):
        track_pyrecest_mht(AnonymousSequence([]))
    with pytest.raises(ValueError, match="fixed exhaustive"):
        track_pyrecest_mht(AnonymousSequence([
            np.zeros((3, 3)), np.zeros((2, 3)),
        ]))


@pytest.mark.parametrize("overrides", [
    {"fixed_lag": -1}, {"fixed_lag": True}, {"hypotheses_per_parent": 0},
    {"measurement_sigma_um": np.nan}, {"reference_cost_weight": -1},
    {"velocity_decay": 1.1}, {"use_gaussian_cost": "yes"},
])
def test_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        replace(PyRecEstMHTConfig(), **overrides)
