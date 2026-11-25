from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyrecest", reason="optional PyRecEst comparison environment")

from deformeot.tracker import DetectionFrame
from deformeot.wormid_benchmark import ReferenceTrackerConfig, track_sequence
from deformeot.wormid_pyrecest_gnn import (
    PyRecEstGNNConfig,
    _mahalanobis_cost,
    track_pyrecest_gnn,
)


def _anonymous_sequence(seed=8, count=24, frame_count=8):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(count, 3)) * [6.0, 3.0, 2.0]
    point_frames = [
        base @ np.asarray([[1.0, 0.02 * step, 0], [0, 1.0, 0.01 * step], [0, 0, 1.0]])
        + np.asarray([0.2, -0.1, 0.05]) * step
        + rng.normal(scale=0.03, size=base.shape)
        for step in range(frame_count)
    ]
    orders = [rng.permutation(count) for _ in point_frames]
    # There is deliberately no truth_indices, initial_track_ids, or source
    # metadata. A tracker accidentally consulting those fields must fail.
    sequence = SimpleNamespace(
        frames=tuple(DetectionFrame(points[order]) for points, order in zip(point_frames, orders))
    )
    return sequence, orders


@pytest.mark.parametrize("descriptor_weight", [0.0, 0.4])
def test_pyrecest_control_reproduces_frozen_dual_objective(descriptor_weight):
    sequence, _ = _anonymous_sequence()
    geometry = ReferenceTrackerConfig(descriptor_weight=descriptor_weight)
    baseline = track_sequence(sequence, "dual_anchor", config=geometry)
    control = track_pyrecest_gnn(
        sequence, config=PyRecEstGNNConfig(mode="equivalence"), geometry=geometry
    )
    for expected, actual in zip(baseline.assignments, control.assignments):
        assert np.array_equal(actual, expected)
    assert np.array_equal(control.cloud_motion_um, baseline.cloud_motion_um)


@pytest.mark.parametrize("mode", ["equivalence", "kalman"])
def test_gnn_preserves_ids_under_row_permutation_and_shared_affine_motion(mode):
    sequence, orders = _anonymous_sequence(seed=10)
    result = track_pyrecest_gnn(sequence, config=PyRecEstGNNConfig(mode=mode))
    for order, assignment in zip(orders, result.assignments):
        assert np.array_equal(order[assignment], orders[0])


def test_adapter_calls_pyrecest_update_and_predict(monkeypatch):
    from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor

    calls = {"update": 0, "predict": 0}
    original_update = GlobalNearestNeighbor.update_linear
    original_predict = GlobalNearestNeighbor.predict_linear

    def update(self, *args, **kwargs):
        calls["update"] += 1
        return original_update(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        calls["predict"] += 1
        return original_predict(self, *args, **kwargs)

    monkeypatch.setattr(GlobalNearestNeighbor, "update_linear", update)
    monkeypatch.setattr(GlobalNearestNeighbor, "predict_linear", predict)
    sequence, _ = _anonymous_sequence(frame_count=5)
    track_pyrecest_gnn(
        sequence,
        config=PyRecEstGNNConfig(local_velocity_weight=0.4, adaptive_process_noise=0.1),
    )
    assert calls == {"update": 4, "predict": 4}


def test_zero_residual_mahalanobis_floor_preserves_physical_cost_scale():
    from pyrecest.distributions import GaussianDistribution
    from scipy.spatial.distance import cdist

    points = np.asarray([[0, 0, 0], [1, 2, 0], [3, 0, 1]], dtype=float)
    tracker = SimpleNamespace(
        filter_state=[GaussianDistribution(point, np.eye(3) * 4) for point in points]
    )
    costs, covariance_scale = _mahalanobis_cost(tracker, points, np.eye(3) * 5)
    assert covariance_scale == pytest.approx(3)
    assert np.median(np.diag(costs)) == 0
    # With isotropic total variance nine, dividing by floor_um/sqrt(9)
    # reproduces the original Euclidean distance divided by floor_um.
    assert np.allclose(costs / (0.1 / covariance_scale), cdist(points, points) / 0.1)


def test_gnn_rejects_nonexhaustive_input():
    sequence, _ = _anonymous_sequence(frame_count=2)
    sequence.frames = (
        sequence.frames[0], DetectionFrame(sequence.frames[1].positions[:-1])
    )
    with pytest.raises(ValueError, match="fixed exhaustive"):
        track_pyrecest_gnn(sequence)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "unknown"},
        {"measurement_noise_um": 0},
        {"process_noise_um": float("nan")},
        {"velocity_decay": 1},
        {"adaptive_process_noise": -0.1},
    ],
)
def test_gnn_configuration_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        PyRecEstGNNConfig(**kwargs)
