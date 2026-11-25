import itertools

import numpy as np

from deformeot.tracker import (
    DetectionFrame,
    DeformEOTTracker,
    TrackerConfig,
    estimate_shared_affine,
    k_best_assignments,
)


def test_k_best_assignments_are_exactly_cost_ordered():
    cost = np.array([[0.2, 1.1, 2.0], [0.5, 0.1, 1.7]])
    actual = k_best_assignments(cost, 5)
    brute_force = sorted(
        (
            sum(cost[row, column] for row, column in enumerate(columns)),
            columns,
        )
        for columns in itertools.permutations(range(cost.shape[1]), cost.shape[0])
    )[:5]
    assert np.allclose([value for _, value in actual], [value for value, _ in brute_force])
    assert {tuple(assignment) for assignment, _ in actual} == {
        tuple(columns) for _, columns in brute_force
    }


def test_robust_shared_affine_recovers_moderate_deformation_with_clutter():
    source = np.array(
        [
            [-2.0, -1.0, -0.5],
            [-1.5, 1.0, 0.2],
            [-0.2, -1.2, 0.9],
            [0.3, 0.7, -0.8],
            [1.2, -0.5, 0.4],
            [1.8, 1.1, 1.0],
            [0.7, 1.8, -0.2],
            [-1.0, 0.2, 1.6],
        ]
    )
    matrix = np.array([[1.05, 0.10, 0.0], [-0.05, 0.93, 0.08], [0.02, 0.0, 1.08]])
    offset = np.array([0.35, -0.2, 0.15])
    target = source @ matrix.T + offset
    target = np.vstack([target[[4, 1, 6, 0, 7, 3, 5, 2]], [8.0, -7.0, 6.0]])
    transform = estimate_shared_affine(source, target, ridge=0.05, iterations=8)
    error = np.linalg.norm(transform.apply(source) - (source @ matrix.T + offset), axis=1)
    assert np.median(error) < 0.12
    assert transform.matched_pairs == len(source)


def _deforming_occlusion_sequence():
    initial = np.array(
        [
            [-2.0, -1.0, -0.5],
            [-1.5, 1.0, 0.2],
            [-0.2, -1.2, 0.9],
            [0.3, 0.7, -0.8],
            [1.2, -0.5, 0.4],
            [1.8, 1.1, 1.0],
        ]
    )
    frames = [DetectionFrame(initial, object_ids=tuple(f"n{i}" for i in range(6)))]
    current = initial.copy()
    for time in range(1, 7):
        affine = np.array(
            [[1.0, 0.025, 0.0], [-0.015, 1.0, 0.018], [0.0, -0.01, 1.0]]
        )
        current = current @ affine.T + np.array([0.28, -0.12, 0.07])
        keep = [index for index in range(6) if not (index == 2 and time in {2, 3})]
        order = list(reversed(keep)) if time % 2 else keep[2:] + keep[:2]
        frames.append(
            DetectionFrame(
                current[order],
                object_ids=tuple(f"n{index}" for index in order),
                time=time,
            )
        )
    return frames


def test_tracker_preserves_identity_through_deformation_and_two_frame_occlusion():
    frames = _deforming_occlusion_sequence()
    tracker = DeformEOTTracker(
        TrackerConfig(
            max_hypotheses=10,
            assignments_per_hypothesis=5,
            fixed_lag=2,
            measurement_sigma=0.25,
            process_sigma=0.18,
            initial_position_sigma=0.15,
            gate_mahalanobis_sq=36,
            affine_ridge=0.08,
        )
    )
    result = tracker.track(frames, initial_track_ids=tuple(f"n{i}" for i in range(6)))

    for frame in result.frames:
        for track_index, track_id in enumerate(result.track_ids):
            detection_index = int(frame.map_assignment[track_index])
            if track_id == "n2" and frame.frame_index in {2, 3}:
                assert detection_index == -1
            else:
                assert frame.object_ids[detection_index] == track_id
    assert result.frames[2].visibility_probability[2] < 0.8
    assert result.frames[3].visibility_probability[2] < result.frames[2].visibility_probability[2]
    assert result.frames[4].visibility_probability[2] > 0.99
    exported = result.to_dataframe(include_misses=True)
    assert set(
        [
            "TrackID",
            "ObjectID",
            "t",
            "z",
            "y",
            "x",
            "association_probability",
            "association_entropy",
            "visibility_probability",
        ]
    ).issubset(exported.columns)
    assert len(exported) == 7 * 6


def test_extent_likelihood_disambiguates_a_crossing():
    small = np.diag([0.10, 0.16, 0.23])
    large = np.diag([0.65, 0.22, 0.14])
    anchors = np.array(
        [[-3.0, -2.0, 0.0], [-3.0, 2.0, 0.0], [3.0, -2.0, 0.0], [3.0, 2.0, 0.0]]
    )
    anchor_extents = [np.diag([0.2, 0.2, 0.2]) for _ in range(4)]
    positions = [
        np.vstack([anchors, [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.vstack([anchors + [0.15, 0.0, 0.0], [0.08, 0.0, 0.0], [-0.08, 0.0, 0.0]]),
        np.vstack([anchors + [0.30, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    ]
    # The two crossing detections are deliberately reversed at frames one and two.
    orders = [list(range(6)), [3, 1, 5, 0, 4, 2], [5, 0, 2, 4, 1, 3]]
    canonical_ids = ["a0", "a1", "a2", "a3", "small", "large"]
    base_extents = np.stack([*anchor_extents, small, large])
    frames = []
    for time, (frame_positions, order) in enumerate(zip(positions, orders)):
        frames.append(
            DetectionFrame(
                frame_positions[order],
                extents=base_extents[order],
                object_ids=tuple(canonical_ids[index] for index in order),
                time=time,
            )
        )
    tracker = DeformEOTTracker(
        TrackerConfig(
            max_hypotheses=12,
            assignments_per_hypothesis=8,
            fixed_lag=1,
            measurement_sigma=0.3,
            process_sigma=0.25,
            extent_weight=3.0,
            extent_log_sigma=0.35,
            gate_mahalanobis_sq=64,
        )
    )
    result = tracker.track(frames, initial_track_ids=canonical_ids)
    for frame in result.frames[1:]:
        for index, track_id in enumerate(canonical_ids):
            assert frame.object_ids[int(frame.map_assignment[index])] == track_id
    assert result.frames[1].association_confidence[4] > 0.95
    assert result.frames[1].association_confidence[5] > 0.95


def test_extent_state_updates_and_remains_positive_definite():
    initial_extent = np.diag([0.10, 0.20, 0.30])
    observed_extent = np.diag([0.25, 0.35, 0.45])
    frames = [
        DetectionFrame([[0.0, 0.0, 0.0]], extents=[initial_extent]),
        DetectionFrame([[0.1, 0.0, 0.0]], extents=[observed_extent]),
    ]
    result = DeformEOTTracker(
        TrackerConfig(extent_update_gain=0.5, extent_process_sigma=0.01)
    ).track(frames)
    updated = result.frames[1].posterior_extents[0]
    assert updated is not None
    assert not np.allclose(updated, initial_extent)
    assert np.all(np.linalg.eigvalsh(updated) > 0)


def test_beam_exposes_entropy_for_exactly_ambiguous_detections():
    anchors = np.array(
        [[-2.0, -2.0, 0.0], [-2.0, 2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0]]
    )
    first = DetectionFrame(np.vstack([anchors, [-0.5, 0, 0], [0.5, 0, 0]]))
    second = DetectionFrame(np.vstack([anchors, [0.0, 0, 0], [0.0, 0, 0]]))
    tracker = DeformEOTTracker(
        TrackerConfig(
            max_hypotheses=16,
            assignments_per_hypothesis=12,
            fixed_lag=0,
            measurement_sigma=0.35,
            process_sigma=0.2,
            gate_mahalanobis_sq=64,
        )
    )
    result = tracker.track([first, second])
    posterior = result.frames[1]
    assert posterior.hypothesis_count > 1
    assert posterior.association_entropy[4] > 0.6
    assert posterior.association_entropy[5] > 0.6
    assert np.isclose(posterior.association_probabilities[4, 4], 0.5, atol=0.08)
    assert np.isclose(posterior.association_probabilities[4, 5], 0.5, atol=0.08)
    assert "not empirically calibrated" in result.calibration_status
