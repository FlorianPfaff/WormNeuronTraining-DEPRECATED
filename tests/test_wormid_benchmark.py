from __future__ import annotations

import numpy as np

from deformeot.tracker import DetectionFrame
from deformeot.wormid import IdentityBlindSequence
from deformeot.wormid_benchmark import (
    METHODS,
    ReferenceTrackerConfig,
    TrackResult,
    distance_profile_descriptor,
    score_tracks,
    track_sequence,
)


def _sequence(point_frames, permutations):
    frames = []
    truth = []
    for time, (points, order) in enumerate(zip(point_frames, permutations)):
        order = np.asarray(order, dtype=int)
        frames.append(DetectionFrame(np.asarray(points)[order], time=time))
        truth.append(order)
    first = np.asarray(permutations[0], dtype=int)
    return IdentityBlindSequence(
        frames=tuple(frames),
        truth_indices=tuple(truth),
        initial_track_ids=tuple(f"n{value}" for value in first),
        source_frame_indices=tuple(range(len(frames))),
    )


def test_distance_profile_is_permutation_equivariant_and_similarity_invariant():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(12, 3))
    order = rng.permutation(len(points))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transformed = 2.7 * points[order] @ rotation + np.asarray([8.0, -2.0, 1.0])
    original = distance_profile_descriptor(points, quantiles=12)
    moved = distance_profile_descriptor(transformed, quantiles=12)
    assert np.allclose(moved, original[order], atol=1e-10)


def test_all_methods_ignore_detection_row_order_on_smooth_motion():
    rng = np.random.default_rng(8)
    base = rng.normal(size=(18, 3)) * np.asarray([3.0, 2.0, 1.0])
    frames = [
        base,
        base @ np.asarray([[1.0, 0.05, 0.0], [0.0, 1.0, 0.03], [0.0, 0.0, 1.0]])
        + [0.3, -0.2, 0.1],
        base + [0.6, -0.4, 0.2],
    ]
    permutations = [rng.permutation(len(base)) for _ in frames]
    sequence = _sequence(frames, permutations)
    config = ReferenceTrackerConfig(descriptor_weight=0.5)
    for method in METHODS:
        result = track_sequence(sequence, method, config=config)
        metrics = score_tracks(sequence, result)
        assert metrics.identity_accuracy == 1.0
        assert metrics.identity_switches == 0


def test_shape_anchor_recovers_after_a_temporal_swap():
    # Two nearby points cross between frames, but their distance-to-cloud
    # profiles remain different because of asymmetric landmarks.
    base = np.asarray(
        [
            [-3.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.2, 2.0, 0.0],
            [4.0, -1.0, 0.0],
            [7.0, 0.5, 0.0],
        ]
    )
    middle = base.copy()
    middle[1, 0], middle[2, 0] = 0.1, -0.1
    final = base + [0.2, 0.0, 0.0]
    sequence = _sequence([base, middle, final], [np.arange(6), np.arange(6), np.arange(6)])
    result = track_sequence(
        sequence,
        "dual_anchor",
        config=ReferenceTrackerConfig(
            reference_weight=2.0,
            descriptor_weight=4.0,
            descriptor_quantiles=8,
        ),
    )
    assert np.array_equal(result.assignments[-1], np.arange(6))


def test_severe_stratum_has_exact_prevalence_despite_quantized_ties():
    base = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 1.0, 0.0], [9.0, 0.0, 0.0]]
    )
    sequence = _sequence([base] * 9, [np.arange(4)] * 9)
    assignments = tuple(np.arange(4) for _ in range(9))
    result = TrackResult(
        method="ties",
        assignments=assignments,
        runtime_seconds=0.0,
        cloud_motion_um=np.asarray([0.0] + [0.25] * 8),
    )
    metrics = score_tracks(sequence, result, severe_quantile=0.75)
    assert metrics.severe_frames == 2
    assert metrics.severe_total == 8
    assert metrics.severe_ties_at_boundary == 8
    assert metrics.severe_selection_mode.startswith("exact_top_fraction")
