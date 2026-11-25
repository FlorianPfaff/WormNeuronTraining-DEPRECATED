from pathlib import Path

import numpy as np
import pytest

from deformeot.fdnc import (
    FDNCFrame,
    _robust_affine_warp,
    evaluate_partial_identity_tracks,
    load_fdnc_sequence,
    track_fdnc_sequence,
)


def _frame(frame_id, points, labels):
    return FDNCFrame(
        frame_id=frame_id,
        points_xyz=np.asarray(points, dtype=float),
        labels=np.asarray(labels, dtype=np.int64),
        path=Path(f"real_{frame_id}.npy"),
    )


def test_loader_sorts_numeric_ids_and_preserves_zero_label(tmp_path):
    np.save(tmp_path / "real_10.npy", np.array([[1.0, 2.0, 3.0, -1.0]]))
    np.save(tmp_path / "real_2.npy", np.array([[4.0, 5.0, 6.0, 0.0]]))
    frames = load_fdnc_sequence(tmp_path)
    assert [frame.frame_id for frame in frames] == [2, 10]
    assert frames[0].labels.tolist() == [0]


def test_loader_rejects_duplicate_manual_identity(tmp_path):
    np.save(
        tmp_path / "real_1.npy",
        np.array([[0.0, 0.0, 0.0, 4.0], [1.0, 0.0, 0.0, 4.0]]),
    )
    with pytest.raises(ValueError, match="repeats a manual identity"):
        load_fdnc_sequence(tmp_path)


def test_perfect_sequential_translation_scores_one_with_partial_labels():
    frames = [
        _frame(1, [[0, 0, 0], [0, 2, 0], [0, 5, 0]], [0, 1, -1]),
        _frame(2, [[0.1, 2, 0], [0.1, 5, 0], [0.1, 0, 0]], [1, -1, 0]),
        _frame(3, [[0.2, 5, 0], [0.2, 0, 0], [0.2, 2, 0]], [-1, 0, 1]),
    ]
    tracks = track_fdnc_sequence(frames, method="hungarian", max_distance=0.5)
    metrics = evaluate_partial_identity_tracks(frames, tracks)
    assert metrics.true_adjacent_links == 4
    assert metrics.correct_adjacent_links == 4
    assert metrics.link_f1 == 1.0
    assert metrics.partial_idf1 == 1.0
    assert metrics.gt_identities == 2


def test_global_idf1_detects_temporal_identity_swap():
    frames = [
        _frame(1, [[0, 0, 0], [0, 1, 0]], [0, 1]),
        _frame(2, [[0, 0.1, 0], [0, 0.9, 0]], [1, 0]),
    ]
    tracks = track_fdnc_sequence(frames, method="hungarian", max_distance=0.5)
    metrics = evaluate_partial_identity_tracks(frames, tracks)
    assert metrics.correct_adjacent_links == 0
    assert metrics.wrong_labeled_adjacent_links == 2
    assert metrics.link_f1 == 0.0
    assert metrics.partial_idf1 == 0.5


def test_tracker_is_invariant_to_evaluation_labels():
    points = [
        [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
        [[0.1, 2, 0], [0.1, 0, 0], [0.1, 1, 0]],
    ]
    labeled = [_frame(1, points[0], [0, 1, -1]), _frame(2, points[1], [2, 0, 1])]
    relabeled = [_frame(1, points[0], [9, -1, 4]), _frame(2, points[1], [-1, 7, 3])]
    labeled_tracks = track_fdnc_sequence(labeled, method="affine_hungarian", max_distance=0.5)
    relabeled_tracks = track_fdnc_sequence(relabeled, method="affine_hungarian", max_distance=0.5)
    for left, right in zip(labeled_tracks.track_ids, relabeled_tracks.track_ids):
        assert np.array_equal(left, right)
    for left, right in zip(labeled_tracks.parent_indices, relabeled_tracks.parent_indices):
        assert np.array_equal(left, right)


def test_robust_affine_icp_reduces_unlabelled_cloud_alignment_error():
    rng = np.random.default_rng(12)
    source = rng.normal(size=(50, 3))
    affine = np.array([[1.02, 0.08, 0.00], [-0.05, 0.97, 0.04], [0.01, 0.02, 1.03]])
    destination = source @ affine + np.array([0.20, -0.12, 0.08])
    before = np.mean(np.min(np.linalg.norm(source[:, None] - destination[None], axis=2), axis=1))
    warped = _robust_affine_warp(
        source,
        destination,
        iterations=5,
        trim_fraction=0.8,
        ridge=0.01,
    )
    after = np.mean(np.min(np.linalg.norm(warped[:, None] - destination[None], axis=2), axis=1))
    assert after < 0.25 * before
