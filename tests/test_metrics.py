import numpy as np
import pandas as pd

from deformeot.metrics import evaluate_partial_tracks, mutual_nearest_matches


def _tracks(rows):
    return pd.DataFrame(rows, columns=["TrackID", "ObjectID", "t", "z", "y", "x"])


def test_mutual_nearest_matches_rejects_nonmutual_candidate():
    gt = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    pred = np.array([[0.0, 0.1, 0.0], [0.0, 0.2, 0.0]])
    gi, pi, distance = mutual_nearest_matches(gt, pred, threshold=5.0)
    assert gi.tolist() == [0]
    assert pi.tolist() == [0]
    assert np.allclose(distance, [0.1])


def test_perfect_partial_tracks_score_one_despite_unlabeled_prediction():
    gt = _tracks(
        [
            ("a", "g0", 0, 0, 0, 0),
            ("b", "g1", 0, 0, 10, 0),
            ("a", "g2", 1, 0, 1, 0),
            ("b", "g3", 1, 0, 11, 0),
        ]
    )
    pred = _tracks(
        [
            ("10", "p0", 0, 0, 0, 0),
            ("11", "p1", 0, 0, 10, 0),
            ("extra", "p2", 0, 0, 30, 0),
            ("10", "p3", 1, 0, 1, 0),
            ("11", "p4", 1, 0, 11, 0),
            ("extra", "p5", 1, 0, 31, 0),
        ]
    )
    result, matches = evaluate_partial_tracks(
        gt, pred, spacing_zyx=(1, 1, 1), threshold_um=2
    )
    assert len(matches) == 4
    assert result.modified_mota == 1
    assert result.partial_idf1 == 1
    assert result.misses == 0
    assert result.mismatches == 0


def test_identity_swap_counts_mismatches_and_reduces_idf1():
    gt = _tracks(
        [
            ("a", "g0", 0, 0, 0, 0),
            ("b", "g1", 0, 0, 10, 0),
            ("a", "g2", 1, 0, 1, 0),
            ("b", "g3", 1, 0, 11, 0),
        ]
    )
    pred = _tracks(
        [
            ("10", "p0", 0, 0, 0, 0),
            ("11", "p1", 0, 0, 10, 0),
            ("11", "p2", 1, 0, 1, 0),
            ("10", "p3", 1, 0, 11, 0),
        ]
    )
    result, _ = evaluate_partial_tracks(gt, pred, spacing_zyx=(1, 1, 1), threshold_um=2)
    assert result.mismatches == 2
    assert result.modified_mota == 0.5
    assert result.partial_idf1 == 0.5

