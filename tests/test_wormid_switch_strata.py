import numpy as np

from deformeot.tracker import DetectionFrame
from deformeot.wormid import IdentityBlindSequence
from deformeot.wormid_benchmark import TrackResult
from deformeot.wormid_switch_strata import (
    benchmark_frame_indices,
    exact_high_motion_mask,
    score_switch_stratum,
)


def _sequence() -> IdentityBlindSequence:
    frames = tuple(
        DetectionFrame(positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        for _ in range(5)
    )
    return IdentityBlindSequence(
        frames=frames,
        truth_indices=(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
        ),
        initial_track_ids=("track-0", "track-1"),
        source_frame_indices=(0, 1, 2, 3, 4),
    )


def test_exact_high_motion_mask_uses_count_and_earlier_tie_break() -> None:
    result = TrackResult(
        method="dual_anchor",
        assignments=tuple(np.arange(2) for _ in range(5)),
        runtime_seconds=0.0,
        cloud_motion_um=np.array([0.0, 1.0, 3.0, 3.0, 2.0]),
    )
    mask = exact_high_motion_mask(result, (0, 1, 2, 3, 4), fraction=0.25)
    np.testing.assert_array_equal(mask, [False, True, False, False])


def test_score_switch_stratum_counts_only_selected_transition() -> None:
    sequence = _sequence()
    result = TrackResult(
        method="dual_anchor",
        assignments=tuple(np.arange(2) for _ in range(5)),
        runtime_seconds=0.0,
        cloud_motion_um=np.arange(5, dtype=float),
    )
    score = score_switch_stratum(sequence, result, np.array([False, True, False, False]))
    assert score["switches"] == 2
    assert score["opportunities"] == 2
    assert score["switch_rate"] == 1.0


def test_benchmark_frame_indices_appends_final_frame() -> None:
    assert benchmark_frame_indices(10, 4) == (0, 4, 8, 9)
