from __future__ import annotations

import h5py
import numpy as np

from deformeot.downstream import (
    RawFluorescence,
    anonymous_detection_fluorescence,
    complete_recording_paired_bootstrap,
    evaluate_wormid_assignments,
    load_wormid_raw_fluorescence,
    reconstruct_track_traces,
    trace_correlation_metrics,
)
from deformeot.tracker import DetectionFrame
from deformeot.wormid import IdentityBlindSequence


def _sequence(permutations: list[np.ndarray]) -> IdentityBlindSequence:
    frames = tuple(
        DetectionFrame(np.column_stack([order, order, order]), time=0.25 * index)
        for index, order in enumerate(permutations)
    )
    return IdentityBlindSequence(
        frames=frames,
        truth_indices=tuple(permutations),
        initial_track_ids=tuple(f"n{value}" for value in permutations[0]),
        source_frame_indices=tuple(range(len(permutations))),
    )


def _correct_assignments(sequence: IdentityBlindSequence) -> tuple[np.ndarray, ...]:
    canonical = sequence.truth_indices[0]
    return tuple(
        np.asarray(
            [int(np.flatnonzero(order == identity)[0]) for identity in canonical],
            dtype=int,
        )
        for order in sequence.truth_indices
    )


def test_correct_tracking_exactly_recovers_anonymous_detection_traces():
    rng = np.random.default_rng(12)
    frames = 40
    neurons = 4
    permutations = [rng.permutation(neurons) for _ in range(frames)]
    sequence = _sequence(permutations)
    time = np.arange(frames, dtype=float) / 4
    values = np.column_stack(
        [
            np.sin(time),
            np.cos(0.7 * time),
            (np.arange(frames) % 7 == 0).astype(float),
            np.linspace(-1, 2, frames) ** 2,
        ]
    )
    raw = RawFluorescence("synthetic", values, time, tuple(range(neurons)), 1.0, 0.0)
    assignments = _correct_assignments(sequence)
    anonymous = anonymous_detection_fluorescence(raw, sequence)
    reconstructed = reconstruct_track_traces(anonymous, assignments)
    expected = values[:, sequence.truth_indices[0]].T
    assert np.array_equal(reconstructed, expected)
    metrics = evaluate_wormid_assignments(raw, sequence, assignments)
    assert metrics["raw"]["pearson_median"] == 1.0
    assert metrics["raw"]["spearman_median"] == 1.0
    assert metrics["source_contamination"]["foreign_source_fraction"] == 0.0


def test_swaps_degrade_trace_fidelity_and_create_measured_contamination():
    rng = np.random.default_rng(3)
    frames = 80
    permutations = [rng.permutation(3) for _ in range(frames)]
    sequence = _sequence(permutations)
    time = np.arange(frames, dtype=float) / 4
    values = np.column_stack(
        [
            np.sin(0.3 * np.arange(frames)),
            np.sign(np.sin(0.71 * np.arange(frames) + 0.4)),
            rng.normal(size=frames),
        ]
    )
    raw = RawFluorescence("synthetic", values, time, (0, 1, 2), 1.0, 0.0)
    correct = list(_correct_assignments(sequence))
    swapped = [assignment.copy() for assignment in correct]
    for assignment in swapped[10:70]:
        assignment[0], assignment[1] = assignment[1], assignment[0]
    correct_metrics = evaluate_wormid_assignments(raw, sequence, correct)
    swapped_metrics = evaluate_wormid_assignments(raw, sequence, swapped)
    assert swapped_metrics["raw"]["pearson_median"] < 0.8
    assert swapped_metrics["first_difference_per_second"]["pearson_median"] < 0.8
    assert swapped_metrics["source_contamination"]["foreign_source_fraction"] == 0.5
    assert correct_metrics["raw"]["pearson_median"] == 1.0


def test_nan_and_constant_tracks_are_excluded_and_reported_without_imputation():
    truth = np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    predicted = np.asarray([[1.0, np.nan, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    result = trace_correlation_metrics(predicted, truth)
    assert result["tracks_valid"] == 1
    assert result["pairwise_finite_samples"] == 7
    assert result["pearson_median"] == 1.0
    assert result["exclusions"]["constant_ground_truth"] == 1
    assert result["per_track"][1]["pearson"] is None
    assert result["nan_policy"] == "pairwise finite; no imputation"


def test_raw_loader_applies_conversion_and_reorders_roi_columns(tmp_path):
    path = tmp_path / "fluorescence.nwb"
    with h5py.File(path, "w") as nwb:
        nwb.create_dataset(
            "processing/CalciumActivity/NeuronIDs/labels",
            data=np.asarray(["A", "B", "C"], dtype=h5py.string_dtype()),
        )
        group = nwb.create_group(
            "processing/CalciumActivity/SignalRawFluor/SignalCalciumImResponseSeries"
        )
        data = group.create_dataset("data", data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        data.attrs["conversion"] = 2.0
        data.attrs["offset"] = -1.0
        group.create_dataset("rois", data=[2, 0, 1])
        start = group.create_dataset("starting_time", data=3.0)
        start.attrs["rate"] = 4.0
    raw = load_wormid_raw_fluorescence(path)
    assert np.array_equal(raw.values, [[3.0, 5.0, 1.0], [9.0, 11.0, 7.0]])
    assert np.array_equal(raw.times_seconds, [3.0, 3.25])


def test_bootstrap_pairs_complete_recordings_within_stride():
    rows = []
    for recording, baseline, improved in (
        ("a", 0.1, 0.4),
        ("b", 0.2, 0.4),
        ("c", 0.3, 0.4),
    ):
        rows.extend(
            [
                {
                    "recording_id": recording,
                    "method": "base",
                    "stride": 5,
                    "raw": {"pearson_median": baseline},
                },
                {
                    "recording_id": recording,
                    "method": "new",
                    "stride": 5,
                    "raw": {"pearson_median": improved},
                },
            ]
        )
    result = complete_recording_paired_bootstrap(
        rows,
        improved="new",
        baseline="base",
        metric="raw.pearson_median",
        strata=("stride",),
        samples=1000,
        seed=4,
    )[0]
    assert result["complete_recordings"] == 3
    assert np.isclose(result["mean_paired_difference"], 0.2)
    assert result["resampling_unit"] == "complete recording"
