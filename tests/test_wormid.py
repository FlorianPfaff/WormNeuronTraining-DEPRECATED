from __future__ import annotations

import h5py
import numpy as np

from deformeot.wormid import (
    identity_accuracy,
    identity_blind_sequence,
    load_wormid_recording,
)


def _write_nwb(path):
    labels = np.asarray(["A", "B", "C"], dtype=h5py.string_dtype())
    with h5py.File(path, "w") as nwb:
        nwb.create_dataset("processing/CalciumActivity/NeuronIDs/labels", data=labels)
        nwb.create_dataset(
            "general/optophysiology/CalciumImVol/grid_spacing",
            data=np.asarray([0.2, 0.3, 1.5]),
        )
        start = nwb.create_dataset("acquisition/CalciumImageSeries/starting_time", data=2.0)
        start.attrs["rate"] = 4.0
        base = nwb.create_group(
            "processing/CalciumActivity/CalciumSeriesSegmentation"
        )
        dtype = np.dtype(
            [("x", "<u4"), ("y", "<u4"), ("z", "<u4"), ("weight", "<f4")]
        )
        for time in range(3):
            group = base.create_group(f"Seg_tpoint_{time}")
            group.create_dataset("id", data=np.arange(3))
            mask = np.zeros(3, dtype=dtype)
            mask["x"] = np.asarray([1, 4, 7]) + time
            mask["y"] = [2, 5, 8]
            mask["z"] = [3, 6, 9]
            mask["weight"] = 1
            group.create_dataset("voxel_mask", data=mask)
            group.create_dataset("voxel_mask_index", data=np.arange(1, 4))


def test_load_wormid_recording_converts_xyz_voxels_to_zyx_micrometres(tmp_path):
    path = tmp_path / "sample.nwb"
    _write_nwb(path)
    recording = load_wormid_recording(path)
    assert recording.frames == 3
    assert recording.neurons == 3
    assert recording.labels == ("A", "B", "C")
    assert np.allclose(recording.times_seconds, [2.0, 2.25, 2.5])
    assert np.allclose(recording.positions_zyx_um[0, 0], [4.5, 0.6, 0.2])
    assert recording.spacing_zyx_um == (1.5, 0.3, 0.2)


def test_identity_blind_sequence_is_permuted_but_scores_perfect_truth(tmp_path):
    path = tmp_path / "sample.nwb"
    _write_nwb(path)
    recording = load_wormid_recording(path)
    sequence = identity_blind_sequence(recording, seed=8)
    assignments = []
    canonical = sequence.truth_indices[0]
    for hidden_order in sequence.truth_indices:
        inverse = {int(identity): row for row, identity in enumerate(hidden_order)}
        assignments.append(np.asarray([inverse[int(identity)] for identity in canonical]))
    score = identity_accuracy(sequence, assignments)
    assert score == {"correct": 9, "total": 9, "misses": 0, "accuracy": 1.0}
    assert all(frame.object_ids == ("d0", "d1", "d2") for frame in sequence.frames)


def test_loader_rejects_dense_or_ragged_voxel_masks(tmp_path):
    path = tmp_path / "sample.nwb"
    _write_nwb(path)
    with h5py.File(path, "a") as nwb:
        group = nwb[
            "processing/CalciumActivity/CalciumSeriesSegmentation/Seg_tpoint_1"
        ]
        del group["voxel_mask_index"]
        group.create_dataset("voxel_mask_index", data=[1, 3, 4])
    try:
        load_wormid_recording(path)
    except ValueError as error:
        assert "one-centre-per-ROI" in str(error)
    else:  # pragma: no cover
        raise AssertionError("ragged representation should have failed")
