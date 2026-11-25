from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np

from deformeot.wormid_dandi import (
    _direct_content_url,
    _extract_coordinate_arrays,
    load_coordinate_cache,
    semantic_coordinate_sha256,
)


def _write_synthetic_nwb(path):
    labels = np.asarray(["A", "", "C"], dtype=h5py.string_dtype())
    with h5py.File(path, "w") as nwb:
        nwb.create_dataset("processing/CalciumActivity/NeuronIDs/labels", data=labels)
        nwb.create_dataset(
            "general/optophysiology/CalciumImVol/grid_spacing",
            data=np.asarray([0.2, 0.3, 1.5]),
        )
        start = nwb.create_dataset(
            "acquisition/CalciumImageSeries/starting_time", data=2.0
        )
        start.attrs["rate"] = 4.0
        segmentation = nwb.create_group(
            "processing/CalciumActivity/CalciumSeriesSegmentation"
        )
        dtype = np.dtype(
            [("x", "<u4"), ("y", "<u4"), ("z", "<u4"), ("weight", "<f4")]
        )
        for time in range(3):
            group = segmentation.create_group(f"Seg_tpoint_{time}")
            group.create_dataset("id", data=np.arange(3))
            mask = np.zeros(3, dtype=dtype)
            mask["x"] = np.asarray([1, 4, 7]) + time
            mask["y"] = [2, 5, 8]
            mask["z"] = [3, 6, 9]
            mask["weight"] = 1
            group.create_dataset("voxel_mask", data=mask)
            group.create_dataset("voxel_mask_index", data=np.arange(1, 4))


def test_direct_content_url_avoids_expiring_api_redirect():
    assert (
        _direct_content_url(
            [
                "https://api.dandiarchive.org/api/assets/id/download/",
                "https://dandiarchive.s3.amazonaws.com/blobs/abc/def/blob",
            ]
        )
        == "https://dandiarchive.s3.amazonaws.com/blobs/abc/def/blob"
    )


def test_coordinate_extraction_retains_rows_but_not_semantic_identity(tmp_path):
    source = tmp_path / "synthetic.nwb"
    _write_synthetic_nwb(source)
    with h5py.File(source, "r") as nwb:
        arrays = _extract_coordinate_arrays(nwb)
    assert np.asarray(arrays["positions_zyx_um"]).shape == (3, 3, 3)
    assert np.allclose(np.asarray(arrays["positions_zyx_um"])[0, 0], [4.5, 0.6, 0.2])
    assert arrays["semantic_label_count"] == 3
    assert arrays["semantic_blank_count"] == 1

    metadata = {
        "cache_schema_version": 1,
        "recording_id": "synthetic",
        "source_content_url": "https://example.invalid/synthetic.nwb",
        "frames": 3,
        "neurons": 3,
        "semantic_coordinate_sha256": semantic_coordinate_sha256(
            np.asarray(arrays["positions_zyx_um"]),
            np.asarray(arrays["times_seconds"]),
            np.asarray(arrays["spacing_zyx_um"]),
        ),
    }
    cache = tmp_path / "synthetic.npz"
    np.savez_compressed(
        cache,
        metadata_json=np.asarray(json.dumps(metadata)),
        positions_zyx_um=arrays["positions_zyx_um"],
        source_xyz_voxels=arrays["source_xyz_voxels"],
        times_seconds=arrays["times_seconds"],
        spacing_zyx_um=arrays["spacing_zyx_um"],
    )
    recording, loaded = load_coordinate_cache(cache)
    assert loaded == metadata
    assert recording.labels == ("roi_000", "roi_001", "roi_002")
    assert recording.frames == 3
    assert recording.neurons == 3


def test_semantic_label_digest_is_stable(tmp_path):
    source = tmp_path / "synthetic.nwb"
    _write_synthetic_nwb(source)
    with h5py.File(source, "r") as nwb:
        arrays = _extract_coordinate_arrays(nwb)
    expected = hashlib.sha256(b'["A","","C"]').hexdigest()
    assert arrays["semantic_labels_sha256"] == expected
