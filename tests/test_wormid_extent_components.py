import h5py
import numpy as np

from deformeot.wormid_extent import PatchFeatureSet, run_wormid_extent_benchmark
from deformeot.wormid_extent_components import (
    CUE_COMPONENTS,
    EXPECTED_SPATIAL_FEATURE_NAMES,
    fit_component_scalers,
    run_extent_component_ablation,
)


def _write_tiny_nwb(
    path,
    *,
    offset: int,
    separation: int = 10,
    frame_step: int = 1,
) -> None:
    base = np.array(
        [
            [45, 35, 5],
            [45 + separation, 35, 5],
            [45, 35 + separation, 7],
            [45 + separation, 35 + separation, 7],
        ],
        dtype=int,
    )
    with h5py.File(path, "w") as nwb:
        nwb.create_dataset(
            "processing/CalciumActivity/NeuronIDs/labels",
            data=np.asarray([b"a", b"b", b"c", b"d"]),
        )
        nwb.create_dataset(
            "general/optophysiology/CalciumImVol/grid_spacing",
            data=[0.3, 0.3, 1.0],
        )
        start = nwb.create_dataset(
            "acquisition/CalciumImageSeries/starting_time",
            data=0.0,
        )
        start.attrs["rate"] = 4.0
        movie = np.zeros((4, 256, 128, 21), dtype=np.uint16)
        segmentation = nwb.create_group(
            "processing/CalciumActivity/CalciumSeriesSegmentation"
        )
        mask_dtype = np.dtype(
            [("x", "i4"), ("y", "i4"), ("z", "i4"), ("weight", "f4")]
        )
        for frame in range(4):
            centers = base + np.array([offset + frame_step * frame, 0, 0])
            group = segmentation.create_group(f"Seg_tpoint_{frame}")
            group.create_dataset("id", data=np.arange(4))
            mask = np.zeros(4, dtype=mask_dtype)
            mask["x"], mask["y"], mask["z"] = centers.T
            mask["weight"] = 1.0
            group.create_dataset("voxel_mask", data=mask)
            group.create_dataset("voxel_mask_index", data=np.arange(1, 5))
            for identity, (x, y, z) in enumerate(centers):
                movie[frame, x - 1 : x + 2, y - 1 : y + 2, z] = 20 + 5 * identity
        nwb.create_dataset("acquisition/CalciumImageSeries/data", data=movie)


def test_component_scalers_use_only_reliable_calibration_patches():
    reliable = np.arange(30, dtype=float).reshape(2, 15)
    unreliable = reliable + 10_000
    descriptors = np.stack([reliable, unreliable])
    features = PatchFeatureSet(
        descriptors=descriptors,
        reliability=np.array([[1.0, 1.0], [0.0, 0.0]]),
        feature_names=EXPECTED_SPATIAL_FEATURE_NAMES,
        extent_indices=tuple(range(15)),
        source_frame_indices=(0, 1),
        extraction_seconds=0.0,
    )
    scalers = fit_component_scalers(features)

    assert [len(scalers[item.method].center) for item in CUE_COMPONENTS] == [15, 3, 12]
    for component in CUE_COMPONENTS:
        np.testing.assert_allclose(
            scalers[component.method].center,
            np.median(reliable[:, component.indices], axis=0),
        )


def test_ablation_is_complete_and_full_arm_exactly_reproduces_reference(tmp_path):
    calibration = tmp_path / "calibration.nwb"
    held_out = tmp_path / "held_out.nwb"
    cache_dir = tmp_path / "features"
    reference_path = tmp_path / "reference.json"
    output_path = tmp_path / "components.json"
    _write_tiny_nwb(calibration, offset=0)
    _write_tiny_nwb(held_out, offset=2, separation=8, frame_step=2)

    run_wormid_extent_benchmark(
        calibration,
        [held_out],
        feature_cache_dir=cache_dir,
        output_path=reference_path,
        strides=(1, 2),
        calibration_gap_strides=(2,),
        extent_weights=(8.0,),
        extent_updates=(0.25,),
        bootstrap_samples=25,
    )
    payload = run_extent_component_ablation(
        calibration,
        [held_out],
        feature_cache_dir=cache_dir,
        reference_result_path=reference_path,
        output_path=output_path,
        strides=(1, 2),
        bootstrap_samples=25,
    )

    assert output_path.is_file()
    assert payload["analysis_role"].startswith("post-hoc mechanism ablation")
    assert payload["full_15d_reference_reproduction"]["exact"] is True
    assert len(payload["held_out_results"]) == 8
    assert set(payload["overall_pooled_by_method"]) == {
        "dual_anchor_point_only",
        "dual_anchor_full_spatial_15d",
        "dual_anchor_centroid_offset_3d",
        "dual_anchor_extent_only_12d",
    }
    assert all(
        comparison["complete_recording_paired_bootstrap"][
            "tracked_identity_accuracy_difference"
        ]["samples"]
        == 25
        for comparison in payload["comparisons_to_point_only"].values()
    )
