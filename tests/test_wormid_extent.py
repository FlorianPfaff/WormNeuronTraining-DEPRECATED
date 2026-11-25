import h5py
import numpy as np

from deformeot.wormid import WormIDRecording, identity_blind_sequence
from deformeot.wormid_benchmark import ReferenceTrackerConfig, track_sequence
from deformeot.wormid_extent import (
    AssociationConfig,
    AnonymousPatchSequence,
    DualAnchorExtentConfig,
    PatchFeatureSet,
    fit_robust_feature_scaler,
    load_patch_feature_cache,
    pair_patch_features_to_anonymous_detections,
    patch_shape_descriptors,
    score_tracking,
    run_wormid_extent_benchmark,
    save_patch_feature_cache,
    track_dual_anchor_extent,
    track_permuted_recording,
)


def _gaussian_volume(axis: int) -> np.ndarray:
    grid = np.meshgrid(
        np.arange(17) - 8,
        np.arange(17) - 8,
        np.arange(9) - 4,
        indexing="ij",
    )
    scales = np.array([1.2, 1.2, 1.2])
    scales[axis] = 2.8
    exponent = sum((coordinates / scale) ** 2 for coordinates, scale in zip(grid, scales))
    return 7.0 + 100.0 * np.exp(-0.5 * exponent)


def test_descriptor_is_additive_and_multiplicative_intensity_invariant():
    volume = _gaussian_volume(0)
    center = np.array([[8.0, 8.0, 4.0]])
    first, quality, names, _ = patch_shape_descriptors(
        volume,
        center,
        (0.3, 0.3, 1.0),
    )
    second, _, second_names, _ = patch_shape_descriptors(
        4.5 * volume + 23.0,
        center,
        (0.3, 0.3, 1.0),
    )
    assert names == second_names
    assert quality[0] > 0
    np.testing.assert_allclose(first, second, atol=1e-10)


def test_extent_descriptor_distinguishes_orthogonal_elongation_and_boundaries_are_finite():
    center = np.array([[8.0, 8.0, 4.0]])
    x_shape, _, _, extent = patch_shape_descriptors(
        _gaussian_volume(0), center, (0.3, 0.3, 1.0)
    )
    y_shape, _, _, _ = patch_shape_descriptors(
        _gaussian_volume(1), center, (0.3, 0.3, 1.0)
    )
    assert x_shape[0, 3] > x_shape[0, 4]
    assert y_shape[0, 4] > y_shape[0, 3]
    assert not np.allclose(x_shape[:, extent], y_shape[:, extent])

    boundary, quality, _, _ = patch_shape_descriptors(
        _gaussian_volume(0), np.array([[0.0, 0.0, 0.0]]), (0.3, 0.3, 1.0)
    )
    assert np.all(np.isfinite(boundary))
    assert 0 <= quality[0] <= 1


def test_permuted_tracker_uses_shape_to_resolve_position_swap():
    # Two objects exchange positions.  Their local descriptors remain unique.
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
            [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
        ]
    )
    recording = WormIDRecording(
        path="/tmp/fake.nwb",
        recording_id="fake",
        positions_zyx_um=positions,
        labels=("a", "b"),
        times_seconds=np.array([0.0, 1.0]),
        spacing_zyx_um=(1.0, 1.0, 1.0),
        source_xyz_voxels=positions[..., ::-1],
    )
    descriptors = np.array([[[0.0], [10.0]], [[0.0], [10.0]]])
    features = PatchFeatureSet(
        descriptors=descriptors,
        reliability=np.ones((2, 2)),
        feature_names=("shape",),
        extent_indices=(0,),
        source_frame_indices=(0, 1),
        extraction_seconds=0.0,
    )
    scaler = fit_robust_feature_scaler(descriptors)

    centroid = track_permuted_recording(
        recording,
        features,
        scaler,
        frame_indices=(0, 1),
        config=AssociationConfig(),
        seed=7,
    )
    shape = track_permuted_recording(
        recording,
        features,
        scaler,
        frame_indices=(0, 1),
        config=AssociationConfig(appearance_weight=10.0),
        seed=7,
    )
    assert score_tracking(centroid, recording)["identity_accuracy"] == 0.5
    assert score_tracking(shape, recording)["identity_accuracy"] == 1.0


def test_zero_extent_weight_exactly_matches_frozen_dual_anchor_geometry():
    rng = np.random.default_rng(42)
    positions = np.stack(
        [rng.normal(size=(8, 3)) + np.array([0.2 * frame, 0.0, 0.0]) for frame in range(4)]
    )
    recording = WormIDRecording(
        path="/tmp/fake.nwb",
        recording_id="fake",
        positions_zyx_um=positions,
        labels=tuple(f"n{index}" for index in range(8)),
        times_seconds=np.arange(4, dtype=float),
        spacing_zyx_um=(1.0, 1.0, 1.0),
        source_xyz_voxels=positions[..., ::-1],
    )
    sequence = identity_blind_sequence(recording, seed=17)
    native_descriptors = np.arange(32, dtype=float).reshape(4, 8, 1)
    features = PatchFeatureSet(
        descriptors=native_descriptors,
        reliability=np.ones((4, 8)),
        feature_names=("extent",),
        extent_indices=(0,),
        source_frame_indices=(0, 1, 2, 3),
        extraction_seconds=0.0,
    )
    anonymous = pair_patch_features_to_anonymous_detections(sequence, features)
    scaler = fit_robust_feature_scaler(native_descriptors)
    geometry = ReferenceTrackerConfig(reference_weight=6.0)
    expected = track_sequence(sequence, "dual_anchor", config=geometry)
    actual = track_dual_anchor_extent(
        sequence,
        anonymous,
        scaler,
        geometry_config=geometry,
        extent_config=DualAnchorExtentConfig(extent_weight=0.0),
    )
    for left, right in zip(expected.assignments, actual.assignments, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(expected.cloud_motion_um, actual.cloud_motion_um)

    # In the strict no-intensity arm, even adversarially different reliability
    # values cannot affect association.
    changed_quality = AnonymousPatchSequence(
        anonymous.descriptors,
        tuple(np.linspace(0.01, 1.0, 8) for _ in range(4)),
    )
    no_gate = DualAnchorExtentConfig(
        extent_weight=8.0,
        extent_update=0.25,
        feature_indices=(0,),
        use_reliability_gate=False,
    )
    normal = track_dual_anchor_extent(
        sequence,
        anonymous,
        scaler,
        geometry_config=geometry,
        extent_config=no_gate,
    )
    altered = track_dual_anchor_extent(
        sequence,
        changed_quality,
        scaler,
        geometry_config=geometry,
        extent_config=no_gate,
    )
    for left, right in zip(normal.assignments, altered.assignments, strict=True):
        np.testing.assert_array_equal(left, right)


def _write_tiny_benchmark_nwb(
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
        start = nwb.create_dataset("acquisition/CalciumImageSeries/starting_time", data=0.0)
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


def test_reproducible_runner_generates_cache_grid_and_frozen_comparisons(tmp_path):
    calibration = tmp_path / "calibration.nwb"
    held_out = tmp_path / "held_out.nwb"
    _write_tiny_benchmark_nwb(calibration, offset=0)
    _write_tiny_benchmark_nwb(held_out, offset=2, separation=8, frame_step=2)
    output = tmp_path / "result.json"
    cache_dir = tmp_path / "features"
    payload = run_wormid_extent_benchmark(
        calibration,
        [held_out],
        feature_cache_dir=cache_dir,
        output_path=output,
        strides=(1, 2),
        calibration_gap_strides=(2,),
        extent_weights=(0.1,),
        extent_updates=(0.0,),
        bootstrap_samples=25,
    )
    assert output.is_file()
    assert payload["calibration_grid"]["selected_config"]["extent_weight"] == 0.1
    assert len(payload["confirmatory_held_out_results"]) == 4
    assert len(payload["no_intensity_held_out_results"]) == 2
    assert payload["analysis_roles"]["confirmatory_primary"].startswith(
        "reliability-gated"
    )

    cached = load_patch_feature_cache(cache_dir / "calibration.npz")
    roundtrip_path = save_patch_feature_cache(cached, tmp_path / "roundtrip.npz")
    roundtrip = load_patch_feature_cache(roundtrip_path)
    np.testing.assert_allclose(cached.descriptors, roundtrip.descriptors)
