import json
from pathlib import Path

import numpy as np
import pytest

import deformeot.wormid_density_sensitivity as sensitivity_module
from deformeot.wormid import WormIDRecording
from deformeot.wormid_density_sensitivity import (
    ANALYSIS_ROLE,
    PROTOCOL_DEVIATION,
    DensityDefinition,
    TrackedDensityBlock,
    calibration_density_definitions,
    run_wormid_density_sensitivity,
    score_density_definitions,
    verify_q25_reproduction,
)
from deformeot.wormid_extent import PatchFeatureSet, save_patch_feature_cache


def _recording(recording_id: str, *, frames: int = 2) -> WormIDRecording:
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [2.0, 0.0, 0.1],
            [2.0, 0.8, 0.1],
            [4.0, 0.0, 0.2],
            [4.0, 0.8, 0.2],
        ]
    )
    positions = np.stack(
        [base + np.array([0.1 * frame, 0.03 * frame, 0.0]) for frame in range(frames)]
    )
    return WormIDRecording(
        path=f"/tmp/{recording_id}.nwb",
        recording_id=recording_id,
        positions_zyx_um=positions,
        labels=tuple(f"n{index}" for index in range(len(base))),
        times_seconds=np.arange(frames, dtype=float),
        spacing_zyx_um=(1.0, 0.3, 0.3),
        source_xyz_voxels=positions[..., ::-1],
    )


def test_calibration_definitions_use_reliable_patches_and_expected_sigma_proxies():
    recording = _recording("calibration", frames=2)
    descriptors = np.log(
        np.array(
            [
                [
                    [1.0, 2.0, 4.0],
                    [2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 2.0, 2.0],
                    [2.0, 3.0, 4.0],
                    [1.0, 1.0, 2.0],
                ],
                [
                    [2.0, 2.0, 4.0],
                    [9.0, 9.0, 9.0],
                    [1.0, 2.0, 3.0],
                    [1.0, 1.0, 3.0],
                    [2.0, 2.0, 3.0],
                    [1.0, 2.0, 4.0],
                ],
            ]
        )
    )
    reliability = np.ones((2, 6))
    reliability[1, 1] = 0.0
    features = PatchFeatureSet(
        descriptors=descriptors,
        reliability=reliability,
        feature_names=("log_std_small", "log_std_middle", "log_std_large"),
        extent_indices=(0, 1, 2),
        source_frame_indices=(0, 1),
        extraction_seconds=0.0,
    )

    definitions, metadata = calibration_density_definitions(recording, features)
    by_id = {definition.definition_id: definition for definition in definitions}
    reliable_sigmas = np.exp(descriptors)[reliability > 0]
    equivalent = np.cbrt(np.prod(reliable_sigmas, axis=1))
    rms = np.sqrt(np.mean(np.square(reliable_sigmas), axis=1))

    assert len(definitions) == 12
    assert metadata["reliable_patch_count"] == 11
    assert metadata["total_patch_count"] == 12
    assert by_id["two_x_median_equivalent_sigma"].threshold_um == pytest.approx(
        2 * np.median(equivalent)
    )
    assert by_id["two_x_median_rms_sigma"].threshold_um == pytest.approx(
        2 * np.median(rms)
    )
    assert by_id["two_x_median_largest_sigma"].threshold_um == pytest.approx(
        2 * np.median(np.max(reliable_sigmas, axis=1))
    )
    assert by_id["fixed_1_00_um"].threshold_um == 1.0
    assert by_id["fixed_4_00_um"].threshold_um == 4.0


def _sensitivity_blocks() -> tuple[TrackedDensityBlock, ...]:
    blocks = []
    for index in range(4):
        baseline = np.array([True, index % 2 == 0, False, True, False])
        extent = np.array([True, True, index >= 2, True, True])
        blocks.append(
            TrackedDensityBlock(
                recording_id=f"held-{index}",
                stride=1,
                nearest_neighbor_um=np.array([0.8, 1.2, 1.4, 2.2, 3.2]),
                baseline_correct=baseline,
                extent_correct=extent,
            )
        )
    return tuple(blocks)


def test_scoring_reuses_blocks_and_labels_every_threshold_and_recording_result():
    definitions = (
        DensityDefinition("calibration_nn_q25", "Q25", 1.5, "calibration"),
        DensityDefinition("fixed_4_00_um", "fixed", 4.0, "grid"),
    )
    recording_ids = tuple(f"held-{index}" for index in range(4))
    rows = score_density_definitions(
        _sensitivity_blocks(),
        definitions,
        recording_ids=recording_ids,
        bootstrap_samples=100,
        seed=17,
    )

    assert len(rows) == 2
    assert rows[0]["dense_prevalence"] == pytest.approx(3 / 5)
    assert rows[1]["dense_prevalence"] == 1.0
    for row in rows:
        assert row["analysis_role"] == ANALYSIS_ROLE
        assert row["protocol_deviation_statement"] == PROTOCOL_DEVIATION
        assert row["complete_recording_paired_bootstrap"]["recordings"] == 4
        assert len(row["complete_recording_paired_bootstrap"]["per_recording_differences"]) == 4
        assert all(
            summary["analysis_role"] == ANALYSIS_ROLE
            for summary in row["recording_summaries"].values()
        )


def test_q25_verification_matches_threshold_differences_and_bootstrap_interval():
    row = score_density_definitions(
        _sensitivity_blocks(),
        (DensityDefinition("calibration_nn_q25", "Q25", 1.5, "calibration"),),
        recording_ids=tuple(f"held-{index}" for index in range(4)),
        bootstrap_samples=100,
        seed=17,
    )[0]
    expected = row["complete_recording_paired_bootstrap"]
    reference = {
        "evaluation_strata": {"dense_threshold_um": 1.5},
        "confirmatory_summary": {
            "complete_recording_paired_bootstrap": {
                "dense_identity_accuracy_difference": expected
            }
        },
    }
    verified = verify_q25_reproduction((row,), reference)
    assert verified["passed"] is True

    reference["evaluation_strata"]["dense_threshold_um"] = 1.51
    with pytest.raises(AssertionError, match="does not reproduce"):
        verify_q25_reproduction((row,), reference)


def _features_for(recording: WormIDRecording, *, offset: float) -> PatchFeatureSet:
    frames = recording.frames
    neurons = recording.neurons
    descriptors = np.empty((frames, neurons, 15), dtype=float)
    for frame in range(frames):
        for neuron in range(neurons):
            descriptors[frame, neuron] = (
                np.linspace(0.1, 1.5, 15) + 0.03 * neuron + 0.01 * frame + offset
            )
    # The expected log-sigma slots must remain physical and finite.
    descriptors[..., 9] = np.log(0.7 + 0.02 * np.arange(neurons)[None, :])
    descriptors[..., 10] = np.log(0.9 + 0.02 * np.arange(neurons)[None, :])
    descriptors[..., 11] = np.log(1.1 + 0.02 * np.arange(neurons)[None, :])
    return PatchFeatureSet(
        descriptors=descriptors,
        reliability=np.ones((frames, neurons)),
        feature_names=(
            "f0",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "f6",
            "f7",
            "f8",
            "log_std_small",
            "log_std_middle",
            "log_std_large",
            "f12",
            "f13",
            "f14",
        ),
        extent_indices=tuple(range(15)),
        source_frame_indices=tuple(range(frames)),
        extraction_seconds=0.0,
    )


def test_runner_tracks_each_recording_stride_once_then_rescores_grid(tmp_path, monkeypatch):
    recordings = [_recording("calibration", frames=4)] + [
        _recording(f"held-{index}", frames=4) for index in range(4)
    ]
    sources = [tmp_path / f"source-{index}.nwb" for index in range(5)]
    source_lookup = {
        str(source.resolve()): recording for source, recording in zip(sources, recordings)
    }
    cache_dir = tmp_path / "features"
    for index, recording in enumerate(recordings):
        save_patch_feature_cache(
            _features_for(recording, offset=0.02 * index),
            cache_dir / f"{recording.recording_id}.npz",
        )

    monkeypatch.setattr(
        sensitivity_module,
        "load_wormid_recording",
        lambda path: source_lookup[str(Path(path).resolve())],
    )
    calls = {"baseline": 0, "extent": 0}
    original_baseline = sensitivity_module.track_sequence
    original_extent = sensitivity_module.track_dual_anchor_extent

    def counted_baseline(*args, **kwargs):
        calls["baseline"] += 1
        return original_baseline(*args, **kwargs)

    def counted_extent(*args, **kwargs):
        calls["extent"] += 1
        return original_extent(*args, **kwargs)

    monkeypatch.setattr(sensitivity_module, "track_sequence", counted_baseline)
    monkeypatch.setattr(sensitivity_module, "track_dual_anchor_extent", counted_extent)
    output = tmp_path / "sensitivity.json"
    payload = run_wormid_density_sensitivity(
        sources[0],
        sources[1:],
        feature_cache_dir=cache_dir,
        output_path=output,
        strides=(1, 2),
        bootstrap_samples=25,
    )

    assert calls == {"baseline": 8, "extent": 8}
    assert payload["tracking_execution"]["assignments_reused_across_all_thresholds"] is True
    assert payload["tracking_execution"]["thresholds_rescored"] == 12
    assert len(payload["sensitivity_results"]) == 12
    assert payload["interpretation_boundary"][
        "can_retroactively_satisfy_preregistered_density_target"
    ] is False
    assert json.loads(output.read_text())["analysis_role"] == ANALYSIS_ROLE
