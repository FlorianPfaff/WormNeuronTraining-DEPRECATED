from __future__ import annotations

import numpy as np

from deformeot.wormid import WormIDRecording, identity_blind_sequence
from deformeot.wormid_calibration import (
    FEATURE_NAMES,
    LabeledDiagnosticEvents,
    _maximum_coverage_at_accuracy,
    _risk_at_coverage,
    fit_blocked_confidence_model,
    label_diagnostic_events,
    track_dual_anchor_extent_with_diagnostics,
)
from deformeot.wormid_benchmark import ReferenceTrackerConfig
from deformeot.wormid_extent import (
    DualAnchorExtentConfig,
    PatchFeatureSet,
    fit_robust_feature_scaler,
    pair_patch_features_to_anonymous_detections,
    track_dual_anchor_extent,
)


def _synthetic_problem():
    rng = np.random.default_rng(51)
    native = rng.normal(size=(5, 9, 3))
    native += np.arange(5)[:, None, None] * np.array([0.15, -0.05, 0.08])
    recording = WormIDRecording(
        path="/tmp/synthetic.nwb",
        recording_id="synthetic",
        positions_zyx_um=native,
        labels=tuple(f"n{index}" for index in range(9)),
        times_seconds=np.arange(5, dtype=float),
        spacing_zyx_um=(1.0, 1.0, 1.0),
        source_xyz_voxels=native[..., ::-1],
    )
    descriptors = rng.normal(size=(5, 9, 3))
    features = PatchFeatureSet(
        descriptors=descriptors,
        reliability=np.full((5, 9), 0.8),
        feature_names=("a", "b", "c"),
        extent_indices=(0, 1, 2),
        source_frame_indices=tuple(range(5)),
        extraction_seconds=0.0,
    )
    sequence = identity_blind_sequence(recording, seed=19)
    anonymous = pair_patch_features_to_anonymous_detections(sequence, features)
    scaler = fit_robust_feature_scaler(descriptors)
    return recording, sequence, anonymous, scaler


def test_diagnostic_tracker_exactly_matches_production_assignments():
    _, sequence, anonymous, scaler = _synthetic_problem()
    geometry = ReferenceTrackerConfig(reference_weight=6.0)
    extent = DualAnchorExtentConfig(
        extent_weight=8.0,
        extent_update=0.25,
        feature_indices=(0, 1, 2),
    )
    expected = track_dual_anchor_extent(
        sequence,
        anonymous,
        scaler,
        geometry_config=geometry,
        extent_config=extent,
    )
    actual, diagnostics = track_dual_anchor_extent_with_diagnostics(
        sequence,
        anonymous,
        scaler,
        geometry_config=geometry,
        extent_config=extent,
    )
    for left, right in zip(expected.assignments, actual.assignments, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(expected.cloud_motion_um, actual.cloud_motion_um)
    assert diagnostics.features.shape == (4 * 9, len(FEATURE_NAMES))
    assert np.all(np.isfinite(diagnostics.features))


def test_truth_is_joined_only_after_diagnostics_and_initial_frame_is_excluded():
    recording, sequence, anonymous, scaler = _synthetic_problem()
    result, diagnostics = track_dual_anchor_extent_with_diagnostics(
        sequence,
        anonymous,
        scaler,
        extent_config=DualAnchorExtentConfig(
            extent_weight=2.0,
            feature_indices=(0, 1, 2),
        ),
    )
    labeled = label_diagnostic_events(
        sequence,
        result,
        diagnostics,
        recording_id=recording.recording_id,
        stride=1,
    )
    assert len(labeled.correctness) == (recording.frames - 1) * recording.neurons
    assert labeled.correctness.dtype == np.bool_
    assert set(labeled.recording_ids) == {recording.recording_id}


def test_blocked_model_fits_one_recording_and_returns_complete_oof_predictions():
    rng = np.random.default_rng(7)
    events = 500
    features = rng.normal(size=(events, len(FEATURE_NAMES)))
    correctness = features[:, 0] + 0.4 * rng.normal(size=events) > 0
    source_frames = np.repeat(np.arange(100), 5)
    calibration = LabeledDiagnosticEvents(
        features=features,
        correctness=correctness,
        recording_ids=np.full(events, "calibration", dtype=object),
        strides=np.tile(np.array([1, 5, 10, 20, 40]), 100),
        source_frame_indices=source_frames,
    )
    model, oof, selection = fit_blocked_confidence_model(
        calibration,
        regularization_grid=(0.1, 1.0),
    )
    assert oof.shape == (events,)
    assert np.all((oof > 0) & (oof < 1))
    assert selection["selected_regularization_C"] in (0.1, 1.0)
    assert model.predict_proba(features).shape == (events, 2)


def test_selective_summaries_use_requested_prefix_and_report_no_qualifying_prefix():
    probability = np.linspace(1.0, 0.01, 10)
    correctness = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    at_eighty = _risk_at_coverage(probability, correctness, 0.8)
    assert at_eighty["selected_events"] == 8
    assert at_eighty["selective_accuracy"] == 0.75
    maximum = _maximum_coverage_at_accuracy(probability, correctness, 0.995)
    assert maximum["selected_events"] == 6

    none = _maximum_coverage_at_accuracy(probability, ~correctness, 0.995)
    assert none["selected_events"] == 0
    assert none["accuracy"] is None
