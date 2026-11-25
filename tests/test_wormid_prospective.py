from __future__ import annotations

import hashlib

import numpy as np

from deformeot.wormid import WormIDRecording
from deformeot.wormid_benchmark import METHODS, ReferenceTrackerConfig
from deformeot.wormid_prospective import (
    _benchmark_recording,
    analyze_prospective_rows,
    verify_frozen_protocol,
)


UNTOUCHED = (
    "sub-20190924-01_ses-20190924_ophys",
    "sub-20190924-02_ses-20190924_ophys",
    "sub-20190924-03_ses-20190924_ophys",
    "sub-20190925-01_ses-20190925_ophys",
    "sub-20190928-01_ses-20190928_ophys",
    "sub-20190928-03_ses-20190928_ophys",
    "sub-20190928-05_ses-20190928_ophys",
    "sub-20190928-07_ses-20190928_ophys",
    "sub-20190928-08_ses-20190928_ophys",
    "sub-20190928-11_ses-20190928_ophys",
    "sub-20190928-13_ses-20190928_ophys",
    "sub-20190929-03_ses-20190929_ophys",
    "sub-20190929-06_ses-20190929_ophys",
    "sub-20191030-03_ses-20191030_ophys",
    "sub-20191104-08_ses-20191104_ophys",
    "sub-20191104-10_ses-20191104_ophys",
)
LEGACY = (
    "sub-20190929-02_ses-20190929_ophys",
    "sub-20190929-05_ses-20190929_ophys",
    "sub-20190929-07_ses-20190929_ophys",
    "sub-20191030-07_ses-20191030_ophys",
)


def test_freeze_verifier_detects_tampering(tmp_path):
    source = tmp_path / "protocol.txt"
    source.write_text("frozen\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "freeze.sha256"
    manifest.write_text(f"{digest}  protocol.txt\n", encoding="utf-8")
    result = verify_frozen_protocol(tmp_path, freeze_path="freeze.sha256")
    assert result["all_verified"] is True
    source.write_text("changed\n", encoding="utf-8")
    try:
        verify_frozen_protocol(tmp_path, freeze_path="freeze.sha256")
    except RuntimeError as error:
        assert "integrity failure" in str(error)
    else:  # pragma: no cover
        raise AssertionError("tampering should have failed the freeze check")


def _synthetic_rows():
    rows = []
    for recording_id in (*LEGACY, *UNTOUCHED):
        cohort = "legacy_evaluation" if recording_id in LEGACY else "untouched_expansion"
        acquisition_date = recording_id.split("_ses-")[1][:8]
        for stride in (1, 5, 10, 20, 40):
            for method in METHODS:
                accuracy = {
                    "previous_hungarian": 0.70,
                    "pairwise_affine": 0.75,
                    "first_affine": 0.80,
                    "dual_anchor": 0.82,
                }[method]
                switches = {
                    "previous_hungarian": 20,
                    "pairwise_affine": 15,
                    "first_affine": 10,
                    "dual_anchor": 5,
                }[method]
                rows.append(
                    {
                        "recording_id": recording_id,
                        "cohort": cohort,
                        "acquisition_date": acquisition_date,
                        "stride": stride,
                        "method": method,
                        "identity_accuracy": accuracy,
                        "identity_switches": switches,
                        "identity_switch_opportunities": 100,
                        "identity_switch_rate": switches / 100,
                        "high_motion_identity_accuracy": accuracy - 0.05,
                        "high_motion_identity_switches": switches // 2,
                        "high_motion_identity_switch_opportunities": 25,
                        "high_motion_identity_switch_rate": (switches // 2) / 25,
                    }
                )
    return rows


def test_prospective_analysis_applies_recording_not_frame_bootstrap():
    result = analyze_prospective_rows(
        _synthetic_rows(),
        untouched_ids=UNTOUCHED,
        legacy_ids=LEGACY,
        primary_samples=500,
        primary_seed=20260720,
        hierarchical_samples=500,
        hierarchical_seed=20260721,
        same_date_exclusion="sub-20190925-01_ses-20190925_ophys",
    )
    primary = result["primary_untouched_accuracy_effect"]
    assert primary["recordings"] == 16
    assert np.isclose(primary["mean"], 0.02)
    assert primary["ci95_low"] > 0
    assert primary["replication_criterion_met"] is True
    assert result["same_tuning_date_exclusion_sensitivity"]["recordings"] == 15
    hierarchy = result["untouched_acquisition_date_hierarchical_bootstrap"]
    assert hierarchy["mean_accuracy_difference"]["dates"] == 6
    assert result["nontuning_20_stratified_summary"]["legacy_evaluation"][
        "recordings"
    ] == 4
    assert result["nontuning_20_stratified_summary"]["all_nontuning"][
        "recordings"
    ] == 20


def test_benchmark_uses_full_inventory_seed_and_roi_identity_semantics():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(12, 3))
    positions = np.stack([base, base + 0.1, base + 0.2])
    recording = WormIDRecording(
        path="https://example.invalid/recording.nwb",
        recording_id="sub-20190924-01_ses-20190924_ophys",
        positions_zyx_um=positions,
        labels=tuple(f"roi_{index:03d}" for index in range(12)),
        times_seconds=np.arange(3, dtype=float),
        spacing_zyx_um=(1.0, 1.0, 1.0),
        source_xyz_voxels=positions[..., ::-1],
    )
    rows = _benchmark_recording(
        (
            recording,
            "untouched_expansion",
            7,
            (1,),
            20260720,
            ReferenceTrackerConfig(reference_weight=6.0, descriptor_weight=0.0),
        )
    )
    assert len(rows) == 4
    assert {row["permutation_seed"] for row in rows} == {20267721}
    assert all(row["identity_semantics"].startswith("persistent initialized ROI") for row in rows)
    assert all(row["semantic_labels_used_for_association"] is False for row in rows)
