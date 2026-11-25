from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest

from deformeot.paper_phase2_rescore import main, rescore_file, rescore_path, score_assignment_arrays


def arrays(assignments, *, truth=None, indices=None, times=None):
    values = np.asarray(assignments, dtype=np.int64)
    frames, neurons = values.shape
    return {
        "assignments": values,
        "truth_indices": np.asarray(truth, dtype=np.int64) if truth is not None else np.tile(np.arange(neurons, dtype=np.int64), (frames, 1)),
        "frame_indices": np.asarray(indices, dtype=np.int64) if indices is not None else np.arange(frames, dtype=np.int64),
        "frame_times": np.asarray(times, dtype=float) if times is not None else np.arange(frames, dtype=float),
    }


def archive(path, values):
    np.savez_compressed(path, **values)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nonidentity_initial_truth_and_anonymous_row_permutations():
    values = arrays([[0, 1, 2], [2, 0, 1], [1, 2, 0]],
                    truth=[[2, 0, 1], [0, 1, 2], [1, 2, 0]])
    result = score_assignment_arrays(**values)
    assert result["correct"] == result["total"] == 9
    assert result["correct_excluding_initialization"] == result["total_excluding_initialization"] == 6
    assert result["identity_switches"] == result["wrong_identity_frames"] == 0
    assert result["coverage"] == 1.0


def test_stable_wrong_identity_does_not_add_switches_after_the_initial_swap():
    result = score_assignment_arrays(**arrays([[0, 1, 2], [1, 0, 2], [1, 0, 2], [1, 0, 2]]))
    assert result["correct"] == 6
    assert result["total"] == 12
    assert result["correct_excluding_initialization"] == 3
    assert result["total_excluding_initialization"] == 9
    assert result["identity_switches"] == 2
    assert result["wrong_identity_frames"] == 6
    assert result["wrong_identity_episode_count"] == 2
    assert result["mean_wrong_identity_duration_selected_scans"] == 3
    assert result["maximum_wrong_identity_duration_selected_scans"] == 3
    assert result["mean_wrong_identity_observed_span_seconds"] == 2
    assert result["right_endpoint_weighted_wrong_exposure_seconds"] == 6


def test_swap_then_recovery_counts_both_track_changes_and_closes_episodes():
    result = score_assignment_arrays(**arrays([[0, 1, 2], [1, 0, 2], [0, 1, 2], [0, 1, 2]]))
    assert result["identity_switches"] == 4
    assert result["correct"] == 10
    assert result["correct_excluding_initialization"] == 7
    assert result["wrong_identity_episode_count"] == 2
    assert result["mean_wrong_identity_duration_selected_scans"] == 1
    assert result["mean_wrong_identity_observed_span_seconds"] == 0
    assert result["right_endpoint_weighted_wrong_exposure_seconds"] == 2


def test_changing_between_two_wrong_identities_stays_one_wrong_episode():
    result = score_assignment_arrays(**arrays([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2]]))
    assert result["identity_switches"] == 9
    assert result["wrong_identity_episode_count"] == 3
    assert result["wrong_identity_frames"] == 6
    assert result["maximum_wrong_identity_duration_selected_scans"] == 2


def test_irregular_tail_counts_observed_scans_without_extrapolating_a_full_stride():
    values = arrays([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]],
                    indices=[0, 10, 20, 30, 31], times=[0, 2, 4, 6, 6.2])
    result = score_assignment_arrays(**values)
    assert result["identity_switches"] == 6
    assert result["wrong_identity_frames"] == 6
    assert result["wrong_identity_episode_count"] == 4
    assert result["mean_wrong_identity_duration_selected_scans"] == 1.5
    assert result["maximum_wrong_identity_duration_selected_scans"] == 2
    assert result["right_endpoint_weighted_wrong_exposure_seconds"] == pytest.approx(8.4)
    assert result["mean_wrong_identity_observed_span_seconds"] == 1
    assert result["source_last_frame"] == 31
    assert result["recording_observed_span_seconds"] == 6.2


def test_single_initialized_frame_has_defined_counts_and_no_excluded_accuracy():
    result = score_assignment_arrays(**arrays([[0, 1]], truth=[[1, 0]], indices=[20], times=[7.5]))
    assert result["correct"] == result["total"] == 2
    assert result["total_excluding_initialization"] == 0
    assert result["accuracy_excluding_initialization"] is None
    assert result["identity_switch_opportunities"] == result["identity_switches"] == 0
    assert result["mean_wrong_identity_duration_selected_scans"] == 0


@pytest.mark.parametrize("name,value,match", [
    ("assignments", np.asarray([[0, 1], [0, 0]], dtype=np.int64), "exhaustive permutation"),
    ("assignments", np.asarray([[0, 1], [-1, 1]], dtype=np.int64), "exhaustive permutation"),
    ("assignments", np.asarray([[0, 1], [0, 2]], dtype=np.int64), "exhaustive permutation"),
    ("truth_indices", np.asarray([[0, 1], [0, 0]], dtype=np.int64), "exhaustive permutation"),
    ("assignments", np.asarray([[1, 0], [0, 1]], dtype=np.int64), "initialized frame"),
    ("assignments", np.asarray([[0., 1.], [0., 1.]]), "signed int64"),
    ("truth_indices", np.asarray([[0, 1], [0, 1]], dtype=np.uint64), "signed int64"),
    ("frame_indices", np.asarray([0, 1], dtype=np.int32), "signed int64"),
    ("frame_indices", np.asarray([0, 0], dtype=np.int64), "strictly increasing"),
    ("frame_indices", np.asarray([-1, 1], dtype=np.int64), "nonnegative"),
    ("frame_times", np.asarray([0., 0.]), "strictly increasing"),
    ("frame_times", np.asarray([0., np.nan]), "finite"),
    ("frame_times", np.asarray([False, True]), "finite numeric"),
])
def test_invalid_contract_is_rejected(name, value, match):
    values = arrays([[0, 1], [0, 1]])
    values[name] = value
    with pytest.raises(ValueError, match=match):
        score_assignment_arrays(**values)


def test_artifact_hash_mismatch_is_detected_before_numpy_loading(tmp_path, monkeypatch):
    path = tmp_path / "case.npz"
    archive(path, arrays([[0, 1], [1, 0]]))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("must reject hash before loading"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        rescore_file(path, expected_sha256="0" * 64)


def test_file_and_directory_rescoring_and_manifest_integrity(tmp_path):
    path = tmp_path / "case.npz"
    checksum = archive(path, arrays([[0, 1], [1, 0]]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "case.npz", "sha256": checksum}]}))
    direct = rescore_file(path, checksum)
    directory = rescore_path(tmp_path, manifest)
    assert directory["artifact_count"] == 1
    assert directory["artifacts"][0] == direct
    assert direct["manifest_hash_verified"]
    assert direct["metrics"]["correct"] == 2


def test_manifest_omission_and_missing_declared_artifact_fail_closed(tmp_path):
    path = tmp_path / "case.npz"
    checksum = archive(path, arrays([[0], [0]]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    with pytest.raises(ValueError, match="missing from manifest"):
        rescore_path(tmp_path, manifest)
    manifest.write_text(json.dumps({"files": [{"path": "case.npz", "sha256": checksum},
                                              {"path": "missing.npz", "sha256": "0" * 64}]}))
    with pytest.raises(FileNotFoundError, match="declared assignment artifact missing"):
        rescore_path(tmp_path, manifest)


def test_manifest_duplicate_and_traversal_are_rejected(tmp_path):
    path = tmp_path / "case.npz"
    checksum = archive(path, arrays([[0], [0]]))
    manifest = tmp_path / "manifest.json"
    row = {"path": "case.npz", "sha256": checksum}
    manifest.write_text(json.dumps({"files": [row, row]}))
    with pytest.raises(ValueError, match="duplicate"):
        rescore_path(path, manifest)
    manifest.write_text(json.dumps({"files": [{"path": "../case.npz", "sha256": checksum}]}))
    with pytest.raises(ValueError, match="parent traversal"):
        rescore_path(path, manifest)


def test_missing_contract_arrays_are_not_silently_scored(tmp_path):
    path = tmp_path / "old_october_audit.npz"
    np.savez(path, assignments=np.asarray([[0, 1]], dtype=np.int64))
    with pytest.raises(ValueError, match="missing arrays"):
        rescore_file(path)


def test_cli_json_output_and_prevention_of_overwriting_any_input_artifact(tmp_path, monkeypatch, capsys):
    path = tmp_path / "case.npz"
    checksum = archive(path, arrays([[0, 1], [0, 1]]))
    monkeypatch.setattr(sys, "argv", ["rescore", str(path)])
    main()
    assert json.loads(capsys.readouterr().out)["artifact_count"] == 1
    monkeypatch.setattr(sys, "argv", ["rescore", str(tmp_path), "--output", str(path)])
    with pytest.raises(SystemExit):
        main()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
