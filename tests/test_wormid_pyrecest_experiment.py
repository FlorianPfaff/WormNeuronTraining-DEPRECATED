from __future__ import annotations

import json

import numpy as np
import pytest

from deformeot.wormid_benchmark import TrackResult
from deformeot.wormid_pyrecest_experiment import (
    _latency_metadata, _validate_exhaustive_result, checkpoint_path,
    read_checkpoint, select_tuning_candidates, summarize_comparisons,
    validate_candidates,
)


def _row(recording, stride, candidate, accuracy, *, phase="evaluate", status="complete", switches=5, runtime=2., lag=0):
    return {"recording_id": recording, "stride": stride, "candidate": candidate,
        "phase": phase, "status": status, "identity_accuracy": accuracy,
        "identity_switch_rate": switches / 100, "identity_switches": switches,
        "runtime_seconds": runtime, "latency_selected_scans": lag}


def test_selection_uses_only_original_tuning_and_requires_complete_strides():
    candidates = [
        {"name": "gnn_good", "method": "pyrecest_gnn", "config": {"process_noise_um": .5}},
        {"name": "gnn_incomplete", "method": "pyrecest_gnn", "config": {}},
        {"name": "mht", "method": "pyrecest_mht", "config": {"fixed_lag": 2}},
    ]
    rows = [_row("tuning", stride, name, accuracy, phase="tune")
            for stride in (1, 5) for name, accuracy in (("gnn_good", .8), ("mht", .85))]
    rows.append(_row("tuning", 1, "gnn_incomplete", 1., phase="tune"))
    selected = select_tuning_candidates(rows, candidates, tuning_recording="tuning", strides=(1, 5))
    assert [candidate["name"] for candidate in selected["selected_candidates"]] == ["gnn_good", "mht"]
    assert selected["global_winner"] == "mht"
    assert selected["selection_uses_evaluation_outcomes"] is False
    with pytest.raises(ValueError, match="only the single tuning"):
        select_tuning_candidates([*rows, _row("evaluation", 1, "gnn_good", 1.)], candidates,
                                 tuning_recording="tuning", strides=(1, 5))


def test_control_is_excluded_and_latency_families_are_selected_separately():
    candidates = [
        {"name": "control", "method": "pyrecest_gnn", "control": True, "config": {}},
        {"name": "gnn", "method": "pyrecest_gnn", "config": {}},
        {"name": "online", "method": "pyrecest_mht", "family": "mht_online", "config": {"fixed_lag": 0}},
        {"name": "delayed", "method": "pyrecest_mht", "family": "mht_delayed", "config": {"fixed_lag": 2}},
    ]
    rows = [_row("tuning", 1, name, accuracy, phase="tune") for name, accuracy in
            (("control", 1.), ("gnn", .8), ("online", .85), ("delayed", .9))]
    selected = select_tuning_candidates(rows, candidates, tuning_recording="tuning", strides=(1,))
    assert [candidate["name"] for candidate in selected["selected_candidates"]] == ["gnn", "online", "delayed"]
    assert selected["global_winner"] == "delayed"


def test_recording_bootstrap_equal_weights_and_incomplete_coverage_is_explicit():
    rows = []
    for recording, delta in (("a", .02), ("b", .04)):
        for stride in (1, 5):
            rows.extend([_row(recording, stride, "dual_anchor", .8),
                         _row(recording, stride, "first_affine", .7),
                         _row(recording, stride, "mht", .8 + delta, lag=2)])
    result = summarize_comparisons(rows, recording_ids=("a", "b"), strides=(1, 5),
        candidate_names=("mht",), bootstrap_samples=200, bootstrap_seed=7)
    comparison = result["comparisons"]["mht_vs_dual_anchor"]
    assert np.isclose(comparison["accuracy_difference"]["mean"], .03)
    assert comparison["accuracy_difference"]["recordings"] == 2
    assert comparison["extra_future_scan_information"] is True
    assert comparison["improvement_criterion_met"] is True
    incomplete = summarize_comparisons(rows[:-1], recording_ids=("a", "b"), strides=(1, 5),
        candidate_names=("mht",), bootstrap_samples=200, bootstrap_seed=7)
    comparison = incomplete["comparisons"]["mht_vs_dual_anchor"]
    assert comparison["paired_complete_recordings"] == 1
    assert comparison["complete"] is False
    assert comparison["improvement_criterion_met"] is False


def test_checkpoint_requires_exact_provenance_and_retries_failed_tasks(tmp_path):
    path = checkpoint_path(tmp_path, "evaluate", "sub-example", 5, "gnn")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"task_signature": "original", "status": "complete", "correct": 5}))
    assert read_checkpoint(path, "original")["correct"] == 5
    with pytest.raises(RuntimeError, match="provenance changed"):
        read_checkpoint(path, "different")
    path.write_text(json.dumps({"task_signature": "original", "status": "failed"}))
    assert read_checkpoint(path, "original") is None


def test_result_validator_rejects_duplicate_associations_and_latency_accounts_for_tail():
    correct = TrackResult("test", (np.array([0, 1]), np.array([1, 0])), .1, np.array([0., .1]))
    _validate_exhaustive_result(correct, 2, 2)
    duplicate = TrackResult("test", (np.array([0, 1]), np.array([1, 1])), .1, np.array([0., .1]))
    with pytest.raises(ValueError, match="one-to-one"):
        _validate_exhaustive_result(duplicate, 2, 2)
    latency = _latency_metadata(np.arange(7, dtype=float), (0, 2, 4, 6),
                               {"method": "pyrecest_mht", "config": {"fixed_lag": 2}})
    assert latency["latency_selected_scans"] == 2
    assert latency["maximum_actual_lookahead_seconds"] == 4.
    assert latency["mean_actual_lookahead_seconds"] == 1.5


def test_candidate_names_and_duplicate_rows_are_rejected():
    with pytest.raises(ValueError, match="unsafe"):
        validate_candidates([{"name": "../escape", "method": "pyrecest_gnn"}])
    with pytest.raises(ValueError, match="reserved"):
        validate_candidates([{"name": "dual_anchor", "method": "pyrecest_gnn"}])
    row = _row("a", 1, "gnn", .9)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_comparisons([row, row], recording_ids=("a",), strides=(1,),
            candidate_names=("gnn",), bootstrap_samples=100, bootstrap_seed=1)
