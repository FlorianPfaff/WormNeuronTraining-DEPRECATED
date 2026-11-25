from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pytest

from deformeot.paper_pilot_audit import (
    array_sha256,
    compare_captured_costs,
    compare_reconstructed_runs,
    observed_track,
    scoring_fixture_audit,
    select_diagnostic_cases,
    _run_worker,
)
from deformeot.tracker import DetectionFrame
from deformeot.wormid import IdentityBlindSequence
from deformeot.wormid_benchmark import ReferenceTrackerConfig, score_tracks, track_sequence


def _sequence():
    rng = np.random.default_rng(12)
    base = rng.normal(size=(12, 3)) * 10
    frames, truth = [], []
    for index in range(5):
        order = rng.permutation(len(base))
        moved = base + [0.1 * index, 0.02 * index, -0.05 * index]
        frames.append(DetectionFrame(moved[order], time=index))
        truth.append(order)
    return IdentityBlindSequence(tuple(frames), tuple(truth), tuple(map(str, truth[0])), tuple(range(5)))


@pytest.mark.parametrize("method", ["dual_anchor", "first_affine"])
def test_observer_preserves_frozen_assignments_and_metrics(method):
    sequence = _sequence()
    original = track_sequence(sequence, method)
    observed, trace, captured = observed_track(sequence, method, ReferenceTrackerConfig(), capture_frame=2)
    assert array_sha256(np.asarray(original.assignments), integer=True) == array_sha256(np.asarray(observed.assignments), integer=True)
    original_metrics, observed_metrics = asdict(score_tracks(sequence, original)), asdict(score_tracks(sequence, observed))
    original_metrics.pop("runtime_seconds")
    observed_metrics.pop("runtime_seconds")
    assert original_metrics == observed_metrics
    assert len(trace) == 4
    assert "outer0_cost" in captured
    assert "affine0_iteration0_cost" in captured
    assert captured["affine0_matrix"].shape == (3, 3)


def _selection_rows():
    old, new = [], []
    for method in ("dual_anchor", "first_affine"):
        for recording, change in (("b", 3), ("a", 3), ("c", -1), ("d", 0), ("e", 0)):
            row = {"recording_id": recording, "method": method, "stride": 1,
                   "correct": 90, "total": 100, "identity_switches": 3, "permutation_seed": 2}
            old.append(row)
            new.append({**row, "correct": 90 + change})
    return old, new


def test_case_selection_has_lexicographic_tie_break_and_exact_metric_control():
    old, new = _selection_rows()
    new[-2]["identity_switches"] = 4  # d is no longer an exact control for first_affine.
    chosen = select_diagnostic_cases(old, new)
    assert [(row["method"], row["recording_id"]) for row in chosen] == [
        ("dual_anchor", "a"), ("dual_anchor", "b"), ("dual_anchor", "d"),
        ("first_affine", "a"), ("first_affine", "b"), ("first_affine", "e")]


def test_case_selection_rejects_duplicate_rows():
    old, new = _selection_rows()
    with pytest.raises(ValueError, match="duplicate"):
        select_diagnostic_cases(old + old[:1], new)


def test_case_selection_does_not_treat_switch_only_difference_as_accuracy_mismatch():
    old, new = _selection_rows()
    new[0]["correct"] = old[0]["correct"]
    new[0]["identity_switches"] = 8
    chosen = select_diagnostic_cases(old, new)
    assert [row["recording_id"] for row in chosen[:3]] == ["a", "c", "d"]


def test_known_switch_and_stable_error_fixtures():
    audit = scoring_fixture_audit()
    assert audit["all_passed"]
    assert audit["cases"][1]["observed_switches"] == 2
    assert audit["cases"][1]["observed_correct"] == 6


def test_first_divergence_and_cross_objective_diagnostics(tmp_path):
    a, b = np.asarray([[0, 1], [0, 1], [0, 1]]), np.asarray([[0, 1], [0, 1], [1, 0]])
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    cost = np.asarray([[0., 1.], [1., 0.]])
    np.savez(left, assignments=a, outer0_cost=cost)
    np.savez(right, assignments=b, outer0_cost=cost)
    common = {"status": "complete", "input": {"source_frame_indices": [0, 10, 20]}, "trace": []}
    compared = compare_reconstructed_runs({**common, "arrays_path": str(left)}, {**common, "arrays_path": str(right)})
    assert compared["inputs_identical"]
    assert compared["first_assignment_divergence_source_frame"] == 20
    assert compared["differing_assignment_entries"] == 2
    objectives = compare_captured_costs(left, right, "first_affine")
    assert objectives["left_objective_penalty_of_right"] == 2
    assert objectives["right_objective_penalty_of_left"] == -2
    assert not objectives["same_assignment"]


def test_assignment_hash_has_explicit_integer_canonicalization():
    assert array_sha256(np.asarray([[1, 2]], dtype=np.int32), integer=True) == array_sha256(np.asarray([[1, 2]], dtype=np.int64), integer=True)


def test_observer_restores_wrapped_functions():
    from deformeot import tracker, wormid_benchmark
    before = (tracker.linear_sum_assignment, wormid_benchmark.linear_sum_assignment, wormid_benchmark._affine)
    observed_track(_sequence(), "dual_anchor", ReferenceTrackerConfig())
    assert before == (tracker.linear_sum_assignment, wormid_benchmark.linear_sum_assignment, wormid_benchmark._affine)


def test_changed_source_signature_rejects_checkpoint_without_execution(tmp_path):
    destination = tmp_path / "result.json"
    destination.write_text(json.dumps({"signature": "old", "status": "complete"}))
    with pytest.raises(RuntimeError, match="provenance changed"):
        _run_worker("must-not-execute", {"output_path": str(destination), "signature": "new"}, tmp_path)
