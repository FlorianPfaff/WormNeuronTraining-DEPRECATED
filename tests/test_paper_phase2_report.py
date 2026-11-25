from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import yaml

from deformeot.paper_phase2_report import (
    BENCHMARK_METHODS, benchmark_analysis, make_report, mechanism_analysis,
    paired_effect, practical_gate, publication_decision, timing_analysis,
    validate_cpd_selection, verify_rescoring,
)


ANIMALS = [f"sub-202001{index % 4 + 1:02d}-{index:02d}_ses-202001{index % 4 + 1:02d}_ophys" for index in range(20)]


def configuration():
    return {"original_protocol": "configs/original.yaml",
            "analysis": {"bootstrap_samples": 100, "bootstrap_seed": 20261004,
                         "candidate": "dual_anchor", "primary_comparator": "cpd_reference",
                         "minimum_switch_reduction_fraction": .25, "maximum_accuracy_loss": .0025},
            "mechanism": {"recordings": ANIMALS[:3], "modes": ["dual_anchor", "online", "rolling_best_path", "beam_best_path", "beam_marginal"],
                          "strides": [1, 10], "timing_modes": ["dual_anchor", "rolling_best_path", "beam_best_path", "beam_marginal"],
                          "timing_repeats": 2, "median_motion_recording": ANIMALS[1],
                          "primary": "beam_marginal", "comparator": "rolling_best_path",
                          "minimum_native_accuracy_gain": .0025, "minimum_positive_animals": 2},
            "cpd": {"anchors": ["reference", "temporal"], "alpha": [2., 5.], "beta": [1., 2.], "strides": [1, 10, 40]}}


def row(phase, animal, candidate, stride=1, accuracy=.9, rate=.02, repeat=None, signature="current"):
    return {"task_id": f"{phase}__{animal}__{candidate}__s{stride}__r{repeat}", "source_signature": signature,
            "phase": phase, "status": "complete", "wall_seconds": 3.,
            "payload": {"recording_id": animal, "candidate": candidate, "stride": stride, "repeat": repeat,
                        "identity_accuracy": accuracy, "accuracy_excluding_initialization": accuracy - .001,
                        "identity_switch_rate": rate, "mean_wrong_identity_duration_selected_scans": 3.,
                        "maximum_wrong_identity_duration_selected_scans": 7, "coverage": 1.,
                        "tracker_call_seconds": 2., "correct": 90, "total": 100,
                        "identity_switches": 4, "identity_switch_opportunities": 98,
                        "wrong_identity_frames": 10, "wrong_identity_episode_count": 2,
                        "independent_integer_agreement": True}}


def benchmark_rows():
    return [row("benchmark", animal, name, accuracy=.95 if name == "dual_anchor" else .9,
                rate=.01 if name == "dual_anchor" else .02) for animal in ANIMALS for name in BENCHMARK_METHODS]


def test_paired_and_cluster_bootstraps_preserve_pairs_and_equal_animal_effects():
    left = [row("benchmark", animal, "dual_anchor", accuracy=.95, rate=.01)["payload"] for animal in ANIMALS]
    right = [row("benchmark", animal, "cpd_reference", accuracy=.9, rate=.02)["payload"] for animal in ANIMALS]
    standard = paired_effect(left, right, samples=100)
    assert standard == paired_effect(left[::-1], right[3:] + right[:3], samples=100)
    cluster = paired_effect(left, right, samples=100, cluster_dates=True)
    assert standard["accuracy_difference"]["mean"] == pytest.approx(.05)
    assert standard["switch_rate_benefit"]["ci95_low"] == pytest.approx(.01)
    assert standard["relative_switch_rate_reduction"] == pytest.approx(.5)
    assert cluster["accuracy_difference"]["ci95_high"] == pytest.approx(.05)
    assert cluster["resampling_unit"] == "whole_acquisition_date_cluster"
    assert len(cluster["acquisition_dates"]) == 4
    with pytest.raises(ValueError, match="unique and exactly matched"):
        paired_effect([left[0], left[0]], [right[0]], samples=100)


def test_practical_gate_uses_strict_interval_bounds_and_inclusive_relative_target():
    config = configuration()["analysis"]
    effect = {"relative_switch_rate_reduction": .25,
              "accuracy_difference": {"ci95_low": -.0024}, "switch_rate_benefit": {"ci95_low": .0001}}
    assert practical_gate(effect, config)["passed"]
    effect["accuracy_difference"]["ci95_low"] = -.0025
    assert not practical_gate(effect, config)["passed"]
    effect["accuracy_difference"]["ci95_low"] = 0
    effect["switch_rate_benefit"]["ci95_low"] = 0
    assert not practical_gate(effect, config)["passed"]


def test_full_benchmark_includes_seed_date_cluster_and_adjacent_date_sensitivities():
    result = benchmark_analysis(benchmark_rows(), ANIMALS, configuration(), "20200101")
    assert result["complete"] and result["complete_unique_tasks"] == 120
    assert result["primary_gate"]["passed"] and result["sensitivity_all_gates_pass"]
    assert len(result["leave_one_date_out"]) == 4
    assert len(result["tuning_date_exclusion"]["excluded_recordings"]) == 5
    assert result["excluding_initialization"]["effect"]["accuracy_field"] == "accuracy_excluding_initialization"
    assert result["date_cluster_excluding_initialization"]["effect"]["resampling_unit"] == "whole_acquisition_date_cluster"


def test_missing_duplicate_and_unplanned_benchmark_cells_fail_completion():
    values = benchmark_rows()
    for variant in (values[:-1], values + [values[0]], values + [row("benchmark", ANIMALS[0], "dual_anchor", stride=10)]):
        assert not benchmark_analysis(variant, ANIMALS, configuration(), "20200101")["complete"]


def test_mechanism_requires_all_controls_accuracy_switch_and_animal_gates():
    config = configuration()
    setting = config["mechanism"]
    values = [row("mechanism", animal, name, stride, accuracy=.91 if name == "beam_marginal" else .90)
              for animal in setting["recordings"] for stride in setting["strides"] for name in setting["modes"]]
    result = mechanism_analysis(values, config)
    assert result["complete_unique_tasks"] == 30
    assert result["complete"] and result["practical_gate"]["passed"]
    assert result["primary_comparison"]["positive_accuracy_animals"] == 3
    assert not mechanism_analysis(values[:-1], config)["practical_gate"]["passed"]
    for value in values:
        if value["payload"]["candidate"] == "beam_marginal":
            value["payload"]["identity_switch_rate"] = .03
    assert not mechanism_analysis(values, config)["practical_gate"]["passed"]


def attach_archive(tmp_path, value, *, swapped=False):
    path = tmp_path / f"{value['task_id']}.npz"
    assignments = np.asarray([[0, 1], [1, 0] if swapped else [0, 1]], dtype=np.int64)
    np.savez(path, assignments=assignments, truth_indices=np.asarray([[0, 1], [0, 1]], dtype=np.int64),
             frame_indices=np.asarray([0, 1], dtype=np.int64), frame_times=np.asarray([0., 1.]))
    value["payload"].update(assignment_file=str(path), assignment_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return value


def test_timing_compares_actual_arrays_and_does_not_count_failed_array_checks_as_verified(tmp_path):
    config = configuration()
    animal = config["mechanism"]["median_motion_recording"]
    values = []
    for name in config["mechanism"]["timing_modes"]:
        values.append(attach_archive(tmp_path, row("mechanism", animal, name)))
        values.extend(attach_archive(tmp_path, row("timing", animal, name, repeat=repeat)) for repeat in (1, 2))
    values.append(attach_archive(tmp_path, row("benchmark", animal, "dual_anchor")))
    result = timing_analysis(values, tmp_path, config)
    assert result["complete"] and result["valid_repeats"] == 8
    changed = next(value for value in values if value["phase"] == "timing" and value["payload"]["candidate"] == "beam_marginal")
    attach_archive(tmp_path, changed, swapped=True)
    result = timing_analysis(values, tmp_path, config)
    assert not result["complete"]
    assert result["collected_metadata_valid_repeats"] == 8 and result["valid_repeats"] == 6


def test_rescoring_rejects_changed_counts_and_stale_derived_metrics(tmp_path):
    value = attach_archive(tmp_path, row("benchmark", ANIMALS[0], "dual_anchor"))
    entry = {"path": value["payload"]["assignment_file"], "sha256": value["payload"]["assignment_file_sha256"],
             "metrics": copy.deepcopy(value["payload"])}
    audit = {"phase": "rescore", "status": "complete", "payload": {"files": [entry]}}
    assert verify_rescoring([value, audit], tmp_path)["passed"]
    entry["metrics"]["identity_switches"] += 1
    assert not verify_rescoring([value, audit], tmp_path)["passed"]
    entry["metrics"]["identity_switches"] -= 1
    entry["metrics"]["identity_accuracy"] += .01
    assert not verify_rescoring([value, audit], tmp_path)["passed"]
    assert not verify_rescoring([value], tmp_path)["passed"]


def test_complete_controls_alone_do_not_trigger_a_paper_decision():
    benchmark = {"complete": True, "primary_gate": {"passed": True}, "sensitivity_all_gates_pass": False}
    mechanism = {"complete": True, "practical_gate": {"passed": True}}
    evidence = {"related_work": {"claims": []}, "external_cohort_status": "conditional"}
    result = publication_decision(benchmark, mechanism, {"complete": True}, {"passed": True}, evidence, True, True)
    assert result["decision"] == "technical_report_no_distinct_publishable_lesson_established"
    benchmark["sensitivity_all_gates_pass"] = True
    result = publication_decision(benchmark, mechanism, {"complete": True}, {"passed": True}, evidence, True, True)
    assert result["decision"] == "conditional_followup_on_robust_development_lesson"
    assert not result["publication_ready"] and not result["distinct_novelty_supported"]


def test_methods_route_needs_novelty_external_route_but_not_positive_mht_screen():
    benchmark = {"complete": True, "primary_gate": {"passed": True}, "sensitivity_all_gates_pass": True}
    mechanism = {"complete": True, "practical_gate": {"passed": False}}
    evidence = {"related_work": {"claims": [{"verdict": "supported_distinct_novelty", "sources": ["review"]}]},
                "external_cohort_status": "eligible", "verified_eligible_new_animals": 2}
    result = publication_decision(benchmark, mechanism, {"complete": True}, {"passed": True}, evidence, True, True)
    assert result["decision"] == "methods_followup_candidate_requires_independent_confirmation"
    mechanism["complete"] = False
    assert publication_decision(benchmark, mechanism, {"complete": True}, {"passed": True}, evidence, True, True)["decision"] == "inconclusive_technical_report"


def cpd_fixture():
    config, rows, board, winners = configuration(), [], [], []
    setting = config["cpd"]
    for anchor in setting["anchors"]:
        for alpha in setting["alpha"]:
            for beta in setting["beta"]:
                name = f"cpd_{anchor}_a{alpha:g}_b{beta:g}"
                spec = {"name": name, "method": "cpd", "anchor": anchor,
                        "config": {"alpha": alpha, "beta": beta, "w": 0., "max_iterations": 100, "tolerance": 1e-5}}
                board.append({"specification": spec, "complete": True, "accuracy": .9, "switch_rate": .02})
                for stride in setting["strides"]:
                    value = row("tuning", "tuning", name, stride)
                    value["payload"]["specification"] = spec
                    rows.append(value)
                if alpha == setting["alpha"][0] and beta == setting["beta"][0]:
                    winners.append({**spec, "name": f"cpd_{anchor}"})
    return rows, {"source_signature": "current", "leaderboard": board, "selected": winners}, config


def test_cpd_requires_exact_grid_active_signature_and_deterministic_winners():
    rows, selected, config = cpd_fixture()
    assert validate_cpd_selection(rows, selected, config, "current", "tuning")["passed"]
    selected["selected"][0] = {**selected["selected"][0], "config": {**selected["selected"][0]["config"], "alpha": 5.}}
    assert not validate_cpd_selection(rows, selected, config, "current", "tuning")["passed"]
    rows, selected, config = cpd_fixture()
    selected.pop("source_signature")
    assert not validate_cpd_selection(rows, selected, config, "current", "tuning")["passed"]
    rows, selected, config = cpd_fixture()
    assert not validate_cpd_selection(rows[:-1], selected, config, "current", "tuning")["passed"]


def test_report_filters_old_signatures_and_writes_inventory_and_six_dimension_review(tmp_path):
    root, output = tmp_path, tmp_path / "results"
    (root / "configs").mkdir()
    (output / "checkpoints").mkdir(parents=True)
    config = configuration()
    (root / config["original_protocol"]).write_text(yaml.safe_dump({"dataset": {
        "legacy_evaluation_recordings": ANIMALS[:4], "untouched_recordings": ANIMALS[4:],
        "tuning_recording": "sub-20200101-99_ses-20200101_ophys"}}))
    paper = {"id": "prior", "title": "Existing method", "url": "https://example.org/prior",
             **{key: "documented" for key in ("anchoring", "deformation", "future_frame_access", "hypothesis_retention", "annotation_requirements", "evaluation_scope")}}
    reviewed = {"related_work": {"papers": [paper], "claims": [{"claim": "New mechanism", "verdict": "not_established", "sources": ["prior"]}]}}
    (root / "configs/paper_phase2_evidence.yaml").write_text(yaml.safe_dump(reviewed))
    (root / "configs/paper_phase2_history.yaml").write_text(yaml.safe_dump({"status": "closed_unresolved_historical_discrepancy"}))
    (output / "active_context.json").write_text(json.dumps({"signature": "current"}))
    (output / "checkpoints/old.json").write_text(json.dumps(row("benchmark", ANIMALS[0], "dual_anchor", signature="old")))
    (output / "checkpoints/failed.json").write_text(json.dumps({"task_id": "failed", "source_signature": "current", "phase": "mechanism", "status": "failed", "error": "fixture"}))
    evidence = {"task_id": "evidence", "phase": "evidence", "source_signature": "current", "status": "complete",
                "payload": {**reviewed, "assets": [{"id": "asset1", "status": "conditional", "reasons": ["time unverified"]}], "external_cohort_status": "conditional"}}
    (output / "checkpoints/evidence.json").write_text(json.dumps(evidence))
    assert make_report(root, output, config)["decision"] == "inconclusive_technical_report"
    summary = json.loads((output / "summaries.json").read_text())
    assert len(summary["excluded_prior_signature_checkpoints"]) == 1
    assert summary["active_failed_tasks"][0]["task_id"] == "failed"
    assert summary["benchmark"]["complete_unique_tasks"] == 0
    assert summary["decision"]["historical_exact_recovery_required"] is False
    for name in ("report.md", "decision.json", "metrics.csv", "claim_evidence.csv", "summaries.json", "eligibility_inventory.csv", "closest_papers.csv"):
        assert (output / name).exists() and (root / "outputs/paper_phase2_20261004" / name).exists()
    report = (output / "report.md").read_text()
    assert "Mean wrong episode" in report and "Coverage" in report
    assert "geometry scores" in report and "control is therefore prior art" in report
