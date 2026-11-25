"""Server-side, development-only phase-two analysis and conservative decisions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .paper_phase2_rescore import file_sha256


BENCHMARK_METHODS = ("pairwise_affine", "first_affine", "dual_anchor", "cpd_reference", "cpd_temporal", "fdnc_reference")
INTEGER_ENDPOINTS = ("correct", "total", "identity_switches", "identity_switch_opportunities",
                     "wrong_identity_frames", "wrong_identity_episode_count",
                     "maximum_wrong_identity_duration_selected_scans")
FLOAT_ENDPOINTS = ("identity_accuracy", "accuracy_excluding_initialization", "identity_switch_rate",
                   "mean_wrong_identity_duration_selected_scans", "coverage")


def _read(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()} for row in rows)


def acquisition_date(recording: str) -> str:
    date = recording.split("_ses-", 1)[1][:8] if "_ses-" in recording else ""
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f"recording has no unambiguous acquisition date: {recording}")
    return date


def _interval(values: np.ndarray) -> dict[str, float]:
    return {"ci95_low": float(np.quantile(values, .025)), "ci95_high": float(np.quantile(values, .975))}


def paired_effect(candidate: Sequence[Mapping[str, Any]], comparator: Sequence[Mapping[str, Any]],
                  *, samples: int = 20_000, seed: int = 20261004,
                  accuracy_field: str = "identity_accuracy", cluster_dates: bool = False) -> dict[str, Any]:
    """Paired animal bootstrap, or whole-date cluster resampling.

    Cluster draws repeat all animals in each sampled date. Means remain weighted
    equally by animal within the resulting sample; animals are not independently
    resampled inside a sampled date. This is sensitivity analysis, not a claim
    that the small set of acquisition dates exhausts laboratory variability.
    """
    if samples < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    left = {str(row["recording_id"]): row for row in candidate}
    right = {str(row["recording_id"]): row for row in comparator}
    if not left or len(left) != len(candidate) or len(right) != len(comparator) or left.keys() != right.keys():
        raise ValueError("paired effect requires unique and exactly matched recording sets")
    names = sorted(left)
    data = np.asarray([[left[name][accuracy_field], right[name][accuracy_field],
                        left[name]["identity_switch_rate"], right[name]["identity_switch_rate"]]
                       for name in names], dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("paired endpoints must be finite")
    rng = np.random.default_rng(seed)
    if cluster_dates:
        dates = sorted({acquisition_date(name) for name in names})
        groups = [np.asarray([index for index, name in enumerate(names) if acquisition_date(name) == date]) for date in dates]
        sums = np.asarray([data[group].sum(axis=0) for group in groups])
        sizes = np.asarray([len(group) for group in groups])
        choices = rng.integers(0, len(groups), size=(samples, len(groups)))
        draws = sums[choices].sum(axis=1) / sizes[choices].sum(axis=1)[:, None]
        resampling_unit = "whole_acquisition_date_cluster"
    else:
        dates = sorted({acquisition_date(name) for name in names})
        choices = rng.integers(0, len(names), size=(samples, len(names)))
        draws = data[choices].mean(axis=1)
        resampling_unit = "paired_recording"
    mean = data.mean(axis=0)
    relative = 1 - mean[2] / mean[3] if mean[3] > 0 else None
    accuracy_difference = float(mean[0] - mean[1])
    switch_benefit = float(mean[3] - mean[2])
    return {"recordings": len(names), "recording_ids": names, "acquisition_dates": dates,
            "accuracy_field": accuracy_field, "bootstrap_samples": samples, "bootstrap_seed": seed,
            "resampling_unit": resampling_unit, "interval_method": "percentile_95_unadjusted_descriptive",
            "candidate_mean_accuracy": float(mean[0]), "comparator_mean_accuracy": float(mean[1]),
            "candidate_mean_switch_rate": float(mean[2]), "comparator_mean_switch_rate": float(mean[3]),
            "accuracy_difference": {"mean": accuracy_difference, **_interval(draws[:, 0] - draws[:, 1])},
            "switch_rate_benefit": {"mean": switch_benefit, **_interval(draws[:, 3] - draws[:, 2])},
            "relative_switch_rate_reduction": float(relative) if relative is not None else None,
            "positive_accuracy_animals": int(np.count_nonzero(data[:, 0] > data[:, 1])),
            "per_recording": [{"recording_id": name, "acquisition_date": acquisition_date(name),
                               "accuracy_difference": float(values[0] - values[1]),
                               "switch_rate_benefit": float(values[3] - values[2])}
                              for name, values in zip(names, data)]}


def practical_gate(effect: Mapping[str, Any], analysis: Mapping[str, Any]) -> dict[str, Any]:
    relative = effect.get("relative_switch_rate_reduction")
    checks = {"relative_switch_reduction_at_least_target": relative is not None and relative >= analysis["minimum_switch_reduction_fraction"],
              "switch_benefit_ci95_lower_positive": effect["switch_rate_benefit"]["ci95_low"] > 0,
              "accuracy_difference_ci95_lower_above_noninferiority_margin": effect["accuracy_difference"]["ci95_low"] > -analysis["maximum_accuracy_loss"]}
    return {"passed": all(checks.values()), "checks": checks,
            "minimum_switch_reduction_fraction": analysis["minimum_switch_reduction_fraction"],
            "maximum_accuracy_loss": analysis["maximum_accuracy_loss"],
            "interpretation": "development feasibility criterion, not independent confirmatory inference"}


def _indexed(rows, phase):
    index, issues = {}, []
    for row in rows:
        if row.get("phase") != phase or row.get("status") != "complete":
            continue
        payload = row.get("payload", {})
        try:
            key = (str(payload["recording_id"]), int(payload["stride"]), str(payload["candidate"]))
        except (KeyError, TypeError, ValueError):
            issues.append({"task_id": row.get("task_id"), "issue": "malformed_scientific_row"})
            continue
        if key in index:
            issues.append({"key": list(key), "issue": "duplicate_scientific_row"})
            index[key] = None
        else:
            index[key] = row
    return index, issues


def _method_means(index, animals, names, strides):
    summaries, missing = [], []
    for name in names:
        for stride in strides:
            selected = [index.get((animal, stride, name)) for animal in animals]
            absent = [animal for animal, row in zip(animals, selected) if row is None]
            if absent:
                missing.append({"candidate": name, "stride": stride, "recordings": absent})
                continue
            payloads = [row["payload"] for row in selected]
            for field in ("identity_accuracy", "accuracy_excluding_initialization", "identity_switch_rate",
                          "mean_wrong_identity_duration_selected_scans", "maximum_wrong_identity_duration_selected_scans", "coverage"):
                if any(payload.get(field) is None or not np.isfinite(payload[field]) for payload in payloads):
                    missing.append({"candidate": name, "stride": stride, "issue": f"nonfinite_or_missing_{field}"})
                    break
            else:
                summaries.append({"candidate": name, "stride": stride, "recordings": len(animals),
                                  **{field: float(np.mean([payload[field] for payload in payloads])) for field in
                                     ("identity_accuracy", "accuracy_excluding_initialization", "identity_switch_rate",
                                      "mean_wrong_identity_duration_selected_scans", "coverage")},
                                  "maximum_wrong_identity_duration_selected_scans": max(payload["maximum_wrong_identity_duration_selected_scans"] for payload in payloads),
                                  "equal_animal_weighting": True})
    return summaries, missing


def benchmark_analysis(rows, animals, config, tuning_date):
    index, issues = _indexed(rows, "benchmark")
    summaries, missing = _method_means(index, animals, BENCHMARK_METHODS, [1])
    expected_keys = {(animal, 1, name) for animal in animals for name in BENCHMARK_METHODS}
    unexpected = sorted(set(index) - expected_keys)
    if unexpected:
        issues.append({"issue": "unexpected_benchmark_cells", "keys": [list(key) for key in unexpected]})
    analysis = config["analysis"]
    candidate, comparator = analysis["candidate"], analysis["primary_comparator"]
    paired = all(index.get((animal, 1, name)) is not None for animal in animals for name in (candidate, comparator))
    output = {"complete": not missing and not issues and len(summaries) == len(BENCHMARK_METHODS),
              "expected_tasks": len(animals) * len(BENCHMARK_METHODS),
              "complete_unique_tasks": sum(row is not None for row in index.values()),
              "method_summaries": summaries, "missing": missing, "issues": issues,
              "primary_comparator": comparator, "primary_available": bool(animals and paired)}
    if not output["primary_available"]:
        return output
    left = [index[(animal, 1, candidate)]["payload"] for animal in animals]
    right = [index[(animal, 1, comparator)]["payload"] for animal in animals]
    kwargs = {"samples": int(analysis["bootstrap_samples"]), "seed": int(analysis["bootstrap_seed"])}
    try:
        effect = paired_effect(left, right, **kwargs)
        excluded_seed = paired_effect(left, right, accuracy_field="accuracy_excluding_initialization", **kwargs)
        cluster = paired_effect(left, right, cluster_dates=True, **kwargs)
        cluster_excluded_seed = paired_effect(left, right, accuracy_field="accuracy_excluding_initialization", cluster_dates=True, **kwargs)
        leave_out = []
        for date in sorted({acquisition_date(animal) for animal in animals}):
            kept = [animal for animal in animals if acquisition_date(animal) != date]
            if not kept:
                continue
            subset = paired_effect([row for row in left if row["recording_id"] in kept],
                                   [row for row in right if row["recording_id"] in kept], **kwargs)
            leave_out.append({"excluded_date": date, "excluded_recordings": [animal for animal in animals if animal not in kept],
                              "effect": subset, "gate": practical_gate(subset, analysis)})
        output.update(primary_effect=effect, primary_gate=practical_gate(effect, analysis),
                      excluding_initialization={"effect": excluded_seed, "gate": practical_gate(excluded_seed, analysis)},
                      date_cluster={"effect": cluster, "gate": practical_gate(cluster, analysis)},
                      date_cluster_excluding_initialization={"effect": cluster_excluded_seed, "gate": practical_gate(cluster_excluded_seed, analysis)},
                      leave_one_date_out=leave_out,
                      tuning_date_exclusion=next((item for item in leave_out if item["excluded_date"] == tuning_date),
                                                {"excluded_date": tuning_date, "not_applicable": True}),
                      sensitivity_all_gates_pass=all(item["gate"]["passed"] for item in leave_out)
                      and practical_gate(excluded_seed, analysis)["passed"] and practical_gate(cluster, analysis)["passed"]
                      and practical_gate(cluster_excluded_seed, analysis)["passed"])
    except (KeyError, TypeError, ValueError) as error:
        output.update(primary_available=False, complete=False, analysis_error=str(error))
    return output


def mechanism_analysis(rows, config):
    setting = config["mechanism"]
    animals, names, strides = setting["recordings"], setting["modes"], setting["strides"]
    index, issues = _indexed(rows, "mechanism")
    means, missing = _method_means(index, animals, names, strides)
    expected_keys = {(animal, stride, name) for animal in animals for stride in strides for name in names}
    unexpected = sorted(set(index) - expected_keys)
    if unexpected:
        issues.append({"issue": "unexpected_mechanism_cells", "keys": [list(key) for key in unexpected]})
    comparisons = []
    pairs = (("beam_marginal", "rolling_best_path"), ("beam_best_path", "rolling_best_path"),
             ("beam_marginal", "beam_best_path"), ("rolling_best_path", "online"), ("online", "dual_anchor"))
    for stride in strides:
        for candidate, comparator in pairs:
            if not all(index.get((animal, stride, name)) for animal in animals for name in (candidate, comparator)):
                continue
            differences = [{"recording_id": animal,
                            "accuracy_difference": index[(animal, stride, candidate)]["payload"]["identity_accuracy"] - index[(animal, stride, comparator)]["payload"]["identity_accuracy"],
                            "switch_rate_benefit": index[(animal, stride, comparator)]["payload"]["identity_switch_rate"] - index[(animal, stride, candidate)]["payload"]["identity_switch_rate"]}
                           for animal in animals]
            comparisons.append({"candidate": candidate, "comparator": comparator, "stride": stride,
                                "mean_accuracy_difference": float(np.mean([item["accuracy_difference"] for item in differences])),
                                "mean_switch_rate_benefit": float(np.mean([item["switch_rate_benefit"] for item in differences])),
                                "positive_accuracy_animals": sum(item["accuracy_difference"] > 0 for item in differences),
                                "per_recording": differences})
    primary = next((item for item in comparisons if item["candidate"] == setting["primary"]
                    and item["comparator"] == setting["comparator"] and item["stride"] == 1), None)
    complete = not missing and not issues and len(means) == len(names) * len(strides)
    checks = {"all_prespecified_controls_complete": complete,
              "native_accuracy_gain_at_least_target": primary is not None and primary["mean_accuracy_difference"] >= setting["minimum_native_accuracy_gain"],
              "no_mean_switch_rate_increase": primary is not None and primary["mean_switch_rate_benefit"] >= 0,
              "positive_in_at_least_required_animals": primary is not None and primary["positive_accuracy_animals"] >= setting["minimum_positive_animals"]}
    gate = all(checks.values())
    if not complete:
        lesson = "Matched-lookahead controls are incomplete; no mechanism conclusion is supported."
    elif gate:
        lesson = "Under the frozen candidate graph and beam settings, the beam-marginal variant exceeded the prespecified native-rate screen versus rolling best path; their path support/pruning and commitment both differ. This is a three-animal development result, not an isolated marginalization effect, universal superiority, or calibrated uncertainty."
    else:
        lesson = "Complete matched-lookahead controls did not meet the prespecified practical screen for the beam-marginal variant versus rolling best path; the recorded pairwise contrasts delimit filtering, lookahead, beam pruning, and commitment effects under these settings."
    return {"complete": complete, "expected_tasks": len(animals) * len(names) * len(strides),
            "complete_unique_tasks": sum(row is not None for row in index.values()),
            "method_summaries": means, "missing": missing, "issues": issues,
            "comparisons": comparisons, "primary_comparison": primary,
            "practical_gate": {"passed": gate, "checks": checks}, "lesson": lesson,
            "interpretation": "Three purposively selected previously examined animals; no small-sample significance claim. Rolling search is exact only over the admitted finite candidate tree, not the unrestricted assignment space."}


def _artifact_path(payload, output):
    path = Path(payload["assignment_file"])
    return path if path.is_absolute() else output / path


def verify_rescoring(rows, output):
    tasks = [row for row in rows if row.get("phase") == "rescore" and row.get("status") == "complete"]
    if len(tasks) != 1:
        return {"passed": False, "issues": ["Exactly one current-signature completed rescore phase is required."]}
    entries = tasks[0].get("payload", {}).get("files", [])
    index, issues = {}, []
    for entry in entries:
        path = Path(entry["path"])
        path = path if path.is_absolute() else output / path
        key = str(path.resolve())
        if key in index:
            issues.append(f"Duplicate rescored artifact: {key}")
        index[key] = entry
    checked = 0
    for row in rows:
        if row.get("status") != "complete" or row.get("phase") not in ("benchmark", "mechanism", "timing"):
            continue
        payload = row["payload"]
        try:
            path = _artifact_path(payload, output)
            fresh = index.get(str(path.resolve()))
            if fresh is None or fresh.get("sha256") != payload.get("assignment_file_sha256") or file_sha256(path) != fresh["sha256"]:
                raise ValueError("artifact missing from independent rescore or SHA256 changed")
            metrics = fresh["metrics"]
            if not all(isinstance(payload.get(field), int) and not isinstance(payload[field], bool)
                       and metrics.get(field) == payload[field] for field in INTEGER_ENDPOINTS):
                raise ValueError("independent integer counts disagree")
            if not all(isinstance(payload.get(field), (int, float)) and isinstance(metrics.get(field), (int, float))
                       and np.isfinite(payload[field]) and np.isfinite(metrics[field])
                       and abs(payload[field] - metrics[field]) <= 1e-12 for field in FLOAT_ENDPOINTS):
                raise ValueError("independent derived accuracy/rate/duration/coverage endpoints disagree")
            if not payload.get("independent_integer_agreement"):
                raise ValueError("original-versus-independent evaluator agreement was not recorded")
            checked += 1
        except (KeyError, ValueError, OSError) as error:
            issues.append(f"{row.get('task_id')}: {error}")
    return {"passed": checked > 0 and not issues, "checked_scientific_artifacts": checked,
            "issues": issues, "float_endpoint_absolute_tolerance": 1e-12,
            "verification_scope": "complete active-signature benchmark/mechanism/timing artifacts; exact integer counts, independently derived endpoints, and NPZ integrity"}


def timing_analysis(rows, output, config):
    setting = config["mechanism"]
    animal, names = setting["median_motion_recording"], setting["timing_modes"]
    mechanisms, _ = _indexed(rows, "mechanism")
    benchmarks, _ = _indexed(rows, "benchmark")
    issues, means, details = [], [], []
    for name in names:
        chosen = [row for row in rows if row.get("phase") == "timing" and row.get("payload", {}).get("candidate") == name]
        pair = []
        for repeat in range(1, int(setting["timing_repeats"]) + 1):
            matches = [row for row in chosen if row.get("payload", {}).get("repeat") == repeat]
            if len(matches) != 1 or matches[0].get("status") != "complete":
                issues.append({"candidate": name, "repeat": repeat, "issue": "missing_failed_or_duplicate_repeat"})
                continue
            row, payload = matches[0], matches[0]["payload"]
            if payload.get("recording_id") != animal or payload.get("stride") != 1 or not isinstance(payload.get("tracker_call_seconds"), (int, float)) or not np.isfinite(payload["tracker_call_seconds"]) or payload["tracker_call_seconds"] < 0:
                issues.append({"candidate": name, "repeat": repeat, "issue": "invalid_timing_metadata"})
                continue
            pair.append(row)
        references = [mechanisms.get((animal, 1, name))]
        if name == "dual_anchor":
            references.append(benchmarks.get((animal, 1, name)))
        identical = False
        if len(pair) == setting["timing_repeats"] and all(references):
            try:
                archive_rows = [*pair, *references]
                arrays = []
                for row in archive_rows:
                    path = _artifact_path(row["payload"], output)
                    if file_sha256(path) != row["payload"]["assignment_file_sha256"]:
                        raise ValueError("assignment file hash changed")
                    with np.load(path, allow_pickle=False) as data:
                        arrays.append({key: data[key].copy() for key in ("assignments", "truth_indices", "frame_indices", "frame_times")})
                identical = all(all(np.array_equal(arrays[0][key], values[key]) for key in arrays[0]) for values in arrays[1:])
            except (KeyError, OSError, ValueError) as error:
                issues.append({"candidate": name, "issue": "array_comparison_failed", "error": str(error)})
            if not identical:
                issues.append({"candidate": name, "issue": "timing_repeats_or_reference_assignments_not_identical"})
        elif len(pair) == setting["timing_repeats"]:
            issues.append({"candidate": name, "issue": "missing_same_case_reference"})
        for row in pair:
            details.append({"candidate": name, "repeat": row["payload"]["repeat"],
                            "tracker_call_seconds": row["payload"]["tracker_call_seconds"],
                            "worker_interval_seconds": row.get("wall_seconds"),
                            "host_load_start": row.get("host_load_start"), "host_load_end": row.get("host_load_end"),
                            "assignment_arrays_equal_across_repeats_and_reference": identical})
        if len(pair) == setting["timing_repeats"] and identical:
            means.append({"candidate": name, "repeats": len(pair),
                          "mean_tracker_call_seconds": float(np.mean([row["payload"]["tracker_call_seconds"] for row in pair])),
                          "assignment_arrays_identical": True})
    dual = next((item for item in means if item["candidate"] == "dual_anchor"), None)
    for item in means:
        item["ratio_to_dual_anchor"] = item["mean_tracker_call_seconds"] / dual["mean_tracker_call_seconds"] if dual and dual["mean_tracker_call_seconds"] > 0 else None
    return {"complete": len(means) == len(names) and not issues,
            "expected_repeats": len(names) * setting["timing_repeats"],
            "collected_metadata_valid_repeats": len(details),
            "valid_repeats": sum(item["assignment_arrays_equal_across_repeats_and_reference"] for item in details),
            "recording_id": animal, "methods": means, "repeats": details, "issues": issues,
            "timing_boundary": "full adapter call including initialization; excludes input loading, independent scoring, and artifact writing",
            "limitations": "Two serial repeats on one development animal; unrelated server load remains uncontrolled. Not a general hardware speedup or a GPU claim."}


def _phase(rows, name):
    found = [row for row in rows if row.get("phase") == name and row.get("status") == "complete"]
    return found[0] if len(found) == 1 else None


def publication_decision(benchmark, mechanism, timing, integrity, evidence, verification_passed, cpd_complete):
    complete = benchmark.get("complete") and mechanism.get("complete") and timing.get("complete") and integrity.get("passed") and verification_passed and cpd_complete and bool(evidence)
    claims = evidence.get("related_work", {}).get("claims", [])
    novelty = any(item.get("verdict") == "supported_distinct_novelty" and item.get("sources") for item in claims)
    external = evidence.get("external_cohort_status") == "eligible" and evidence.get("verified_eligible_new_animals", 0) >= 2
    primary_pass = benchmark.get("primary_gate", {}).get("passed", False)
    mechanism_pass = mechanism.get("practical_gate", {}).get("passed", False)
    robust_primary = primary_pass and benchmark.get("sensitivity_all_gates_pass", False)
    empirical_lesson = ("The equal-animal dual-anchor advantage over the fixed stronger CPD-reference comparator meets the prespecified accuracy/continuity criterion and survives every prespecified acquisition-date and seeded-frame sensitivity. This is a reproducible matched-input development observation, not a new tracking principle or independent biological validation."
                        if complete and robust_primary else
                        "A distinct substantive paper contribution has not been established. Complete controls or a numerical difference alone do not establish novelty or a generalizable mechanism lesson.")
    blockers = []
    for check, message in ((benchmark.get("complete"), "The six-method, twenty-animal native-rate comparison is incomplete."),
                           (mechanism.get("complete"), "Matched-lookahead mechanism controls are incomplete."),
                           (timing.get("complete"), "Serial timing/assignment-repeat verification is incomplete or failed."),
                           (integrity.get("passed"), "Independent rescoring or assignment artifact integrity is incomplete or failed."),
                           (verification_passed, "Tests or adapter fidelity/determinism checks have not fully passed."),
                           (cpd_complete, "The complete expanded CPD tuning grid and both frozen winners are not established."),
                           (bool(evidence), "The current-signature literature and public-asset evidence audit is unavailable."),
                           (novelty, "The reviewed closest literature does not establish a distinct new-method contribution."),
                           (external, "No credible independently verified, previously unused multi-animal external route is established.")):
        if not check:
            blockers.append(message)
    if not complete:
        decision = "inconclusive_technical_report"
    elif primary_pass and novelty and external:
        decision = "methods_followup_candidate_requires_independent_confirmation"
    elif robust_primary:
        decision = "conditional_followup_on_robust_development_lesson"
    else:
        decision = "technical_report_no_distinct_publishable_lesson_established"
    return {"decision": decision, "execution_complete": bool(complete), "blockers": blockers,
            "primary_practical_gate_passed": bool(primary_pass), "mechanism_practical_gate_passed": bool(mechanism_pass),
            "distinct_novelty_supported": novelty, "credible_external_route": external,
            "robust_primary_development_lesson": bool(complete and robust_primary),
            "empirical_lesson": empirical_lesson,
            "sensitivity_judgment": "Requiring every sensitivity gate for the robust-development label is a conservative follow-up judgment, not an additional prespecified primary-screen requirement.",
            "matched_tree_versus_beam_control_is_prior_art": True,
            "publication_ready": False, "independent_confirmation_completed": False,
            "interpretation": "Automated feasibility triage only; a complete comparison or positive development result does not establish publishability."}


def validate_cpd_selection(rows, selected, config, signature, tuning_recording):
    setting = config["cpd"]
    expected = {}
    for anchor in setting["anchors"]:
        for alpha in setting["alpha"]:
            for beta in setting["beta"]:
                expected[f"cpd_{anchor}_a{alpha:g}_b{beta:g}"] = {
                    "anchor": anchor, "order": len(expected), "config": {"alpha": alpha, "beta": beta,
                    "w": setting.get("w", 0.), "max_iterations": setting.get("max_iterations", 100),
                    "tolerance": setting.get("tolerance", 1e-5)}}
    issues, cells = [], {}
    if not signature or selected.get("source_signature") != signature:
        issues.append("CPD selection lacks the active explicit source signature")
    for row in rows:
        if row.get("phase") != "tuning" or row.get("status") != "complete":
            continue
        payload, spec = row.get("payload", {}), row.get("payload", {}).get("specification", {})
        name, stride = payload.get("candidate"), payload.get("stride")
        check = expected.get(name)
        if check is None or stride not in setting["strides"] or payload.get("recording_id") != tuning_recording or spec.get("anchor") != check["anchor"] or spec.get("config") != check["config"]:
            issues.append(f"Unplanned or malformed CPD tuning cell: {row.get('task_id')}")
            continue
        if (name, stride) in cells:
            issues.append(f"Duplicate CPD tuning cell: {name}/{stride}")
        cells[(name, stride)] = payload
    if set(cells) != {(name, stride) for name in expected for stride in setting["strides"]}:
        issues.append("The exact prespecified CPD configuration-by-stride grid is incomplete")
    leaderboard = selected.get("leaderboard", [])
    seen, verified = set(), []
    for item in leaderboard:
        spec = item.get("specification", {})
        name = spec.get("name")
        check = expected.get(name)
        if name in seen or check is None or spec.get("anchor") != check["anchor"] or spec.get("config") != check["config"] or not item.get("complete"):
            issues.append("Invalid or repeated CPD leaderboard configuration")
            continue
        seen.add(name)
        pair = [cells.get((name, stride)) for stride in setting["strides"]]
        if not all(pair):
            continue
        accuracy = float(np.mean([value["identity_accuracy"] for value in pair]))
        rate = float(np.mean([value["identity_switch_rate"] for value in pair]))
        if abs(item.get("accuracy", float("inf")) - accuracy) > 1e-12 or abs(item.get("switch_rate", float("inf")) - rate) > 1e-12:
            issues.append("CPD leaderboard endpoints disagree with current tuning checkpoints")
        verified.append({"name": name, "accuracy": accuracy, "rate": rate, **check})
    if seen != set(expected):
        issues.append("CPD leaderboard does not contain every prespecified configuration exactly once")
    winners = selected.get("selected", selected.get("selected_candidates", []))
    if len(winners) != 2 or {item.get("name") for item in winners} != {"cpd_reference", "cpd_temporal"}:
        issues.append("Both named CPD winners are required exactly once")
    for anchor in setting["anchors"]:
        eligible = [item for item in verified if item["anchor"] == anchor]
        claimed = [item for item in winners if item.get("name") == f"cpd_{anchor}"]
        if eligible and len(claimed) == 1:
            best = min(eligible, key=lambda item: (-item["accuracy"], item["rate"], item["order"]))
            if claimed[0].get("anchor") != anchor or claimed[0].get("config") != best["config"]:
                issues.append(f"CPD {anchor} winner differs from deterministic tuning selection")
    return {"passed": not issues, "issues": issues, "expected_tasks": len(expected) * len(setting["strides"]),
            "verified_tuning_cells": len(cells)}


def make_report(root: str | Path, output: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    context = _read(output / "active_context.json", {})
    signature = context.get("signature")
    all_rows = [_read(path) for path in sorted((output / "checkpoints").glob("*.json"))]
    rows = [row for row in all_rows if signature and row.get("source_signature") == signature]
    excluded = [{"task_id": row.get("task_id"), "phase": row.get("phase"), "status": row.get("status"),
                 "source_signature": row.get("source_signature")} for row in all_rows if row not in rows]
    original = yaml.safe_load((root / config["original_protocol"]).read_text())
    animals = sorted([*original["dataset"]["legacy_evaluation_recordings"], *original["dataset"]["untouched_recordings"]])
    tuning_date = acquisition_date(original["dataset"]["tuning_recording"])
    if len(animals) != 20 or len(set(animals)) != 20:
        raise ValueError("phase-two report requires the twenty prespecified, previously examined animals")
    benchmark = benchmark_analysis(rows, animals, config, tuning_date)
    mechanism = mechanism_analysis(rows, config)
    integrity = verify_rescoring(rows, output)
    timing = timing_analysis(rows, output, config)
    evidence_row = _phase(rows, "evidence")
    evidence = evidence_row.get("payload", {}) if evidence_row else {}
    reviewed = yaml.safe_load((root / "configs/paper_phase2_evidence.yaml").read_text())
    history = yaml.safe_load((root / "configs/paper_phase2_history.yaml").read_text())
    tests, adapters = _phase(rows, "tests"), _phase(rows, "verify_adapters")
    adapter_payload = adapters.get("payload", {}) if adapters else {}
    verification = bool(tests and adapter_payload.get("deterministic_all_methods") and
                        adapter_payload.get("fdnc_released_adapter_check", {}).get("status") == "passed")
    cpd = _read(output / "cpd_selection.json", {})
    winners = cpd.get("selected", cpd.get("selected_candidates", []))
    cpd_audit = validate_cpd_selection(rows, cpd, config, signature, original["dataset"]["tuning_recording"])
    cpd_complete = cpd_audit["passed"]
    boundary = [{"candidate": winner["name"], "alpha": winner["config"]["alpha"], "beta": winner["config"]["beta"],
                 "on_grid_boundary": winner["config"]["alpha"] in (min(config["cpd"]["alpha"]), max(config["cpd"]["alpha"])) or
                 winner["config"]["beta"] in (min(config["cpd"]["beta"]), max(config["cpd"]["beta"]))} for winner in winners
                if isinstance(winner.get("config"), dict) and "alpha" in winner["config"] and "beta" in winner["config"]]
    decision = publication_decision(benchmark, mechanism, timing, integrity, evidence, verification, cpd_complete)
    if not evidence_row:
        decision["blockers"].append("The current-signature public-source evidence audit is missing or incomplete.")
    decision.update(historical_discrepancy_status=history["status"],
                    historical_exact_recovery_required=False, source_signature=signature)
    summary = {"schema_version": 1, "source_signature": signature, "decision": decision,
               "benchmark": benchmark, "mechanism": mechanism, "timing": timing,
               "independent_rescoring": integrity, "verification_passed": verification,
               "cpd_grid_complete": cpd_complete, "cpd_selection_audit": cpd_audit, "cpd_selected_boundary_sensitivity": boundary,
               "historical_archive_search": history, "excluded_prior_signature_checkpoints": excluded,
               "active_failed_tasks": [{"task_id": row.get("task_id"), "phase": row.get("phase"), "status": row.get("status"), "error": row.get("error")}
                                       for row in rows if row.get("status") != "complete"],
               "evidence_status": evidence.get("external_cohort_status", "unavailable"),
               "budget_snapshot": _read(output / "budget.json", {}),
               "external_test_outcomes_scored": False,
               "inference_boundary": "All twenty evaluation animals and all mechanism animals were previously examined; intervals and gates are descriptive development evidence."}
    metrics_rows = [{"phase": row["phase"], "task_id": row.get("task_id"),
                     **{key: row["payload"].get(key) for key in ("recording_id", "candidate", "stride", "repeat", "correct", "total",
                        "identity_accuracy", "accuracy_excluding_initialization", "identity_switches", "identity_switch_rate",
                        "wrong_identity_episode_count", "mean_wrong_identity_duration_selected_scans", "maximum_wrong_identity_duration_selected_scans",
                        "right_endpoint_weighted_wrong_exposure_seconds", "coverage", "tracker_call_seconds", "assignment_file", "assignment_file_sha256")}}
                    for row in rows if row.get("status") == "complete" and row.get("phase") in ("benchmark", "mechanism", "timing")]
    papers = {item["id"]: item for item in reviewed["related_work"]["papers"]}
    closest_papers = [dict(item) for item in reviewed["related_work"]["papers"]]
    inventory_rows = []
    for item in [*evidence.get("assets", []), *evidence.get("supplementary_assets", [])]:
        observation = item.get("observation", {})
        roi = observation.get("temporal_roi", {})
        verification_detail = item.get("reference_verification", {})
        inventory_rows.append({"id": item.get("id"), "family": item.get("family"),
                               "source_url": item.get("source_url", item.get("provenance_source")),
                               "status": item.get("status", "unverified"), "reasons": item.get("reasons", []),
                               "validation_scope": item.get("validation_scope"),
                               "verified_animal_id": item.get("verified_animal_id"),
                               "aliases": item.get("aliases", []), "alias_review": item.get("alias_review", {}),
                               "prior_exposure": item.get("prior_exposure"),
                               "calibration_verified": observation.get("calibration", {}).get("verified"),
                               "time_verified": observation.get("time", {}).get("verified"),
                               "temporal_roi_schema": roi.get("schema"), "complete_temporal_coordinates": roi.get("complete"),
                               "expert_verification_status": verification_detail.get("status"),
                               "release_mapping_verified": verification_detail.get("release_mapping_verified"),
                               "verification_production": verification_detail.get("production"),
                               "license_verified": item.get("license_verified"),
                               "training_overlap": item.get("training_overlap", item.get("comparator_training_overlap")),
                               "comparator_eligibility": item.get("comparator_eligibility", {}),
                               "error": item.get("error")})
    summary["external_asset_inventory"] = inventory_rows
    summary["external_validation_scopes"] = {key: evidence.get(key) for key in
        ("verified_eligible_new_animals", "dense_external_cohort_verified", "sparse_external_validation_available", "validation_scope_counts")}
    claims = [{"claim": item["claim"], "verdict": item["verdict"], "source_ids": item["sources"],
               "source_urls": [papers[source]["url"] for source in item["sources"]],
               "evidence_scope": "reviewed_primary_literature_not_a_universal_novelty_proof"} for item in reviewed["related_work"]["claims"]]
    claims += [{"claim": "Exact original July results must be recovered before a new study can be credible", "verdict": "not_required_historical_discrepancy_closed_unresolved",
                "source_ids": ["historical_archive_search"], "source_urls": [], "evidence_scope": "original artifacts preserved; new assignments and independent evaluator are separately archived"},
               {"claim": "The current six-method development comparison passes the prespecified primary screen", "verdict": "supported_development_only" if decision["primary_practical_gate_passed"] else "not_demonstrated",
                "source_ids": ["phase2_primary_analysis"], "source_urls": [], "evidence_scope": "previously examined animals; no independent confirmation"},
               {"claim": "The beam-marginal variant versus rolling best path meets the matched-lookahead practical screen", "verdict": "supported_development_only" if decision["mechanism_practical_gate_passed"] else "not_demonstrated",
                "source_ids": ["phase2_matched_controls"], "source_urls": [], "evidence_scope": "three-animal bounded-candidate mechanism comparison"}]
    lines = ["# Phase-two paper-worthiness assessment", "", f"Decision: **{decision['decision'].replace('_', ' ')}**.", "",
             "This is development evidence on previously examined recordings, not a newly untouched replication or a submission-ready paper.", "",
             "## Native-rate comparison", "", "| Method | Accuracy | Without seed | Switches / 1000 | Mean wrong episode (scans) | Longest wrong episode (scans) | Coverage | Animals |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in benchmark["method_summaries"]:
        lines.append(f"| {item['candidate']} | {100*item['identity_accuracy']:.3f}% | {100*item['accuracy_excluding_initialization']:.3f}% | {1000*item['identity_switch_rate']:.3f} | {item['mean_wrong_identity_duration_selected_scans']:.2f} | {item['maximum_wrong_identity_duration_selected_scans']} | {100*item['coverage']:.1f}% | {item['recordings']} |")
    lines += ["", f"Completed unique comparison tasks: {benchmark['complete_unique_tasks']} / {benchmark['expected_tasks']}. Each animal has equal weight; CPD-reference is the fixed primary comparator."]
    if benchmark.get("primary_effect"):
        effect = benchmark["primary_effect"]
        acc, switch = effect["accuracy_difference"], effect["switch_rate_benefit"]
        relative = effect["relative_switch_rate_reduction"]
        lines += [f"Dual-anchor accuracy difference: {100*acc['mean']:+.3f} percentage points (95% paired-animal bootstrap interval {100*acc['ci95_low']:+.3f} to {100*acc['ci95_high']:+.3f}).",
                  f"Absolute switch-rate benefit: {1000*switch['mean']:+.3f} per 1000 transitions (95% interval {1000*switch['ci95_low']:+.3f} to {1000*switch['ci95_high']:+.3f}).",
                  f"Relative mean switch-rate reduction: {100*relative:+.1f}%." if relative is not None else "Relative switch reduction is undefined because the comparator has zero switches.",
                  f"Prespecified practical screen passed: `{benchmark['primary_gate']['passed']}`. All acquisition-date/initialization sensitivity gates passed: `{benchmark['sensitivity_all_gates_pass']}`.",
                  "", "The screen requires at least 25% relative switch reduction, a positive lower 95% bound on switch benefit, and an accuracy-difference lower bound above −0.25 percentage points. Intervals use 20,000 paired draws with seed 20261004 and are unadjusted descriptive intervals, not confirmatory significance claims.",
                  "Whole-acquisition-date bootstrap, every leave-one-date-out analysis, exclusion of the tuning-adjacent acquisition date, and exclusion of the seeded frame are retained in `summaries.json`; date clusters repeat all their animals, with equal-animal means in each bootstrap sample."]
        lines += [decision["sensitivity_judgment"]]
    lines += ["Coverage refers only to the initialized exhaustive supplied-ROI cohort, not detector recall or coverage of every biological neuron. The mean wrong-episode duration is an equal-animal mean of each animal's episode mean; the longest episode is the maximum across animals. Low switch counts alone can conceal persistent wrong identities."]
    lines += ["", "Expanded CPD tuning uses only the original tuning animal. Boundary winners are flagged in the summary; a finite grid does not establish a globally optimal registration baseline. fDNC is a coordinate-only released-model adapter: model-data overlap and transfer limits remain explicit, and this is not a ranking of complete image-based tracking pipelines.",
              "", "## Matched-information mechanism", "", mechanism["lesson"], "",
              f"Complete unique controls: {mechanism['complete_unique_tasks']} / {mechanism['expected_tasks']}. Practical screen passed: `{mechanism['practical_gate']['passed']}`.", "",
              "| Candidate vs comparator | Stride | Accuracy difference (pp) | Switch benefit / 1000 | Positive animals |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for item in mechanism["comparisons"]:
        lines.append(f"| {item['candidate']} vs {item['comparator']} | {item['stride']} | {100*item['mean_accuracy_difference']:+.3f} | {1000*item['mean_switch_rate_benefit']:+.3f} | {item['positive_accuracy_animals']} / 3 |")
    lines += ["", "The primary mechanism screen requires at least +0.25 accuracy points at native rate, no mean switch-rate increase, and positive accuracy differences in at least two animals. Rolling-path and beam modes share two future selected observations; final windows are shortened. Rolling optimization is exact only over its admitted candidate tree. These controls do not establish calibrated posterior probabilities or an extended-object benefit.",
              "The selected configuration uses geometry costs, so beam weights are normalized geometry scores rather than validated likelihoods or calibrated identity probabilities. The primary beam-marginal versus rolling-path contrast jointly changes retained/pruned path support and commitment; its gain cannot be attributed solely to marginalization. Beam-best versus rolling is not a pure retention ablation either. Beam-marginal versus beam-best is the narrower commitment contrast, after which their trajectories can diverge.",
              "", "## Timing and assignment reproducibility", "",
              f"Verified serial repeats: {timing['valid_repeats']} / {timing['expected_repeats']}; complete timing/array check: `{timing['complete']}`.", "",
              "| Method | Mean full adapter call (s) | Ratio to dual anchor |", "| --- | ---: | ---: |"]
    for item in timing["methods"]:
        ratio = f"{item['ratio_to_dual_anchor']:.2f}" if item["ratio_to_dual_anchor"] is not None else "unavailable"
        lines.append(f"| {item['candidate']} | {item['mean_tracker_call_seconds']:.3f} | {ratio} |")
    lines += ["", timing["timing_boundary"] + ". " + timing["limitations"],
              "Exact saved assignments, truth permutations, frame indices, and timestamps are compared across timing repeats and their same-case scientific reference runs. Independent rescoring checks full-coverage permutations and integer endpoints, not biological label validity.",
              "", "## Publication decision and unresolved evidence", ""]
    lines += [f"- {message}" for message in decision["blockers"]] or ["- No automatic readiness blocker; independent confirmation is still required."]
    lines += ["", "### Public external-data eligibility", "",
              "| Asset | Status | Verified scope | Main unresolved reasons |",
              "| --- | --- | --- | --- |"]
    for item in inventory_rows:
        reasons = "; ".join(str(value) for value in item["reasons"]) or "None recorded"
        lines.append(f"| {item['id']} | {item['status']} | {item['validation_scope'] or 'unverified'} | {reasons.replace('|', '/')} |")
    lines += ["", "`eligibility_inventory.csv` retains alias, calibration/time, expert-verification/release mapping, licensing, and comparator-specific training-overlap checks. Unknown provenance means unverified, not that usable labels do not exist. Sparse documented expert-verified subsets may support scoped validation; they do not establish dense all-frame identity truth. Expert-checked algorithm-assisted references are eligible only with exact release/subset mapping. Unknown fDNC training overlap does not by itself disqualify an untrained geometric comparator.",
              "The ten closest primary papers and their anchoring, deformation, future-frame access, hypothesis retention, annotation requirements, and evaluation scope are tabulated in `closest_papers.csv`."]
    lines += ["", decision["empirical_lesson"], "",
              "The closest-paper claim audit is in `claim_evidence.csv`. A quantitative comparison alone does not establish a novel tracking principle. MHHT already contrasts depth-first ranked-assignment trees with retained K-best hypotheses; the rolling-versus-beam control is therefore prior art. Complete controls alone do not justify a benchmark or mechanism paper. A robust practical development lesson can motivate conditional follow-up, with novelty and external validation still unresolved; otherwise the outcome remains a technical report.",
              "", "## Historical closure and scope", "",
              "One bounded search found the July aggregate results, source manifest, and environment record, but no original per-frame assignment arrays or full BLAS/thread record. The discrepancy remains unexplained; the October reconstruction is not the July execution. Original numbers remain preserved and qualified. Exact July recovery is retired as a prerequisite for a newly frozen study; it is not claimed solved.",
              "All current analyses are restricted to the active source signature. Older attempts and failures remain in the checkpoint archive and are listed separately, never silently pooled. Fresh NPZ assignments and an independently implemented scorer permit exact rescoring of this execution.",
              "Accuracy measures agreement with algorithm-assisted, expert-verified supplied ROI trajectories, not independently adjudicated identity at every frame. No external test identities were scored. No neuroscience efficacy, downstream calcium recovery, detection-gap, merge/split, extended-object, or calibrated uncertainty claim is supported by this phase.",
              "", "Machine-readable effects, sensitivity analyses, completion issues, integrity checks, and resource snapshots are in `summaries.json`; per-case endpoints are in `metrics.csv`; the feasibility decision is in `decision.json`.", ""]
    destinations = [output, root / "outputs/paper_phase2_20261004"]
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        _write(destination / "summaries.json", summary)
        _write(destination / "decision.json", decision)
        _csv(destination / "metrics.csv", metrics_rows)
        _csv(destination / "claim_evidence.csv", claims)
        _csv(destination / "eligibility_inventory.csv", inventory_rows)
        _csv(destination / "closest_papers.csv", closest_papers)
        (destination / "report.md").write_text("\n".join(lines))
    return {"report": str(destinations[-1] / "report.md"), "decision": decision["decision"],
            "source_signature": signature, "execution_complete": decision["execution_complete"],
            "primary_practical_gate_passed": decision["primary_practical_gate_passed"],
            "mechanism_practical_gate_passed": decision["mechanism_practical_gate_passed"]}
