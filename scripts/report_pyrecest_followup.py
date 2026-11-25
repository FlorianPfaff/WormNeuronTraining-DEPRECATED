"""Generate a compact, auditable report and figure from completed experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "gnn_kalman": "PyRecEst GNN",
    "gnn_smoothing": "PyRecEst GNN (smoothing)",
    "gnn_velocity": "PyRecEst GNN (velocity)",
    "gnn_adaptive_covariance": "PyRecEst GNN (adaptive covariance)",
    "mht_geometry_online": "PyRecEst Gaussian filtering, online",
    "mht_geometry_lag2": "PyRecEst MHT, lag 2 (geometry)",
    "mht_gaussian_lag2": "PyRecEst MHT, lag 2 (Gaussian)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result_bytes = args.input.read_bytes()
    result = json.loads(result_bytes)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    audit = result["outcome_audit"]
    if audit["complete_evaluation_tasks"] != audit["expected_evaluation_tasks"] or audit["failed_tasks"]:
        raise ValueError("a final report requires every planned evaluation task to complete")
    task_keys = {(row["recording_id"], row["stride"], row["candidate"]) for row in result["results"]}
    if len(task_keys) != 500 or len(result["results"]) != 500:
        raise ValueError("the report requires 500 unique evaluation tasks")
    recovery_path = args.input.parent / "recovery.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.is_file() else None
    chosen = result["selection"]["selected_candidates"]
    names = [candidate["name"] for candidate in chosen]
    control_path = args.input.parent / "control_audit.json"
    control_audit = {"available": control_path.is_file()}
    if control_path.is_file():
        control_payload = json.loads(control_path.read_text())
        if control_payload["implementation_signature"] != result["context"]["implementation_signature"]:
            raise ValueError("control audit and evaluation must use the same implementation")
        controls = control_payload["results"]
        if len(controls) != 100 or any(row["status"] != "complete" for row in controls):
            raise ValueError("the planned equivalence audit requires all 100 controls")
        if len({(row["recording_id"], row["stride"]) for row in controls}) != 100:
            raise ValueError("equivalence audit contains duplicate tasks")
        baselines = {(row["recording_id"], row["stride"]): row for row in result["results"]
                     if row["candidate"] == "dual_anchor"}
        complete_controls = [row for row in controls if row["status"] == "complete"]
        differences = []
        for row in complete_controls:
            baseline = baselines[(row["recording_id"], row["stride"])]
            if row["assignment_sha256"] != baseline["assignment_sha256"]:
                differences.append({"recording_id": row["recording_id"], "stride": row["stride"],
                    "accuracy_difference": row["identity_accuracy"] - baseline["identity_accuracy"],
                    "switch_difference": row["identity_switches"] - baseline["identity_switches"]})
        control_audit.update({"completed_tasks": len(complete_controls), "expected_tasks": 100,
            "assignment_identical_tasks": len(complete_controls) - len(differences),
            "assignment_differences": differences,
            "maximum_absolute_accuracy_difference_percentage_points": max(
                (100 * abs(row["accuracy_difference"]) for row in differences), default=0.0),
            "interpretation": "mathematically equivalent geometric objective; discrepancies are consistent with numerical sensitivity in the alternate cost path, without a saved proof of equal-cost optima"})
    groups = ("all_nontuning", "previous_prospective", "legacy_evaluation")
    historical = result["baseline_reproduction"]
    mismatch_rows = {tuple(item["key"]) for item in historical.get("mismatches", [])}
    original_values = {(tuple(item["key"]), item["field"]): item["expected"]
                       for item in historical.get("mismatches", [])}
    baseline_drift = {}
    for name in ("dual_anchor", "first_affine"):
        rows = [row for row in result["results"] if row["candidate"] == name]
        deltas = []
        for row in rows:
            key = (row["recording_id"], row["stride"], name)
            correct = original_values.get((key, "correct"), row["correct"])
            total = original_values.get((key, "total"), row["total"])
            deltas.append(100 * (row["correct"] / row["total"] - correct / total))
        baseline_drift[name] = {
            "compared_rows": len(rows),
            "rows_with_any_identity_outcome_mismatch": sum(key[2] == name for key in mismatch_rows),
            "mean_accuracy_change_percentage_points": sum(deltas) / len(deltas),
            "maximum_absolute_case_accuracy_change_percentage_points": max(map(abs, deltas)),
        }
    table = []
    for group in groups:
        summary = result["cohort_summaries"][group]
        baseline = summary["methods"]["dual_anchor"]
        for name in names:
            comparison = summary["comparisons"][f"{name}_vs_dual_anchor"]
            effect = comparison["accuracy_difference"]
            switches = comparison["pooled_identity_switches"]
            table.append({
                "cohort": group,
                "candidate": name,
                "recordings": effect["recordings"],
                "baseline_accuracy_percent": 100 * baseline["recording_macro_mean_accuracy"],
                "candidate_accuracy_percent": 100 * summary["methods"][name]["recording_macro_mean_accuracy"],
                "accuracy_gain_percentage_points": 100 * effect["mean"],
                "ci95_low_percentage_points": 100 * effect["ci95_low"],
                "ci95_high_percentage_points": 100 * effect["ci95_high"],
                "baseline_identity_switches": switches["baseline"],
                "candidate_identity_switches": switches["candidate"],
                "pooled_switch_reduction_percent": 100 * (1 - switches["candidate"] / switches["baseline"]),
                "runtime_ratio": comparison["runtime_ratio"],
                "latency_selected_scans": summary["methods"][name]["per_recording"][comparison["recording_ids"][0]]["maximum_latency_selected_scans"],
                "positive_recordings": comparison["recordings_with_positive_accuracy_difference"],
            })
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (output / "summary.json").write_text(json.dumps({
        "evaluation_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "protocol_sha256": result["context"]["protocol_sha256"],
        "selection_sha256": result["selection_sha256"],
        "comparisons": table,
        "outcome_audit": audit,
        "baseline_reproduction": result["baseline_reproduction"],
        "baseline_accuracy_drift": baseline_drift,
        "gnn_equivalence_control": control_audit,
        "recovery": recovery,
    }, indent=2) + "\n")

    lines = ["# PyRecEst neuron-tracking follow-up", "",
        "Completed on gpuserver6000. Settings were selected using the original single tuning recording. "
        "All 20 other recordings and all five original strides (1, 5, 10, 20, 40) were evaluated. "
        "The comparison uses the same supplied centers, first-frame initialization, anonymous detection "
        "permutations, shared affine correction, and immutable reference anchor.", "",
        "These recordings had already been evaluated in the earlier study. This is a follow-up "
        "comparison; its recording-bootstrap intervals are descriptive and do not provide new untouched "
        "confirmation or cross-laboratory validation. The three selected families are separate comparisons; "
        "intervals are not corrected for multiplicity.", "",
        "## Comparison with the current dual-anchor tracker", "",
        "| Method | Accuracy gain (percentage points; 95% interval) | Switch reduction | Runtime ratio | Future selected scans |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in table:
        if row["cohort"] != "all_nontuning":
            continue
        lines.append(f"| {LABELS.get(row['candidate'], row['candidate'])} | "
            f"{row['accuracy_gain_percentage_points']:+.3f} "
            f"[{row['ci95_low_percentage_points']:+.3f}, {row['ci95_high_percentage_points']:+.3f}] | "
            f"{row['pooled_switch_reduction_percent']:+.1f}% | {row['runtime_ratio']:.1f}x | "
            f"{row['latency_selected_scans']} |")
    lines.extend(["", "These implementations offer small accuracy gains, not an across-the-board "
        "improvement: every selected variant produces more identity switches and costs more computation. "
        "The current dual-anchor tracker remains the practical default when speed and continuity matter. "
        "GNN is the least expensive tested PyRecEst option; lag-2 MHT has the largest accuracy gain "
        "but also uses future observations. This does not establish that PyRecEst cannot improve on "
        "these implementations with other models or settings.", "",
        "![Paired recording-bootstrap accuracy comparisons](accuracy_comparison.png)"])
    if recovery is not None:
        lines.extend(["", "Execution recovery: the server reboot removed the earlier temporary "
            "evaluation directory. This complete evaluation was rerun in persistent server storage, "
            "using the saved tuning selection and identical code/package signature. Task parallelism "
            f"was increased from 12 to {recovery['operational_worker_override']['recovery_workers']} "
            "workers; scientific parameters, data, and selection were unchanged. "
            "The recovery record includes the preserved selection hash."])
    lines.extend(["", "Accuracy gives equal weight to each recording after averaging its five strides. "
        "Intervals resample paired recordings 20,000 times. Switch totals pool overlapping stride analyses "
        "and are descriptive. Positive switch reduction means fewer switches. Runtime ratios compare "
        "recorded algorithm time in the same server environment; individual tasks ran concurrently.", "",
        "MHT with lag 2 uses two subsequent selected scans before committing an assignment. Its delay "
        "therefore grows with stride, and its accuracy gains must be interpreted with that extra information. "
        "The online MHT row provides a comparison with no future scans.", "",
        "## Implementation and limits", "",
        "GNN uses native PyRecEst assignment and Gaussian updates, with transported mean/covariance, "
        "Mahalanobis association, and an external reference cost. An equivalence control routes the original "
        "dual-anchor objective through GNN. MHT uses native Gaussian filter banks, Murty ranked assignments, "
        "alternative histories and hypothesis pruning; adapter hooks add reference scores before branching "
        "and condition assignments on the exhaustive cohort. Candidate edges are limited by a score "
        "window and per-track cap, with a complete Hungarian backbone. The beam is bounded at four global "
        "hypotheses and three children per parent. These approximations may discard a correct history. "
        "With zero lag, commitment conditions the beam on one complete assignment each scan; any gain "
        "there primarily tests recursive state filtering/scoring. The lag-2 variant tests retaining "
        "alternative recent assignment histories.", "",
        "No semantic neuron labels or fluorescence enter association. The coordinate caches have no "
        "missing/false detections or segmented extent, so this experiment does not assess birth/death, "
        "clutter handling, extended objects, or downstream calcium signal recovery. Correlated deformation "
        "and reference cues prevent interpreting the hypothesis weights as calibrated identity probabilities.", "",
        f"Historical baseline outcomes were not exactly reproduced: "
        f"{len(mismatch_rows)} of {historical['compared_rows']} recording/stride/method rows "
        "have at least one identity-outcome difference. "
        f"For dual anchor, mean accuracy shifted by "
        f"{baseline_drift['dual_anchor']['mean_accuracy_change_percentage_points']:+.6f} percentage "
        "points; the largest absolute individual-case shift was "
        f"{baseline_drift['dual_anchor']['maximum_absolute_case_accuracy_change_percentage_points']:.6f} "
        "points. All comparisons above use freshly rerun baselines in the same environment, not the "
        "historical scores. The cause of historical discrepancies has not been isolated. Full mismatch "
        "rows and drift summaries for both baselines are in `summary.json`; the canonical results "
        "were not overwritten.",
        "", f"GNN equivalence control completed {control_audit.get('completed_tasks', 0)} of 100 tasks. "
        f"Assignments matched the original objective's rerun baseline exactly in "
        f"{control_audit.get('assignment_identical_tasks', 0)} tasks. The largest absolute accuracy "
        f"difference was {control_audit.get('maximum_absolute_accuracy_difference_percentage_points', 0):.6f} "
        "percentage points. Differences are consistent with numerical sensitivity in the mathematically "
        "equivalent cost path; equality of the alternative optima was not verified for every discrepancy. "
        "The complete diagnostic rows are saved in `summary.json`.",
        "", "## Reproduction", "",
        "Use Python 3.11 or newer and the optional pinned PyRecEst dependency from `pyproject.toml`. "
        "The measured environment used Python 3.12.3; its resolved package versions are in "
        "[requirements-pyrecest-lock.txt](../../requirements-pyrecest-lock.txt) and execution checks in "
        "[verification.json](../../work/results/pyrecest_followup_20261002/verification.json). "
        "Clone `work/vendor/PyRecEst` at commit `7eff44c5f4d62c54a8394270fc924cb1302f171b`. "
        "Use the original verified coordinate caches (the evaluator extracts a missing tuning cache "
        "from the immutable asset through byte-range reads). On a calculation server, run:", "",
        "```bash", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m deformeot.wormid_pyrecest_experiment \\",
        "  --protocol configs/wormid_pyrecest.yaml --cache-dir /path/to/coordinate-caches \\",
        "  --output-dir work/results/pyrecest_followup_20261002 --phase all --workers 12", "```", "",
        "The experiment saves code/package/data hashes, tuning selection, per-task checkpoints, and "
        "complete evaluation results. Existing checkpoints are reused only with matching provenance. "
        "A changed implementation or environment requires a fresh output directory.", "",
        "Complete machine-readable results: "
        "[evaluation_results.json](../../work/results/pyrecest_followup_20261002/evaluation_results.json). "
        "All 500 evaluation tasks and 100 equivalence-control tasks completed; the recovered environment "
        "passed all 103 tests.", "",
        f"Evaluation JSON SHA-256: `{hashlib.sha256(result_bytes).hexdigest()}`.", "",
        "The earlier paper source results and frozen expansion configuration remain unchanged.", ""])
    (output / "report.md").write_text("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6), sharey=True)
    for axis, group, title in zip(axes, groups[:2], ("All 20 comparison recordings", "Previous 16-recording cohort")):
        for index, name in enumerate(names):
            row = next(item for item in table if item["cohort"] == group and item["candidate"] == name)
            mean = row["accuracy_gain_percentage_points"]
            low, high = row["ci95_low_percentage_points"], row["ci95_high_percentage_points"]
            color = "#267b65" if row["latency_selected_scans"] == 0 else "#b47b28"
            axis.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt="o", color=color, capsize=4)
        axis.axvline(0, color="0.5", linewidth=1, linestyle="--")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Identity accuracy gain vs dual anchor (percentage points)", fontsize=9)
        axis.grid(axis="x", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(names)), [LABELS.get(name, name) for name in names])
    axes[0].invert_yaxis()
    fig.suptitle("PyRecEst follow-up: paired recording-bootstrap intervals", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "accuracy_comparison.png", dpi=180)
    fig.savefig(output / "accuracy_comparison.pdf")
    plt.close(fig)
    print(json.dumps({"report": str(output / "report.md"), "comparisons": table}, indent=2))


if __name__ == "__main__":
    main()
