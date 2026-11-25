"""Build frozen paper figures, tables, and a compact result manifest.

Every plotted value is read from a versioned JSON result artifact.  The script
performs a few high-value protocol checks before writing anything so a stale or
partially regenerated benchmark cannot silently enter the manuscript.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "work" / "results"

SOURCE_FILES = {
    "reference": RESULTS / "wormid_reference_summary.json",
    "reference_raw": RESULTS / "wormid_reference_benchmark.json",
    "prospective": RESULTS / "wormid_prospective_expansion.json",
    "severe_switches": RESULTS / "wormid_severe_switches.json",
    "extent_reproducible": RESULTS / "wormid_extent" / "benchmark_reproducible.json",
    "extent_components": (
        RESULTS / "wormid_extent" / "extent_component_ablation.json"
    ),
    "density_sensitivity": (
        RESULTS / "wormid_extent" / "density_definition_sensitivity.json"
    ),
    "calibration": (
        RESULTS / "wormid_extent" / "dual_anchor_extent_calibration_heldout.json"
    ),
    "downstream": RESULTS / "wormid_downstream.json",
    "targettrack": RESULTS / "targettrack_extent_benchmark.json",
    "fdnc": RESULTS / "fdnc_sequential_baselines.json",
    "ascent_official": RESULTS / "ascent_official_evaluator.json",
    "ascent_current": RESULTS / "ascent_end_to_end_metrics.json",
}

FIGURE_STEMS = {
    "figure-1-method",
    "figure-2-dual-anchor",
    "figure-3-extent",
    "figure-4-calibration",
    "figure-5-external",
    "figure-6-downstream",
    "figure-s1-density-sensitivity",
}

TABLE_FILES = {
    "results-summary.json",
    "table-density-sensitivity.csv",
    "table-prospective-by-stride.csv",
    "table-prospective-recordings.csv",
    "table-internal-target-ledger.csv",
    "table-reference-by-stride.csv",
}

PROSPECTIVE_RESULT_SHA256 = (
    "c210e6fe2350fa1e2988d4ad20579b3afc76ae33a7a49e82a10ff27235945b3a"
)


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gold": "#E69F00",
    "sky": "#56B4E9",
    "gray": "#666666",
    "light_gray": "#D5D5D5",
    "dark": "#202124",
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extent_views(unified: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project the unified reproducible artifact into compact plotting views."""

    baseline_rows = [
        row
        for row in unified["confirmatory_held_out_results"]
        if row["method"] == "dual_anchor"
    ]
    dense_total = sum(int(row["dense_total"]) for row in baseline_rows)
    tracked_total = sum(int(row["tracked_total"]) for row in baseline_rows)
    shared = {
        "held_out_recordings": list(unified["held_out_recording_ids"]),
        "evaluation_strata": {
            "dense_threshold_um": unified["evaluation_strata"]["dense_threshold_um"],
            "held_out_dense_prevalence_overall": dense_total / tracked_total,
        },
    }
    confirmatory = {
        **shared,
        "extent_config": dict(unified["calibration_grid"]["selected_config"]),
        "held_out_micro_by_stride": unified["confirmatory_summary"]["micro_by_stride"],
        "complete_recording_paired_bootstrap": unified["confirmatory_summary"][
            "complete_recording_paired_bootstrap"
        ],
    }
    no_intensity = {
        **shared,
        "extent_config": dict(unified["no_intensity_config"]),
        "held_out_micro_by_stride": unified["no_intensity_summary"]["micro_by_stride"],
        "complete_recording_paired_bootstrap": unified["no_intensity_summary"][
            "complete_recording_paired_bootstrap"
        ],
    }
    return confirmatory, no_intensity


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fixed_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fig.savefig(
        directory / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "DeformEOT paper asset builder"},
    )
    fig.savefig(
        directory / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "DeformEOT paper asset builder",
            "Producer": "Matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    plt.close(fig)


def _clean_generated_outputs(output: Path) -> None:
    """Remove only stale files in namespaces owned by this builder."""

    figures = output / "figures"
    if figures.exists():
        allowed = {
            f"{stem}{suffix}"
            for stem in FIGURE_STEMS
            for suffix in (".png", ".pdf")
        }
        for path in figures.iterdir():
            if path.is_file() and path.name.startswith("figure-") and path.name not in allowed:
                path.unlink()
    tables = output / "tables"
    if tables.exists():
        for path in tables.iterdir():
            if (
                path.is_file()
                and (path.name.startswith("table-") or path.name == "results-summary.json")
                and path.name not in TABLE_FILES
            ):
                path.unlink()
    frozen = output / "source-results"
    if frozen.exists():
        allowed = {f"{key}.json" for key in SOURCE_FILES}
        for path in frozen.glob("*.json"):
            if path.name not in allowed:
                path.unlink()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.10,
        1.045,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _inside_panel_label(axis: plt.Axes, label: str, *, right: bool = False) -> None:
    axis.text(
        0.98 if right else 0.02,
        0.98,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="right" if right else "left",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.88},
        zorder=10,
    )


def _expected_inputs(payloads: dict[str, dict[str, Any]]) -> None:
    reference = payloads["reference"]
    if reference.get("held_out_recording_ids") is None or len(
        reference["held_out_recording_ids"]
    ) != 4:
        raise ValueError("reference result must contain four held-out recordings")
    if reference.get("strides") != [1, 5, 10, 20, 40]:
        raise ValueError("unexpected reference stride grid")

    prospective = payloads["prospective"]
    prospective_audit = prospective.get("outcome_audit", {})
    if _digest(SOURCE_FILES["prospective"]) != PROSPECTIVE_RESULT_SHA256:
        raise ValueError("canonical prospective result digest changed")
    if prospective_audit.get("all_16_untouched_recordings_evaluated") is not True:
        raise ValueError("prospective result does not contain all 16 untouched recordings")
    if prospective_audit.get("permutation_seeds_match_frozen_full_inventory_rule") is not True:
        raise ValueError("prospective result does not use the frozen seed rule")
    if prospective_audit.get("outcome_based_exclusions") != []:
        raise ValueError("prospective result contains an outcome-based exclusion")
    if prospective_audit.get("structural_or_network_failures") != []:
        raise ValueError("prospective result contains a structural or network failure")
    if prospective_audit.get("primary_replication_criterion_met") is not True:
        raise ValueError("prospective replication criterion is not met")
    if prospective.get("protocol_sha256") != (
        "b99c18526c104993aced172231607e0588c4e84e0b0cce6919d94af7407b56c3"
    ):
        raise ValueError("prospective protocol digest changed")
    if prospective.get("config_sha256") != (
        "fdca2e18c61bcb3544106c4cd14e0945be789e6818ecd5e8904d9990c3b42bf0"
    ):
        raise ValueError("prospective config digest changed")

    severe_switches = payloads["severe_switches"]
    if severe_switches.get("target_audit", {}).get("all_strides_pass_25_percent") is not True:
        raise ValueError("severe-stratum switch audit no longer passes the frozen gate")

    extent = payloads["extent_confirmatory"]
    if len(extent.get("held_out_recordings", [])) != 4:
        raise ValueError("extent result must contain four held-out recordings")
    if extent.get("extent_config", {}).get("extent_weight") != 8:
        raise ValueError("extent result does not use the frozen weight")
    if extent.get("extent_config", {}).get("extent_update") != 0.25:
        raise ValueError("extent result does not use the frozen update")

    components = payloads["extent_components"]
    if components.get("analysis_role", "").startswith("post-hoc mechanism ablation") is False:
        raise ValueError("cue-component analysis role unexpectedly changed")
    if components.get("full_15d_reference_reproduction", {}).get("exact") is not True:
        raise ValueError("full-cue component arm no longer reproduces the reference")

    density_sensitivity = payloads["density_sensitivity"]
    if density_sensitivity.get("q25_reproduction_check", {}).get("passed") is not True:
        raise ValueError("density sensitivity does not reproduce the frozen Q25 result")
    if density_sensitivity.get("analysis_role") != "post-hoc protocol-deviation sensitivity":
        raise ValueError("density sensitivity role unexpectedly changed")

    calibration = payloads["calibration"]
    if len(calibration.get("held_out_recording_ids", [])) != 4:
        raise ValueError("calibration result must contain four held-out recordings")
    if calibration["preregistered_target"]["passed"] is not False:
        raise ValueError("calibration gate unexpectedly changed")

    downstream = payloads["downstream"]
    if downstream["primary_target_assessment"]["status"] != "FAILED":
        raise ValueError("downstream primary-target status unexpectedly changed")


def _reference_rows(reference: dict[str, Any]) -> list[dict[str, Any]]:
    return list(reference["aggregate"])


def _paired_metric(
    downstream: dict[str, Any], metric: str, *, stride: int = 1
) -> dict[str, Any]:
    candidates = [
        row
        for row in downstream["confirmatory_heldout_bootstrap"]
        if row["metric"] == metric
        and int(row["stratum"]["stride"]) == stride
        and row["improved"] == "dual_anchor"
        and row["baseline"] == "first_affine"
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one downstream contrast for {metric}")
    return candidates[0]


def build_summary(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = payloads["reference"]
    prospective = payloads["prospective"]
    extent = payloads["extent_confirmatory"]
    strict = payloads["extent_no_intensity"]
    components = payloads["extent_components"]
    calibration = payloads["calibration"]
    downstream = payloads["downstream"]
    targettrack = payloads["targettrack"]["anonymous_temporal_extent_benchmark"]
    target_unseen = targettrack["metrics_by_split"]["unseen"]
    target_forced = targettrack["forced_assignment_sensitivity"]["metrics_by_split"][
        "unseen"
    ]
    fdnc = payloads["fdnc"]
    ascent = payloads["ascent_official"]

    primary_downstream = _paired_metric(downstream, "raw.pearson_median")
    derivative = _paired_metric(
        downstream, "first_difference_per_second.pearson_median"
    )
    event_f1 = _paired_metric(downstream, "positive_derivative_events.micro_f1")
    contamination = _paired_metric(
        downstream, "source_contamination.foreign_source_fraction"
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_from": {
            key: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _digest(path),
                "bytes": path.stat().st_size,
            }
            for key, path in SOURCE_FILES.items()
        },
        "wormid_dual_anchor": {
            "contrast": "dual_anchor minus first_affine",
            "four_heldout_animals_macro_across_five_strides": reference[
                "paired_macro_across_strides"
            ],
            "paired_by_stride": reference["paired_by_stride"],
        },
        "wormid_untouched_expansion": {
            "identity_semantics": prospective["identity_semantics"],
            "protocol_sha256": prospective["protocol_sha256"],
            "config_sha256": prospective["config_sha256"],
            "primary_untouched_accuracy_effect": prospective[
                "primary_untouched_accuracy_effect"
            ],
            "untouched_by_stride": prospective["untouched_by_stride"],
            "untouched_pooled_switch_rate_effect": prospective[
                "untouched_pooled_switch_rate_effect"
            ],
            "untouched_high_motion_accuracy_effect": prospective[
                "untouched_high_motion_accuracy_effect"
            ],
            "untouched_high_motion_switch_rate_effect": prospective[
                "untouched_high_motion_switch_rate_effect"
            ],
            "untouched_recording_effects": prospective[
                "untouched_recording_effects"
            ],
            "untouched_acquisition_date_hierarchical_bootstrap": prospective[
                "untouched_acquisition_date_hierarchical_bootstrap"
            ],
            "same_tuning_date_exclusion_sensitivity": prospective[
                "same_tuning_date_exclusion_sensitivity"
            ],
            "nontuning_20_stratified_summary": prospective[
                "nontuning_20_stratified_summary"
            ],
            "outcome_audit": prospective["outcome_audit"],
        },
        "wormid_extent_confirmatory": {
            "contrast": "dual_anchor_extent minus dual_anchor",
            "four_heldout_animals_macro_across_five_strides": extent[
                "complete_recording_paired_bootstrap"
            ],
            "dense_threshold_um": extent["evaluation_strata"]["dense_threshold_um"],
            "dense_prevalence": extent["evaluation_strata"][
                "held_out_dense_prevalence_overall"
            ],
            "analysis_role": (
                "tuning-frozen full local-cue analysis; the executed density definition "
                "differs from the radius-based internal plan"
            ),
        },
        "wormid_extent_no_intensity": {
            "contrast": "dual_anchor_extent_no_intensity_gate minus dual_anchor",
            "four_heldout_animals_macro_across_five_strides": strict[
                "complete_recording_paired_bootstrap"
            ],
            "analysis_role": (
                "post-hoc no-reliability-gate robustness analysis; spatial features "
                "remain intensity weighted; no evaluation-recording retuning"
            ),
        },
        "wormid_local_cue_components": {
            "analysis_role": components["analysis_role"],
            "comparisons_to_point_only": components["comparisons_to_point_only"],
            "full_15d_reference_reproduction": components[
                "full_15d_reference_reproduction"
            ],
        },
        "wormid_calibration": {
            "heldout_pooled": {
                key: calibration["held_out_pooled"][key]
                for key in (
                    "events",
                    "empirical_accuracy",
                    "mean_probability",
                    "negative_log_likelihood",
                    "brier",
                    "selective_accuracy_at_80_percent_coverage",
                    "frozen_calibration_threshold_at_99_5_percent_accuracy",
                    "retrospective_maximum_coverage_at_99_5_percent_accuracy",
                )
            },
            "fixed_ece": calibration["held_out_pooled"]["fixed_ece"]["ece"],
            "adaptive_ece": calibration["held_out_pooled"]["adaptive_ece"]["ece"],
            "complete_recording_bootstrap": calibration["held_out_pooled"][
                "complete_recording_bootstrap"
            ],
            "preregistered_target": calibration["preregistered_target"],
            "analysis_boundary": (
                "recordings held out from confidence-model and threshold fitting, not "
                "prospectively unseen to the earlier tracking benchmark"
            ),
        },
        "wormid_downstream": {
            "primary_raw_median_pearson": primary_downstream,
            "posthoc_first_difference_median_pearson": derivative,
            "exploratory_derivative_event_f1": event_f1,
            "wrong_neuron_source_fraction": contamination,
            "primary_target": downstream["primary_target_assessment"],
        },
        "targettrack_single_recording": {
            "selective_unseen": {
                "centroid_canonical_idf1": target_unseen["centroid_only"][
                    "canonical_idf1"
                ],
                "extent_canonical_idf1": target_unseen[
                    "centroid_plus_dynamic_extent"
                ]["canonical_idf1"],
                "centroid_switches": target_unseen["centroid_only"][
                    "identity_switches_between_spatially_matched_annotated_events"
                ],
                "extent_switches": target_unseen["centroid_plus_dynamic_extent"][
                    "identity_switches_between_spatially_matched_annotated_events"
                ],
            },
            "forced_assignment_unseen": {
                "centroid_canonical_idf1": target_forced["centroid_only"][
                    "canonical_idf1"
                ],
                "extent_canonical_idf1": target_forced[
                    "centroid_plus_dynamic_extent"
                ]["canonical_idf1"],
                "centroid_global_idf1": target_forced["centroid_only"][
                    "global_mapping_idf1"
                ],
                "extent_global_idf1": target_forced[
                    "centroid_plus_dynamic_extent"
                ]["global_mapping_idf1"],
                "centroid_switches": target_forced["centroid_only"][
                    "identity_switches_between_spatially_matched_annotated_events"
                ],
                "extent_switches": target_forced["centroid_plus_dynamic_extent"][
                    "identity_switches_between_spatially_matched_annotated_events"
                ],
            },
            "boundary": "one recording; offline bidirectional; supervised anchor",
        },
        "fdnc": {
            "dataset": fdnc["dataset"],
            "methods": {
                method: {
                    "link_f1": values["link_f1"],
                    "partial_idf1": values["partial_idf1"],
                    "link_precision": values["link_precision"],
                    "link_recall": values["link_recall"],
                }
                for method, values in fdnc["methods"].items()
            },
        },
        "ascent_reproduction": ascent,
        "wormid_severe_switches": payloads["severe_switches"],
        "density_definition_sensitivity": {
            "analysis_role": payloads["density_sensitivity"]["analysis_role"],
            "protocol_deviation_statement": payloads["density_sensitivity"][
                "protocol_deviation_statement"
            ],
            "calibration_proxy_derivation": payloads["density_sensitivity"][
                "calibration_proxy_derivation"
            ],
            "results": [
                {
                    "definition_id": row["definition_id"],
                    "label": row["label"],
                    "threshold_um": row["threshold_um"],
                    "dense_prevalence": row["dense_prevalence"],
                    "recording_macro_gain": row[
                        "complete_recording_paired_bootstrap"
                    ],
                }
                for row in payloads["density_sensitivity"]["sensitivity_results"]
            ],
        },
        "claim_status": {
            "1_dual_anchor_identity_or_switch": "PASSED",
            "2_extent_dense_identity": "PROTOCOL_DEVIATION",
            "3_calibration_80_percent": "FAILED",
            "4_gap_recovery": "INCONCLUSIVE",
            "5_raw_trace_or_decoder": "FAILED",
        },
    }
    return summary


def figure_method_schematic(directory: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.05), gridspec_kw={"wspace": 0.34})
    rng = np.random.default_rng(8)
    reference = np.column_stack(
        [np.linspace(0.1, 0.9, 12), 0.48 + 0.22 * np.sin(np.linspace(0, 2.6, 12))]
    )
    current = reference.copy()
    current[:, 0] += 0.06 * np.sin(np.linspace(0, 3.5, 12)) + 0.05
    current[:, 1] += 0.10 * np.cos(np.linspace(0, 2.2, 12))
    current += rng.normal(scale=0.008, size=current.shape)

    ax = axes[0]
    for point in reference:
        ax.add_patch(
            Ellipse(
                point,
                0.075,
                0.038,
                angle=25,
                facecolor=COLORS["blue"],
                edgecolor="none",
                alpha=0.30,
            )
        )
    for point in current:
        ax.add_patch(
            Ellipse(
                point,
                0.055,
                0.065,
                angle=-20,
                facecolor=COLORS["orange"],
                edgecolor="none",
                alpha=0.42,
            )
        )
    for left, right in zip(reference[::2], current[::2]):
        ax.add_patch(
            FancyArrowPatch(
                left,
                right,
                arrowstyle="-|>",
                mutation_scale=8,
                color=COLORS["gray"],
                linewidth=0.7,
                alpha=0.75,
            )
        )
    ax.scatter(reference[:, 0], reference[:, 1], s=11, color=COLORS["blue"], label="anchor")
    ax.scatter(current[:, 0], current[:, 1], s=11, color=COLORS["orange"], label="current")
    ax.set_title("Position and local signal both change", pad=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.15, 0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4)
    _inside_panel_label(ax, "A")

    ax = axes[1]
    boxes = [
        (0.05, 0.66, 0.30, 0.17, "previous\nframe", COLORS["sky"]),
        (0.05, 0.18, 0.30, 0.17, "immutable\nfirst frame", COLORS["blue"]),
        (0.61, 0.42, 0.32, 0.20, "joint Hungarian\nassignment", COLORS["green"]),
    ]
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1,
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8)
    text_box = {"facecolor": "white", "edgecolor": "none", "pad": 0.25, "alpha": 0.88}
    ax.text(
        0.48,
        0.63,
        "temporal cost",
        ha="center",
        fontsize=7,
        color=COLORS["gray"],
        bbox=text_box,
    )
    ax.text(
        0.48,
        0.39,
        "reference cost",
        ha="center",
        fontsize=7,
        color=COLORS["gray"],
        bbox=text_box,
    )
    ax.text(
        0.77,
        0.31,
        "+ local-patch cost",
        ha="center",
        fontsize=7,
        color=COLORS["orange"],
        bbox=text_box,
    )
    for start, end in [((0.35, 0.745), (0.61, 0.55)), ((0.35, 0.265), (0.61, 0.47))]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color=COLORS["dark"],
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            (0.77, 0.35),
            (0.77, 0.42),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color=COLORS["orange"],
        )
    )
    ax.set_title("Two anchors constrain drift", pad=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _inside_panel_label(ax, "B", right=True)

    ax = axes[2]
    t = np.arange(0, 18)
    truth = 0.55 + 0.10 * np.sin(t / 3)
    propagated = truth.copy()
    propagated[7:] += np.linspace(0.08, 0.26, len(t) - 7)
    anchored = truth + 0.018 * np.sin(1.7 * t)
    ax.plot(t, truth, color=COLORS["dark"], linewidth=1.8, label="true identity path")
    ax.plot(
        t,
        propagated,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.6,
        label="temporal only",
    )
    ax.plot(t, anchored, color=COLORS["blue"], linewidth=1.6, label="dual anchor")
    ax.axvspan(6.5, 8.5, color=COLORS["light_gray"], alpha=0.45, linewidth=0)
    ax.text(7.5, 0.36, "ambiguous\ntransition", ha="center", va="center", fontsize=7)
    ax.set_title("Immutable memory arrests drift", pad=9)
    ax.set_xlabel("time")
    ax.set_ylabel("schematic position")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper left", handlelength=2.2)
    _inside_panel_label(ax, "C")

    fig.suptitle("Dual-anchor initialized-cohort tracking (schematic)", y=1.08, fontsize=12)
    _save_figure(fig, directory, "figure-1-method")


def figure_reference(prospective: dict[str, Any], directory: Path) -> None:
    strides = [1, 5, 10, 20, 40]
    effects = sorted(
        prospective["untouched_recording_effects"],
        key=lambda row: row["recording_id"],
    )
    primary = prospective["primary_untouched_accuracy_effect"]
    by_stride = prospective["untouched_by_stride"]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), gridspec_kw={"wspace": 0.48})
    ax = axes[0]
    recording_values = 100 * np.asarray(
        [row["mean_accuracy_difference"] for row in effects]
    )
    y = np.arange(len(effects))
    point_colors = [COLORS["blue"] if value > 0 else COLORS["orange"] for value in recording_values]
    ax.scatter(recording_values, y, c=point_colors, s=23, zorder=3)
    overall_y = len(effects) + 0.7
    overall = 100 * float(primary["mean"])
    lower = overall - 100 * float(primary["ci95_low"])
    upper = 100 * float(primary["ci95_high"]) - overall
    ax.errorbar(
        [overall],
        [overall_y],
        xerr=np.asarray([[lower], [upper]]),
        fmt="D",
        color=COLORS["green"],
        ecolor=COLORS["green"],
        capsize=3,
        markersize=6,
        zorder=4,
    )
    labels = [
        row["recording_id"].removeprefix("sub-").split("_ses-")[0]
        for row in effects
    ]
    ax.set_yticks([*y, overall_y], [*labels, "16-recording mean"])
    ax.tick_params(axis="y", labelsize=6.5)
    ax.axvline(0, color=COLORS["dark"], linewidth=0.8)
    ax.axhline(len(effects) - 0.15, color=COLORS["light_gray"], linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("dual-anchor gain (percentage points)")
    ax.set_title("15 of 16 recording effects positive", pad=9)
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    _panel_label(ax, "A")

    ax = axes[1]
    stride_rows = [by_stride[str(stride)]["accuracy_difference"] for stride in strides]
    values = 100 * np.asarray([row["mean"] for row in stride_rows])
    lower = values - 100 * np.asarray([row["ci95_low"] for row in stride_rows])
    upper = 100 * np.asarray([row["ci95_high"] for row in stride_rows]) - values
    ax.errorbar(
        strides,
        values,
        yerr=np.vstack([lower, upper]),
        fmt="D-",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        capsize=3,
        markersize=5,
    )
    ax.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(strides, labels=[str(value) for value in strides])
    ax.minorticks_off()
    ax.set_ylim(0, max(3.5, float(np.max(values + upper)) + 0.3))
    ax.set_xlabel("temporal stride (volumes)")
    ax.set_ylabel("identity-accuracy gain (points)")
    ax.set_title("Positive effect at every stride", pad=9)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    _panel_label(ax, "B")

    ax = axes[2]
    all_reductions = [
        100 * float(by_stride[str(stride)]["all_transition_switches"]["relative_switch_reduction"])
        for stride in strides
    ]
    high_motion_reductions = [
        100 * float(by_stride[str(stride)]["high_motion_switches"]["relative_switch_reduction"])
        for stride in strides
    ]
    positions = np.arange(len(strides))
    width = 0.36
    all_bars = ax.bar(
        positions - width / 2,
        all_reductions,
        color=COLORS["blue"],
        width=width,
        label="all transitions",
    )
    severe_bars = ax.bar(
        positions + width / 2,
        high_motion_reductions,
        color=COLORS["green"],
        width=width,
        label="top-quartile motion",
    )
    for bars, values_to_label in (
        (all_bars, all_reductions),
        (severe_bars, high_motion_reductions),
    ):
        for bar, value in zip(bars, values_to_label):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.1,
                f"{value:.0f}",
                ha="center",
                fontsize=6.5,
            )
    ax.axhline(
        25,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.1,
        label="internal-plan gate",
    )
    ax.set_xticks(positions, labels=[str(value) for value in strides])
    ax.set_ylim(0, 84)
    ax.set_xlabel("temporal stride")
    ax.set_ylabel("identity-switch reduction (%)")
    ax.set_title("Switching falls in both strata", pad=9)
    ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    _panel_label(ax, "C")

    fig.suptitle(
        "A frozen outcome-untouched cohort replicates dual anchoring",
        y=1.04,
        fontsize=12,
    )
    _save_figure(fig, directory, "figure-2-dual-anchor")


def figure_extent(
    extent: dict[str, Any], components: dict[str, Any], directory: Path
) -> None:
    strides = [1, 5, 10, 20, 40]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.30), gridspec_kw={"wspace": 0.45})

    ax = axes[0]
    for metric, label, linestyle in [
        ("tracked_identity_accuracy", "all neurons", "-"),
        ("dense_identity_accuracy", "dense stratum", "--"),
    ]:
        base = [extent["held_out_micro_by_stride"][str(s)]["dual_anchor"][metric] * 100 for s in strides]
        improved = [
            extent["held_out_micro_by_stride"][str(s)]["dual_anchor_extent"][metric] * 100
            for s in strides
        ]
        ax.plot(
            strides,
            base,
            color=COLORS["gray"],
            linestyle=linestyle,
            marker="o",
            linewidth=1.4,
            alpha=0.85,
            label=f"point only, {label}",
        )
        ax.plot(
            strides,
            improved,
            color=COLORS["orange"],
            linestyle=linestyle,
            marker="D",
            linewidth=1.7,
            label=f"full local cue, {label}",
        )
    ax.set_xscale("log")
    ax.set_xticks(strides, labels=[str(value) for value in strides])
    ax.minorticks_off()
    ax.set_ylim(80, 94)
    ax.set_xlabel("temporal stride")
    ax.set_ylabel("seeded identity accuracy (%)")
    ax.set_title("Frozen full cue across strides", pad=9)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right", fontsize=7)
    _inside_panel_label(ax, "A")

    ax = axes[1]
    pooled = extent["complete_recording_paired_bootstrap"]
    items = [
        ("All", pooled["tracked_identity_accuracy_difference"]),
        ("Dense", pooled["dense_identity_accuracy_difference"]),
    ]
    y = np.arange(len(items))
    values = np.asarray([item[1]["mean_difference"] for item in items]) * 100
    low = values - np.asarray([item[1]["ci95_low"] for item in items]) * 100
    high = np.asarray([item[1]["ci95_high"] for item in items]) * 100 - values
    for j, (label, item) in enumerate(items):
        per_recording = np.asarray(item["per_recording_differences"]) * 100
        ax.scatter(per_recording, np.full(len(per_recording), j), s=26, facecolor="white", edgecolor=COLORS["orange"], zorder=3)
    ax.errorbar(values, y, xerr=np.vstack([low, high]), fmt="D", color=COLORS["orange"], capsize=3, markersize=6, zorder=4)
    ax.axvline(0, color=COLORS["dark"], linewidth=0.8)
    ax.axvline(
        2,
        color=COLORS["blue"],
        linestyle="--",
        linewidth=1.0,
        label="2 pp plan target",
    )
    ax.set_yticks(y, [item[0] for item in items])
    ax.set_xlabel("full-cue gain (percentage points)")
    ax.set_xlim(-0.2, 5.7)
    ax.set_title("Four-animal paired effects", pad=9)
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    _panel_label(ax, "B")

    ax = axes[2]
    component_order = [
        ("dual_anchor_full_spatial_15d", "Full 15D", COLORS["orange"]),
        ("dual_anchor_centroid_offset_3d", "Centroid 3D", COLORS["blue"]),
        ("dual_anchor_extent_only_12d", "Covariance 12D", COLORS["green"]),
    ]
    x = np.arange(len(component_order))
    width = 0.34
    comparisons = components["comparisons_to_point_only"]
    overall = [
        100
        * comparisons[key]["complete_recording_paired_bootstrap"][
            "tracked_identity_accuracy_difference"
        ]["mean_difference"]
        for key, _, _ in component_order
    ]
    dense = [
        100
        * comparisons[key]["complete_recording_paired_bootstrap"][
            "dense_identity_accuracy_difference"
        ]["mean_difference"]
        for key, _, _ in component_order
    ]
    colors = [color for _, _, color in component_order]
    ax.bar(x - width / 2, overall, width, color=colors, alpha=0.65, label="overall")
    ax.bar(
        x + width / 2,
        dense,
        width,
        color=colors,
        hatch="///",
        label="Q25 dense",
    )
    ax.set_xticks(x, [label for _, label, _ in component_order], rotation=12)
    ax.set_ylabel("gain (percentage points)")
    ax.set_ylim(0, 2.55)
    ax.set_title("Post-hoc cue decomposition", pad=9)
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    _panel_label(ax, "C")

    fig.suptitle(
        "A combined local image-patch cue improves identity under a changed density rule",
        y=1.09,
        fontsize=12,
    )
    _save_figure(fig, directory, "figure-3-extent")


def figure_density_sensitivity(sensitivity: dict[str, Any], directory: Path) -> None:
    rows = sorted(sensitivity["sensitivity_results"], key=lambda row: row["threshold_um"])
    thresholds = np.asarray([row["threshold_um"] for row in rows], dtype=float)
    bootstraps = [row["complete_recording_paired_bootstrap"] for row in rows]
    values = 100 * np.asarray([row["mean_difference"] for row in bootstraps])
    lower = values - 100 * np.asarray([row["ci95_low"] for row in bootstraps])
    upper = 100 * np.asarray([row["ci95_high"] for row in bootstraps]) - values

    fig, ax = plt.subplots(figsize=(7.6, 3.35))
    ax.errorbar(
        thresholds,
        values,
        yerr=np.vstack([lower, upper]),
        fmt="o-",
        color=COLORS["gray"],
        ecolor=COLORS["gray"],
        linewidth=1.1,
        markersize=4,
        capsize=2,
        label="post-hoc threshold sensitivity",
    )
    q25 = float(sensitivity["calibration_proxy_derivation"]["calibration_nearest_neighbor_q25_um"])
    q25_row = next(row for row in rows if row["definition_id"] == "calibration_nn_q25")
    q25_gain = 100 * float(
        q25_row["complete_recording_paired_bootstrap"]["mean_difference"]
    )
    ax.scatter([q25], [q25_gain], color=COLORS["blue"], marker="D", s=42, zorder=5)
    ax.axvline(q25, color=COLORS["blue"], linestyle=":", linewidth=1.0)
    ax.annotate(
        "reported Q25\n2.26 pp",
        xy=(q25, q25_gain),
        xytext=(1.08, 3.35),
        arrowprops={"arrowstyle": "-", "color": COLORS["blue"], "linewidth": 0.8},
        color=COLORS["dark"],
        fontsize=8,
        ha="center",
    )
    proxies = sensitivity["calibration_proxy_derivation"]["median_sigma_proxies_um"]
    proxy_thresholds = [
        2 * float(proxies["equivalent_isotropic_sigma_um"]),
        2 * float(proxies["rms_sigma_um"]),
        2 * float(proxies["largest_sigma_um"]),
    ]
    proxy_labels = ["2 x equiv. sigma", "2 x RMS sigma", "2 x largest sigma"]
    for index, (threshold, label) in enumerate(zip(proxy_thresholds, proxy_labels)):
        ax.axvline(
            threshold,
            color=COLORS["green"],
            alpha=0.55,
            linestyle="--",
            linewidth=0.9,
        )
        ax.text(
            threshold + (0.025 if index < 2 else -0.025),
            0.44 + 0.22 * index,
            label,
            rotation=90,
            va="bottom",
            ha="left" if index < 2 else "right",
            fontsize=7,
            color=COLORS["dark"],
        )
    ax.axhline(
        2.0,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.2,
        label="2 pp internal-plan target",
    )
    ax.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xlim(0.85, 4.15)
    ax.set_ylim(0.35, 4.65)
    ax.set_xlabel("nearest-neighbor threshold (um)")
    ax.set_ylabel("full-cue gain (percentage points)")
    ax.set_title("Post-hoc density-definition sensitivity", pad=9)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    _save_figure(fig, directory, "figure-s1-density-sensitivity")


def figure_calibration(calibration: dict[str, Any], directory: Path) -> None:
    pooled = calibration["held_out_pooled"]
    bins = pooled["adaptive_ece"]["bins"]
    curve = pooled["risk_coverage"]["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    probability = np.asarray([row["mean_probability"] for row in bins])
    accuracy = np.asarray([row["empirical_accuracy"] for row in bins])
    size = np.asarray([row["count"] for row in bins])
    size = 18 + 82 * size / size.max()
    ax.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.scatter(probability, accuracy, s=size, color=COLORS["blue"], alpha=0.82, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Split-evaluation reliability (ECE {pooled['fixed_ece']['ece']:.3f})")
    ax.grid(color=COLORS["light_gray"], linewidth=0.5)
    _panel_label(ax, "A")

    ax = axes[1]
    coverage = np.asarray([row["realized_coverage"] for row in curve])
    selective = np.asarray([row["selective_accuracy"] for row in curve])
    ax.plot(coverage * 100, selective * 100, color=COLORS["blue"], linewidth=2)
    ax.axhline(99.5, color=COLORS["orange"], linestyle="--", linewidth=1.2, label="99.5% target")
    at80 = pooled["selective_accuracy_at_80_percent_coverage"]
    ax.scatter([80], [100 * at80["selective_accuracy"]], color=COLORS["orange"], marker="D", s=45, zorder=3)
    ax.annotate(
        "98.46% at 80%",
        xy=(80, 100 * at80["selective_accuracy"]),
        xytext=(51, 95.4),
        arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 0.8},
        fontsize=8,
    )
    frozen = pooled["frozen_calibration_threshold_at_99_5_percent_accuracy"]
    ax.scatter(
        [100 * frozen["coverage"]],
        [100 * frozen["accuracy"]],
        color=COLORS["green"],
        marker="s",
        s=38,
        zorder=3,
        label="tuning-frozen policy",
    )
    ax.set_xlim(0, 101)
    ax.set_ylim(90, 100.2)
    ax.set_xlabel("retained associations (%)")
    ax.set_ylabel("selective identity accuracy (%)")
    ax.set_title("Useful ranking, failed safety gate")
    ax.grid(color=COLORS["light_gray"], linewidth=0.5)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    _panel_label(ax, "B")
    fig.suptitle(
        "Pooled ECE is low, but the selective safety gate fails",
        y=1.02,
        fontsize=12,
    )
    _save_figure(fig, directory, "figure-4-calibration")


def figure_external(
    targettrack: dict[str, Any], fdnc: dict[str, Any], ascent: dict[str, Any], directory: Path
) -> None:
    temporal = targettrack["anonymous_temporal_extent_benchmark"]
    unseen = temporal["metrics_by_split"]["unseen"]
    forced = temporal["forced_assignment_sensitivity"]["metrics_by_split"]["unseen"]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    labels = ["Selective\ncanonical", "Forced\ncanonical", "Forced\nglobal"]
    centroid = [
        unseen["centroid_only"]["canonical_idf1"],
        forced["centroid_only"]["canonical_idf1"],
        forced["centroid_only"]["global_mapping_idf1"],
    ]
    extent = [
        unseen["centroid_plus_dynamic_extent"]["canonical_idf1"],
        forced["centroid_plus_dynamic_extent"]["canonical_idf1"],
        forced["centroid_plus_dynamic_extent"]["global_mapping_idf1"],
    ]
    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, np.asarray(centroid) * 100, width, color=COLORS["gray"], label="centroid")
    ax.bar(x + width / 2, np.asarray(extent) * 100, width, color=COLORS["orange"], label="+ dynamic extent")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 50)
    ax.set_ylabel("IDF1 (%)")
    ax.set_title("Targettrack: one external movie", pad=9)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    _inside_panel_label(ax, "A", right=True)

    ax = axes[1]
    methods = ["nearest", "hungarian", "affine_hungarian"]
    labels = ["Nearest", "Hungarian", "Affine +\nHungarian"]
    link = [fdnc["methods"][method]["link_f1"] * 100 for method in methods]
    identity = [fdnc["methods"][method]["partial_idf1"] * 100 for method in methods]
    x = np.arange(3)
    ax.bar(x - width / 2, link, width, color=COLORS["green"], label="adjacent link F1")
    ax.bar(x + width / 2, identity, width, color=COLORS["purple"], label="temporal IDF1")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 92)
    ax.set_ylabel("score (%)")
    ax.set_title("fDNC: link F1 does not ensure identity", pad=9)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    _inside_panel_label(ax, "B", right=True)

    ax = axes[2]
    names = ["Paper\nfull pipeline", "Current code\n+ detections", "Released\ntrack-only"]
    values = [
        ascent["paper_reported_full_pipeline_MOTA"],
        ascent["current_code"]["MOTA"],
        ascent["released_track_only"]["MOTA"],
    ]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["green"]]
    bars = ax.bar(np.arange(3), np.asarray(values) * 100, color=colors, width=0.68)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 100 + 0.05, f"{value * 100:.3f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(3), names)
    ax.set_ylim(98.5, 100.08)
    ax.set_ylabel("modified MOTA (%)")
    ax.set_title("ASCENT evaluator audit", pad=9)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    _inside_panel_label(ax, "C")

    fig.suptitle("Independent sources delimit what each metric can establish", y=1.09, fontsize=12)
    _save_figure(fig, directory, "figure-5-external")


def figure_downstream(downstream: dict[str, Any], directory: Path) -> None:
    specifications = [
        ("Raw Pearson\nmedian (primary)", "raw.pearson_median", COLORS["orange"]),
        ("Raw Pearson\nmean (post-hoc)", "raw.pearson_mean", COLORS["gray"]),
        (
            "First-difference\nPearson median (post-hoc)",
            "first_difference_per_second.pearson_median",
            COLORS["blue"],
        ),
        (
            "Derivative-event\nF1 (exploratory)",
            "positive_derivative_events.micro_f1",
            COLORS["green"],
        ),
        (
            "Wrong-neuron\nsource fraction",
            "source_contamination.foreign_source_fraction",
            COLORS["purple"],
        ),
    ]
    rows = [_paired_metric(downstream, metric) for _, metric, _ in specifications]
    y = np.arange(len(rows))[::-1]
    values = np.asarray([row["mean_paired_difference"] for row in rows])
    low = values - np.asarray([row["ci95_low"] for row in rows])
    high = np.asarray([row["ci95_high"] for row in rows]) - values

    fig, ax = plt.subplots(figsize=(7.0, 3.55))
    for index, ((label, _, color), value, left, right) in enumerate(
        zip(specifications, values, low, high)
    ):
        ax.errorbar(
            [value],
            [y[index]],
            xerr=np.asarray([[left], [right]]),
            fmt="D",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
    ax.axvline(0, color=COLORS["dark"], linewidth=0.8)
    ax.axvline(0.03, color=COLORS["orange"], linestyle="--", linewidth=1.0)
    ax.text(
        0.034,
        len(rows) - 0.72,
        "internally specified raw-median target",
        color=COLORS["orange"],
        fontsize=7,
        va="top",
    )
    ax.set_yticks(y, [item[0] for item in specifications])
    ax.set_xlim(-0.07, 0.45)
    ax.set_xlabel("dual anchor minus first-frame affine")
    ax.set_title("Trace dynamics reveal damage hidden by raw-correlation ceilings", pad=14)
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    ax.text(
        0.02,
        -0.23,
        "Four split-evaluation animals, stride 1; paired recording bootstrap. Only the first row was internally specified.",
        transform=ax.transAxes,
        fontsize=7.5,
    )
    _save_figure(fig, directory, "figure-6-downstream")


def write_tables(
    payloads: dict[str, dict[str, Any]], summary: dict[str, Any], directory: Path
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    reference = payloads["reference"]
    prospective = payloads["prospective"]
    with (directory / "table-reference-by-stride.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stride",
                "method",
                "recording_macro_identity_accuracy",
                "ci95_low",
                "ci95_high",
                "recording_macro_high_motion_accuracy",
                "identity_switches",
                "runtime_seconds",
            ],
        )
        writer.writeheader()
        for row in reference["aggregate"]:
            writer.writerow(
                {
                    "stride": row["stride"],
                    "method": row["method"],
                    "recording_macro_identity_accuracy": row["identity_accuracy"]["mean"],
                    "ci95_low": row["identity_accuracy"]["ci95_low"],
                    "ci95_high": row["identity_accuracy"]["ci95_high"],
                    "recording_macro_high_motion_accuracy": row["severe_identity_accuracy"][
                        "mean"
                    ],
                    "identity_switches": row["identity_switches"],
                    "runtime_seconds": row["runtime_seconds"],
                }
            )

    with (directory / "table-prospective-by-stride.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stride",
                "accuracy_gain",
                "ci95_low",
                "ci95_high",
                "all_transition_switch_reduction",
                "high_motion_accuracy_gain",
                "high_motion_ci95_low",
                "high_motion_ci95_high",
                "high_motion_switch_reduction",
            ]
        )
        for stride in (1, 5, 10, 20, 40):
            row = prospective["untouched_by_stride"][str(stride)]
            accuracy = row["accuracy_difference"]
            high_accuracy = row["high_motion_accuracy_difference"]
            writer.writerow(
                [
                    stride,
                    accuracy["mean"],
                    accuracy["ci95_low"],
                    accuracy["ci95_high"],
                    row["all_transition_switches"]["relative_switch_reduction"],
                    high_accuracy["mean"],
                    high_accuracy["ci95_low"],
                    high_accuracy["ci95_high"],
                    row["high_motion_switches"]["relative_switch_reduction"],
                ]
            )

    with (directory / "table-prospective-recordings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "recording_id",
                "acquisition_date",
                "mean_accuracy_difference",
                "switch_rate_reduction",
                "mean_high_motion_accuracy_difference",
                "high_motion_switch_rate_reduction",
            ],
        )
        writer.writeheader()
        for row in prospective["untouched_recording_effects"]:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    with (directory / "table-internal-target-ledger.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["internal_target", "status", "finding"])
        writer.writerow(
            [
                1,
                "MET AND REPLICATED WITHIN SOURCE",
                "Frozen outcome-untouched cohort: exact high-motion switches fell 45.4-65.9% across strides and accuracy gained 1.498 pp.",
            ]
        )
        writer.writerow(
            [
                2,
                "PROTOCOL DEVIATION",
                "Frozen Q25 stratum gained 2.260 pp, but it differed from the radius-based plan wording.",
            ]
        )
        writer.writerow(
            [
                3,
                "FAILED",
                "Split-evaluation fixed-bin ECE was 0.0167, but 80%-coverage accuracy was 98.46%, below 99.5%.",
            ]
        )
        writer.writerow(
            [
                4,
                "INCONCLUSIVE",
                "Gap-event denominators were sparse and method-dependent in the single Targettrack movie.",
            ]
        )
        writer.writerow(
            [
                5,
                "FAILED",
                "Primary raw median trace-correlation gain was 0.0071, below the 0.03 target.",
            ]
        )

    with (directory / "table-density-sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "definition_id",
                "label",
                "threshold_um",
                "dense_prevalence",
                "recording_macro_gain",
                "ci95_low",
                "ci95_high",
                "analysis_role",
            ]
        )
        for row in sorted(
            payloads["density_sensitivity"]["sensitivity_results"],
            key=lambda item: item["threshold_um"],
        ):
            bootstrap = row["complete_recording_paired_bootstrap"]
            writer.writerow(
                [
                    row["definition_id"],
                    row["label"],
                    row["threshold_um"],
                    row["dense_prevalence"],
                    bootstrap["mean_difference"],
                    bootstrap["ci95_low"],
                    bootstrap["ci95_high"],
                    row["analysis_role"],
                ]
            )

    (directory / "results-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def copy_frozen_results(directory: Path) -> None:
    destination = directory / "source-results"
    destination.mkdir(parents=True, exist_ok=True)
    for key, path in SOURCE_FILES.items():
        shutil.copy2(path, destination / f"{key}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "paper",
        help="paper artifact directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    figures = output / "figures"
    tables = output / "tables"
    _clean_generated_outputs(output)
    payloads = {key: _load(path) for key, path in SOURCE_FILES.items()}
    extent_view, no_intensity_view = _extent_views(payloads["extent_reproducible"])
    payloads["extent_confirmatory"] = extent_view
    payloads["extent_no_intensity"] = no_intensity_view
    _expected_inputs(payloads)
    _style()
    summary = build_summary(payloads)
    figure_method_schematic(figures)
    figure_reference(payloads["prospective"], figures)
    figure_extent(
        payloads["extent_confirmatory"], payloads["extent_components"], figures
    )
    figure_density_sensitivity(payloads["density_sensitivity"], figures)
    figure_calibration(payloads["calibration"], figures)
    figure_external(
        payloads["targettrack"],
        payloads["fdnc"],
        payloads["ascent_official"],
        figures,
    )
    figure_downstream(payloads["downstream"], figures)
    write_tables(payloads, summary, tables)
    copy_frozen_results(output)
    print(json.dumps({"output": str(output), "figures": 7, "sources": len(SOURCE_FILES)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
