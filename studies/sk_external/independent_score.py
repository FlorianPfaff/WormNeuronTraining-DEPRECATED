"""Independently score saved SK assignments and compare with the frozen October scorer."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ANIMALS = (
    "SK1_20230322_185704", "SK1_20230322_201650",
    "SK1_20230506_150145", "SK1_20230506_153351",
    "SK2_20220426_w00", "SK2_20220426_w01",
)
METHODS = ("dual_anchor", "cpd_reference")
KEYS = ("assignments", "truth_indices", "frame_indices", "frame_times")
FROZEN = (Path.home() / "wha-paper-phase2-20261004-iPztef/project/src/deformeot"
          / "paper_phase2_rescore.py")
FROZEN_SHA256 = "c218cb9527475906590c4839a9998ee0e8883eeb9b7f190dd6eea81ac08b95e4"
ROOT = Path(__file__).resolve().parent
DRAWS = 20_000
SEED = 20261004


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_scorer(path):
    if digest(path) != FROZEN_SHA256:
        raise ValueError(f"Frozen scorer digest differs: {path}")
    spec = importlib.util.spec_from_file_location("frozen_assignment_scorer", path)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module.score_assignment_arrays


def score(arrays, reference):
    assignments, truth, indices, times = arrays
    # The frozen pure scorer checks permutations, initialized order and monotonic times.
    check = reference(*arrays)
    identities = np.take_along_axis(truth, assignments, axis=1)
    wrong = identities != truth[0]
    n_frames, n_tracks = wrong.shape
    if n_frames < 2:
        raise ValueError("Persistence comparison requires at least two selected frames.")
    opportunities = (n_frames - 1) * n_tracks
    switches = int(np.count_nonzero(identities[1:] != identities[:-1]))
    wrong_count = int(np.count_nonzero(wrong))
    # In track-major order each start pairs with the next exclusive end, including at T.
    transitions = np.diff(wrong.T.astype(np.int8), prepend=0, append=0, axis=1)
    start_track, start_t = np.where(transitions == 1)
    end_track, end_t = np.where(transitions == -1)
    if not np.array_equal(start_track, end_track):
        raise ValueError("Episode ends do not match starts.")
    lengths = end_t - start_t
    if np.any(lengths <= 0) or int(lengths.sum()) != wrong_count:
        raise ValueError("Episode lengths do not conserve the number of incorrect assignments.")
    n_episodes = len(lengths)
    duration = wrong_count / n_episodes if n_episodes else None
    computed = {
        "frames": n_frames, "neurons": n_tracks, "correct": wrong.size - wrong_count,
        "total": wrong.size, "identity_accuracy": 1 - wrong_count / wrong.size,
        "correct_excluding_initialization": int(np.count_nonzero(~wrong[1:])),
        "total_excluding_initialization": opportunities,
        "accuracy_excluding_initialization": float((~wrong[1:]).mean()),
        "identity_switches": switches, "identity_switch_opportunities": opportunities,
        "identity_switch_rate": switches / opportunities,
        "wrong_identity_frames": wrong_count, "wrong_identity_episode_count": n_episodes,
        "mean_wrong_identity_duration_selected_scans": 0.0 if duration is None else duration,
        "maximum_wrong_identity_duration_selected_scans": int(lengths.max(initial=0)),
    }
    for k, v in computed.items():
        if not np.isclose(check[k], v, atol=1e-12, rtol=0):
            raise AssertionError(f"Vectorized and frozen loop scores disagree for {k}: {v}, {check[k]}")
    return {
        **computed, "switches_per_1000": 1000 * switches / opportunities,
        "mean_episode_length_if_any": duration, "first_frame_index": int(indices[0]),
        "last_frame_index": int(indices[-1]),
        "maximum_source_frame_gap": int(np.diff(indices).max()),
        "observed_span_seconds": float(times[-1] - times[0]),
    }


def read_case(path):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(archive[key] for key in KEYS)


def interval(point, draws):
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"estimate": float(point), "low95": float(low), "high95": float(high)}


def contrast(names, cases, cluster_dates=False):
    def endpoints(name, method):
        r = cases[name][method]
        return [100 * r["identity_accuracy"], 100 * r["accuracy_excluding_initialization"],
                r["switches_per_1000"], r["mean_wrong_identity_duration_selected_scans"]]

    # X[animal, method, endpoint]. Both methods always use the same sampled animals.
    x = np.array([[endpoints(name, method) for method in METHODS] for name in names])
    groups = [[i] for i in range(len(names))]
    if cluster_dates:
        dates = [name.split("_")[1] for name in names]
        groups = [[i for i, d in enumerate(dates) if d == label] for label in sorted(set(dates))]
    summed = np.array([x[group].sum(axis=0) for group in groups])
    sizes = np.array([len(group) for group in groups])
    choices = np.random.default_rng(SEED).integers(len(groups), size=(DRAWS, len(groups)))
    samples = summed[choices].sum(axis=1) / sizes[choices].sum(axis=1)[:, None, None]
    means = x.mean(axis=0)
    diffs = means[0] - means[1]
    sampled = samples[:, 0] - samples[:, 1]
    accuracy = interval(diffs[0], sampled[:, 0])
    unseeded = interval(diffs[1], sampled[:, 1])
    switch = interval(-diffs[2], -sampled[:, 2])
    relative = float(1 - means[0, 2] / means[1, 2]) if means[1, 2] else None

    def screen(acc):
        checks = {"relative_reduction_at_least_25_percent": relative is not None and relative >= .25,
                  "switch_benefit_lower_bound_positive": switch["low95"] > 0,
                  "accuracy_difference_lower_bound_above_minus_0_25_pp": acc["low95"] > -.25}
        return {"passed": all(checks.values()), "checks": checks}

    with_episodes = [i for i, name in enumerate(names)
                     if all(cases[name][m]["wrong_identity_episode_count"] > 0 for m in METHODS)]
    return {
        "animals": list(names), "resampling_unit": "whole acquisition date" if cluster_dates else "animal",
        "independent_resampling_units": len(groups), "bootstrap_samples": DRAWS, "bootstrap_seed": SEED,
        "dual_minus_cpd_accuracy_pp": accuracy, "dual_minus_cpd_unseeded_accuracy_pp": unseeded,
        "cpd_minus_dual_switches_per_1000": switch, "relative_switch_reduction_fraction": relative,
        "dual_minus_cpd_episode_length_scans_historical": interval(diffs[3], sampled[:, 3]),
        "episode_length_difference_both_methods_have_errors": {
            "animals": [names[i] for i in with_episodes],
            "estimate": float((x[with_episodes, 0, 3] - x[with_episodes, 1, 3]).mean())
            if with_episodes else None,
        },
        "higher_accuracy_animals": int(np.count_nonzero(x[:, 0, 0] > x[:, 1, 0])),
        "lower_switch_rate_animals": int(np.count_nonzero(x[:, 0, 2] < x[:, 1, 2])),
        "longer_mean_wrong_episode_animals_paper_convention": int(np.count_nonzero(x[:, 0, 3] > x[:, 1, 3])),
        "historical_screen": screen(accuracy), "historical_screen_excluding_seed": screen(unseeded),
    }


def summarize_method(rows):
    durations = [r["mean_episode_length_if_any"] for r in rows if r["mean_episode_length_if_any"] is not None]
    return {
        "animals": len(rows),
        "mean_accuracy_pct": 100 * float(np.mean([r["identity_accuracy"] for r in rows])),
        "mean_unseeded_accuracy_pct": 100 * float(np.mean([r["accuracy_excluding_initialization"] for r in rows])),
        "total_switches": sum(r["identity_switches"] for r in rows),
        "total_transition_opportunities": sum(r["identity_switch_opportunities"] for r in rows),
        "mean_switches_per_1000": float(np.mean([r["switches_per_1000"] for r in rows])),
        "mean_episode_length_scans_historical": float(np.mean([r["mean_wrong_identity_duration_selected_scans"] for r in rows])),
        "animals_with_episodes": len(durations),
        "mean_episode_length_scans_among_animals_with_episodes": float(np.mean(durations)) if durations else None,
        "longest_episode_scans": max(r["maximum_wrong_identity_duration_selected_scans"] for r in rows),
    }


def atomic_text(path, content):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(content)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scores")
    parser.add_argument("--frozen-scorer", type=Path, default=FROZEN)
    args = parser.parse_args()
    reference = frozen_scorer(args.frozen_scorer)
    rows = []
    cases = {}
    for name in ANIMALS:
        cases[name] = {}
        first = None
        for method in METHODS:
            path = args.results / name / (method + ".npz")
            arrays = read_case(path)
            if first is None:
                first = arrays[1:]
            elif any(not np.array_equal(a, b) for a, b in zip(first, arrays[1:])):
                raise ValueError(f"{name}: mismatched truth, source indices or times across methods")
            record = {"animal": name, "method": method, "npz_sha256": digest(path), **score(arrays, reference)}
            cases[name][method] = record
            rows.append(record)
    primary = contrast(ANIMALS, cases)
    dates = sorted({name.split("_")[1] for name in ANIMALS})
    payload = {
        "independent_verification": {"cases_matched_frozen_scorer": len(rows),
                                     "frozen_scorer_sha256": FROZEN_SHA256,
                                     "this_scorer_sha256": digest(Path(__file__))},
        "method_means": {method: summarize_method([cases[name][method] for name in ANIMALS]) for method in METHODS},
        "paired_equal_animal": primary,
        "whole_date_bootstrap_sensitivity": contrast(ANIMALS, cases, cluster_dates=True),
        "leave_date_out_sensitivity": {date: contrast([n for n in ANIMALS if n.split("_")[1] != date], cases)
                                      for date in dates},
        "by_source_descriptive": {s: {m: summarize_method([cases[n][m] for n in ANIMALS if n.startswith(s)])
                                      for m in METHODS} for s in ("SK1", "SK2")},
        "interpretation": [
            "Percentile intervals use paired recording resamples (or whole acquisition dates for the labeled sensitivity).",
            "Relative reduction uses the ratio of the two equal-animal mean rates. Total switch counts are descriptive.",
            "The historical screen is reported as a transferred reference criterion. The EY tuning-date exclusion is inapplicable.",
            "For maximal contiguous wrong runs the historical convention assigns zero to an animal without errors. Its conditional mean episode length is undefined.",
            "The first initialized frame counts in primary accuracy. Removing it from accuracy leaves the transition-based switch statistic unchanged.",
            "There are only six animals and three dates; the single SK2 date is confounded with source.",
        ],
        "per_animal_method": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(args.output_dir / "per_animal.csv", output.getvalue())
    atomic_text(args.output_dir / "summary.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Independently rescored all {len(rows)} files and matched frozen scores: {args.output_dir}")


if __name__ == "__main__":
    main()

