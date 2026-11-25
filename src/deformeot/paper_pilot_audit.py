"""Numerical reproducibility diagnosis without changing frozen tracking code.

All numerical work is intended to run on the configured calculation server.
The observer wraps the actual historical tracker; it does not reimplement it.
Historical aggregate results contain no assignment arrays, so a reconstruction
can establish aggregate agreement but cannot prove identical historical tracks.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import numpy as np

from . import tracker as affine_module
from . import wormid_benchmark as benchmark
from .wormid import IdentityBlindSequence, identity_blind_sequence
from .wormid_dandi import load_coordinate_cache, sha256_file
from .wormid_switch_strata import benchmark_frame_indices


METHODS = ("dual_anchor", "first_affine")
IDENTITY_FIELDS = ("correct", "total", "identity_switches", "permutation_seed")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def array_sha256(values: np.ndarray, *, integer: bool = False) -> str:
    """Use an explicit little-endian type, matching the earlier assignment hash."""
    return hashlib.sha256(np.asarray(values, dtype="<i8" if integer else "<f8").tobytes()).hexdigest()


def _key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["recording_id"]), int(row["stride"]), str(row["method"])


def select_diagnostic_cases(historical: Sequence[Mapping[str, Any]],
                            followup: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Two largest accuracy discrepancies and one fully matching case/method."""
    old = {_key(row): row for row in historical}
    new = {_key(row): row for row in followup if row.get("status", "complete") == "complete"}
    if len(old) != len(historical) or len(new) != sum(row.get("status", "complete") == "complete" for row in followup):
        raise ValueError("duplicate historical or follow-up method/recording/stride")
    cases = []
    for method in METHODS:
        candidates = []
        for key in sorted(old.keys() & new.keys()):
            if key[2] != method:
                continue
            baseline, observed = old[key], new[key]
            if int(baseline["total"]) != int(observed["total"]):
                raise ValueError(f"incomparable total for {key}")
            difference = (int(observed["correct"]) - int(baseline["correct"])) / int(baseline["total"])
            candidates.append({"recording_id": key[0], "stride": key[1], "method": method,
                               "accuracy_difference": difference,
                               "historical": dict(baseline), "followup": dict(observed)})
        differences = sorted((row for row in candidates if row["accuracy_difference"] != 0),
                             key=lambda row: (-abs(row["accuracy_difference"]), _key(row)))
        matching = [row for row in candidates if all(row["historical"][field] == row["followup"][field]
                                                    for field in IDENTITY_FIELDS)]
        if len(differences) < 2 or not matching:
            raise ValueError(f"cannot obtain the prespecified three diagnostic cases for {method}")
        cases.extend({**row, "selection_reason": "largest_absolute_accuracy_discrepancy"} for row in differences[:2])
        cases.append({**matching[0], "selection_reason": "lexicographically_first_identity_metric_exact"})
    return cases


def observed_track(sequence: IdentityBlindSequence, method: str,
                   config: benchmark.ReferenceTrackerConfig, *, capture_frame: int | None = None
                   ) -> tuple[benchmark.TrackResult, list[dict[str, Any]], dict[str, np.ndarray]]:
    """Observe original function calls, preserving every numerical operation.

    Only observer-owned arrays are copied. Capturing a frame stores full costs
    at that frame, including each internal affine-registration iteration.
    """
    if method not in METHODS:
        raise ValueError("the audit observes only the two historical baselines")
    original_affine = benchmark._affine
    original_assignment = benchmark.linear_sum_assignment
    original_inner_assignment = affine_module.linear_sum_assignment
    calls_per_frame = 3 if method == "dual_anchor" else 1
    trace: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    final_count = 0
    affine_count = 0
    outer_count = 0
    inner: list[dict[str, Any]] = []
    transforms: list[dict[str, Any]] = []
    cost_calls: list[dict[str, Any]] = []

    def inner_assignment(cost):
        rows, columns = original_inner_assignment(cost)
        entry = {"cost_sha256": array_sha256(cost),
                 "assignment_sha256": array_sha256(columns, integer=True),
                 "objective": float(np.asarray(cost)[rows, columns].sum())}
        inner.append(entry)
        if capture_frame == final_count + 1:
            prefix = f"affine{affine_count}_iteration{len(inner) - 1}"
            matrices[prefix + "_cost"] = np.asarray(cost).copy()
            matrices[prefix + "_assignment"] = np.asarray(columns).copy()
        return rows, columns

    def affine(source, target, settings):
        nonlocal inner, affine_count
        inner = []
        transform = original_affine(source, target, settings)
        transforms.append({"matrix": transform.matrix.tolist(), "offset": transform.offset.tolist(),
                           "source_sha256": array_sha256(source), "target_sha256": array_sha256(target),
                           "iterations": inner, "robust_rmse": transform.robust_rmse})
        if capture_frame == final_count + 1:
            matrices[f"affine{affine_count}_matrix"] = transform.matrix.copy()
            matrices[f"affine{affine_count}_offset"] = transform.offset.copy()
            matrices[f"affine{affine_count}_source"] = np.asarray(source).copy()
            matrices[f"affine{affine_count}_target"] = np.asarray(target).copy()
        affine_count += 1
        return transform

    def assignment(cost):
        nonlocal final_count, outer_count, affine_count, transforms, cost_calls
        rows, columns = original_assignment(cost)
        cost_calls.append({"cost_sha256": array_sha256(cost),
                           "assignment_sha256": array_sha256(columns, integer=True),
                           "objective": float(np.asarray(cost)[rows, columns].sum())})
        if capture_frame == final_count + 1:
            matrices[f"outer{outer_count}_cost"] = np.asarray(cost).copy()
            matrices[f"outer{outer_count}_assignment"] = np.asarray(columns).copy()
        outer_count += 1
        if outer_count == calls_per_frame:
            final_count += 1
            trace.append({"selected_frame_index": final_count,
                          "source_frame_index": int(sequence.source_frame_indices[final_count]),
                          "affine": transforms, "costs": cost_calls,
                          "assignment_sha256": array_sha256(columns, integer=True)})
            outer_count = 0
            affine_count = 0
            transforms = []
            cost_calls = []
        return rows, columns

    with patch.object(benchmark, "_affine", affine), \
         patch.object(benchmark, "linear_sum_assignment", assignment), \
         patch.object(affine_module, "linear_sum_assignment", inner_assignment):
        result = benchmark.track_sequence(sequence, method, config=config)
    if len(trace) != len(sequence.frames) - 1:
        raise AssertionError("frozen tracker call structure no longer matches observer")
    return result, trace, matrices


def environment_provenance() -> dict[str, Any]:
    versions = {}
    for name in ("numpy", "scipy", "pandas", "h5py", "fsspec", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "packages": versions,
            "numpy_build_configuration": buffer.getvalue(),
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def _worker(request_path: Path) -> dict[str, Any]:
    request = _json(request_path)
    case = request["case"]
    cache = Path(request["cache_dir"]) / f"{case['recording_id']}.npz"
    recording, metadata = load_coordinate_cache(cache)
    selected = benchmark_frame_indices(recording.frames, int(case["stride"]))
    seed = int(case["historical"]["permutation_seed"])
    sequence = identity_blind_sequence(recording, frame_indices=selected, seed=seed)
    capture = request.get("capture_frame")
    if capture is not None:
        stop = int(capture) + 1
        sequence = IdentityBlindSequence(sequence.frames[:stop], sequence.truth_indices[:stop],
                                         sequence.initial_track_ids, sequence.source_frame_indices[:stop])
    config = benchmark.ReferenceTrackerConfig(**request["geometry"])
    started = time.perf_counter()
    result, trace, matrices = observed_track(sequence, str(case["method"]), config, capture_frame=capture)
    metrics = asdict(benchmark.score_tracks(sequence, result))
    for field, value in tuple(metrics.items()):
        if isinstance(value, float) and not np.isfinite(value):
            metrics[field] = None
    destination = Path(request["output_path"])
    arrays_path = destination.with_suffix(".npz")
    np.savez_compressed(arrays_path, assignments=np.asarray(result.assignments, dtype=np.int64), **matrices)
    provenance = {"cache_sha256": sha256_file(cache),
                  "semantic_coordinate_sha256": metadata["semantic_coordinate_sha256"],
                  "positions_sha256": array_sha256(recording.positions_zyx_um),
                  "source_frame_indices": list(sequence.source_frame_indices),
                  "permutation_seed": seed,
                  "permutations_sha256": array_sha256(np.asarray(sequence.truth_indices), integer=True),
                  "anonymous_frames_sha256": array_sha256(np.asarray([frame.positions for frame in sequence.frames]))}
    payload = {"status": "complete", "signature": request["signature"], "metrics": metrics,
               "assignment_sha256": array_sha256(np.asarray(result.assignments), integer=True),
               "input": provenance, "environment": environment_provenance(), "trace": trace,
               "capture_frame": capture, "wall_seconds": time.perf_counter() - started,
               "arrays_path": str(arrays_path), "arrays_sha256": sha256_file(arrays_path)}
    _write(destination, payload)
    return payload


def _subprocess_environment(root: Path) -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(root / "src"), "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def _run_worker(python: str, request: dict[str, Any], root: Path) -> dict[str, Any]:
    destination = Path(request["output_path"])
    if destination.is_file():
        cached = _json(destination)
        if cached.get("signature") != request["signature"]:
            raise RuntimeError(f"audit checkpoint provenance changed: {destination}")
        if cached.get("status") == "complete" and destination.with_suffix(".npz").is_file():
            if cached.get("arrays_sha256") != sha256_file(destination.with_suffix(".npz")):
                raise RuntimeError(f"audit checkpoint arrays changed: {destination}")
            return cached
    request_path = destination.with_suffix(".request.json")
    _write(request_path, request)
    started = time.perf_counter()
    try:
        completed = subprocess.run([python, "-m", "deformeot.paper_pilot_audit", "--worker", str(request_path)],
                                   env=_subprocess_environment(root), cwd=root, capture_output=True, text=True,
                                   timeout=600, check=False)
        if completed.returncode:
            failed = {"status": "failed", "signature": request["signature"],
                      "error": completed.stderr[-10000:], "stdout": completed.stdout[-2000:],
                      "job_wall_seconds": time.perf_counter() - started}
            _write(destination, failed)
            return failed
        payload = _json(destination)
        payload["job_wall_seconds"] = time.perf_counter() - started
        _write(destination, payload)
        return payload
    except subprocess.TimeoutExpired:
        failed = {"status": "failed", "signature": request["signature"], "error": "600-second worker timeout",
                  "job_wall_seconds": time.perf_counter() - started}
        _write(destination, failed)
        return failed


def _first_trace_difference(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if len(left) != len(right):
        return {"error": "trace lengths differ", "left_frames": len(left), "right_frames": len(right)}
    for a, b in zip(left, right):
        if a != b:
            return {"selected_frame_index": a["selected_frame_index"], "source_frame_index": a["source_frame_index"],
                    "affine_differs": a["affine"] != b["affine"], "costs_differ": a["costs"] != b["costs"],
                    "assignment_differs": a["assignment_sha256"] != b["assignment_sha256"]}
    return None


def compare_reconstructed_runs(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare input, trace, and assignments independently of metric agreement."""
    if left.get("status") != "complete" or right.get("status") != "complete":
        return {"available": False}
    with np.load(left["arrays_path"], allow_pickle=False) as content:
        a = content["assignments"]
    with np.load(right["arrays_path"], allow_pickle=False) as content:
        b = content["assignments"]
    if a.shape != b.shape:
        raise ValueError("reconstruction assignment shapes differ")
    differing = np.flatnonzero(np.any(a != b, axis=1))
    return {"available": True, "inputs_identical": left["input"] == right["input"],
            "assignments_identical": not len(differing), "differing_assignment_entries": int(np.count_nonzero(a != b)),
            "first_assignment_divergence_selected_frame": int(differing[0]) if len(differing) else None,
            "first_assignment_divergence_source_frame": int(left["input"]["source_frame_indices"][differing[0]]) if len(differing) else None,
            "first_numerical_trace_difference": _first_trace_difference(left["trace"], right["trace"])}


def compare_captured_costs(left_path: Path, right_path: Path, method: str) -> dict[str, Any]:
    """Check both returned assignments under both frame-specific cost matrices.

    Equal objective values are evidence only for this captured scan, not a
    blanket assertion that all historical discrepancies arise from ties.
    """
    slot = 2 if method == "dual_anchor" else 0
    with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
        a, b = left[f"outer{slot}_cost"], right[f"outer{slot}_cost"]
        ia, ib = left["assignments"][-1], right["assignments"][-1]
        rows = np.arange(len(ia))
        coordinate_diagnostics: dict[str, Any] = {}
        if "affine0_target" in left and "affine0_target" in right:
            left_target, right_target = left["affine0_target"], right["affine0_target"]
            changed = ia != ib
            same_coordinates = np.all(left_target[ia] == left_target[ib], axis=1)
            coordinate_diagnostics = {
                "target_coordinates_identical_across_environments": bool(np.array_equal(left_target, right_target)),
                "changed_assignment_tracks": int(np.count_nonzero(changed)),
                "changed_tracks_assigned_identical_coordinates": int(np.count_nonzero(changed & same_coordinates)),
                "maximum_assignment_coordinate_distance_um": float(np.max(np.linalg.norm(left_target[ia] - left_target[ib], axis=1))),
                "target_detections": len(left_target),
                "unique_target_coordinates": len(np.unique(left_target, axis=0)),
            }
        return {"maximum_absolute_cost_difference": float(np.max(np.abs(a - b))),
                "cost_matrices_identical": bool(np.array_equal(a, b)),
                "left_cost_left_assignment": float(a[rows, ia].sum()),
                "left_cost_right_assignment": float(a[rows, ib].sum()),
                "right_cost_left_assignment": float(b[rows, ia].sum()),
                "right_cost_right_assignment": float(b[rows, ib].sum()),
                "left_objective_penalty_of_right": float(a[rows, ib].sum() - a[rows, ia].sum()),
                "right_objective_penalty_of_left": float(b[rows, ia].sum() - b[rows, ib].sum()),
                "same_assignment": bool(np.array_equal(ia, ib)),
                "coordinate_diagnostics": coordinate_diagnostics,
                "tie_interpretation": "Only the recorded objectives and coordinates establish ties for this captured scan; no inference about all historical discrepancies."}


def scoring_fixture_audit() -> dict[str, Any]:
    """Make switch counting and sustained identity errors explicit."""
    from .tracker import DetectionFrame
    points = np.asarray([[0., 0., 0.], [1., 2., 3.], [4., 1., 2.]])
    truth = np.arange(3)
    sequence = IdentityBlindSequence(tuple(DetectionFrame(points, time=index) for index in range(4)),
                                     tuple(truth.copy() for _ in range(4)), ("a", "b", "c"), tuple(range(4)))
    cases = {"perfect": ([truth, truth, truth, truth], 0, 12),
             "stable_wrong_after_swap": ([truth, np.asarray([1, 0, 2]), np.asarray([1, 0, 2]), np.asarray([1, 0, 2])], 2, 6),
             "swap_then_recover": ([truth, np.asarray([1, 0, 2]), truth, truth], 4, 10)}
    rows = []
    for name, (assignments, switches, correct) in cases.items():
        result = benchmark.TrackResult(name, tuple(assignments), 0., np.zeros(4))
        score = benchmark.score_tracks(sequence, result)
        rows.append({"name": name, "expected_switches": switches, "observed_switches": score.identity_switches,
                     "expected_correct": correct, "observed_correct": score.correct,
                     "passed": score.identity_switches == switches and score.correct == correct})
    return {"cases": rows, "all_passed": all(row["passed"] for row in rows),
            "semantics": "number of track identity changes, not swap events; a stable wrong identity accumulates accuracy error but no further switches"}


def run_audit(repository_root: str | Path, cache_dir: str | Path, output_dir: str | Path,
              historical_python: str | Path | None = None) -> dict[str, Any]:
    """Run six prespecified cases three times/environment with saved diagnostics."""
    root, cache_dir, output_dir = Path(repository_root).resolve(), Path(cache_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    historic_path = root / "outputs/paper/source-results/prospective.json"
    followup_path = root / "work/results/pyrecest_followup_20261002/evaluation_results.json"
    historical, followup = _json(historic_path), _json(followup_path)
    cases = select_diagnostic_cases(historical["results"],
                                    [row for row in followup["results"] if row["method"] in METHODS])
    context = followup["context"]
    geometry = context["geometry"]
    source_files = ("src/deformeot/paper_pilot_audit.py", "src/deformeot/tracker.py", "src/deformeot/wormid.py",
                    "src/deformeot/wormid_benchmark.py", "src/deformeot/wormid_dandi.py", "src/deformeot/wormid_switch_strata.py")
    hashes = {name: sha256_file(root / name) for name in source_files}
    historical_source_names = {"identity_blind_sequence_source": "src/deformeot/wormid.py",
                               "track_sequence_source": "src/deformeot/wormid_benchmark.py",
                               "range_extractor_source": "src/deformeot/wormid_dandi.py"}
    historical_source_checks = {key: hashes[path] == historical["code_sha256"].get(key)
                                for key, path in historical_source_names.items()}
    followup_source_checks = {path: checksum == context["code_sha256"].get(path)
                              for path, checksum in hashes.items() if path != "src/deformeot/paper_pilot_audit.py"}
    interpreters = {"current": sys.executable}
    if historical_python is not None:
        # Do not resolve a virtualenv's python symlink: its invocation path
        # determines pyvenv.cfg discovery and therefore the loaded environment.
        interpreters["historical_reconstruction"] = str(Path(historical_python).absolute())
    environments = {}
    for label, python in interpreters.items():
        probe = subprocess.run([python, "-m", "deformeot.paper_pilot_audit", "--environment"],
                               cwd=root, env=_subprocess_environment(root), capture_output=True, text=True, timeout=60, check=False)
        environments[label] = json.loads(probe.stdout) if probe.returncode == 0 else {"unavailable": probe.stderr[-8000:]}
    request_base = {"cache_dir": str(cache_dir), "geometry": geometry}
    started = time.perf_counter()
    rows = []
    for case_index, case in enumerate(cases):
        case_path = output_dir / f"case{case_index:02d}_{case['recording_id']}_s{case['stride']}_{case['method']}"
        case_path.mkdir(exist_ok=True)
        runs: dict[str, list[dict[str, Any]]] = {}
        input_recording, input_metadata = load_coordinate_cache(cache_dir / f"{case['recording_id']}.npz")
        frame_indices = benchmark_frame_indices(input_recording.frames, int(case["stride"]))
        historical_asset = next(asset for asset in historical["dandiset"]["assets"] if asset["recording_id"] == case["recording_id"])
        expected_semantic_hash = historical_asset["coordinate_cache"]["semantic_coordinate_sha256"]
        expected_seed = int(context["base_seed"]) + 1000 * context["inventory"].index(case["recording_id"]) + int(case["stride"])
        provenance_checks = {
            "semantic_coordinates_match_historical": input_metadata["semantic_coordinate_sha256"] == expected_semantic_hash,
            "semantic_coordinates_match_followup": input_metadata["semantic_coordinate_sha256"] == case["followup"]["semantic_coordinate_sha256"],
            "historical_seed_matches_frozen_rule": case["historical"]["permutation_seed"] == expected_seed,
            "followup_seed_matches_frozen_rule": case["followup"]["permutation_seed"] == expected_seed,
            "historical_frames_match": case["historical"]["frames"] == len(frame_indices),
            "followup_frames_match": case["followup"]["frames"] == len(frame_indices),
            "historical_first_last_frame_match": (case["historical"]["first_source_frame"], case["historical"]["last_source_frame"]) == (frame_indices[0], frame_indices[-1]),
            "followup_first_last_frame_match": (case["followup"]["first_source_frame"], case["followup"]["last_source_frame"]) == (frame_indices[0], frame_indices[-1]),
        }
        row = {"case": case, "runs": {}, "comparisons": {}, "input_provenance_checks": provenance_checks,
               "all_input_provenance_checks_pass": all(provenance_checks.values())}
        for label, python in interpreters.items():
            if "unavailable" in environments[label]:
                row["runs"][label] = {"status": "unavailable", "error": environments[label]["unavailable"]}
                continue
            signature = _digest({"case": case, "geometry": geometry, "sources": hashes,
                                 "cache_sha256": sha256_file(cache_dir / f"{case['recording_id']}.npz"),
                                 "environment": environments[label]})
            runs[label] = []
            for repeat in range(3):
                destination = case_path / f"{label}_repeat{repeat}.json"
                run = _run_worker(python, {**request_base, "case": case, "signature": signature,
                                          "output_path": str(destination)}, root)
                runs[label].append(run)
                print(f"audit case={case_index + 1}/6 {label} repeat={repeat + 1}/3 {run['status']}", flush=True)
            valid = [run for run in runs[label] if run.get("status") == "complete"]
            row["runs"][label] = {"completed": len(valid), "deterministic_assignments": len(valid) == 3 and len({run["assignment_sha256"] for run in valid}) == 1,
                                  "assignment_hashes": [run.get("assignment_sha256") for run in runs[label]],
                                  "metrics": [run.get("metrics") for run in runs[label]],
                                  "historical_metrics_match": [all(run.get("metrics", {}).get(field) == case["historical"][field]
                                                                    for field in ("correct", "total", "identity_switches")) for run in runs[label]],
                                  "followup_metrics_match": [all(run.get("metrics", {}).get(field) == case["followup"][field]
                                                                  for field in ("correct", "total", "identity_switches")) for run in runs[label]],
                                  "followup_assignment_hash_match": [run.get("assignment_sha256") == case["followup"].get("assignment_sha256") for run in runs[label]]}
        comparisons = []
        for label, available in runs.items():
            comparisons.extend((f"{label}_repeat0_vs_repeat{repeat}", label, 0, label, repeat) for repeat in (1, 2))
        if "historical_reconstruction" in runs:
            comparisons.append(("current_vs_historical_reconstruction", "current", 0, "historical_reconstruction", 0))
        for name, left_label, left_repeat, right_label, right_repeat in comparisons:
            left, right = runs[left_label][left_repeat], runs[right_label][right_repeat]
            compared = compare_reconstructed_runs(left, right)
            target = compared.get("first_assignment_divergence_selected_frame")
            if target is not None:
                captures = []
                for side, label, original in (("left", left_label, left), ("right", right_label, right)):
                    destination = case_path / f"diagnostic_{name}_{side}.json"
                    request = {**request_base, "case": case, "capture_frame": target,
                               "signature": _digest({"run": original["signature"], "capture_frame": target}),
                               "output_path": str(destination)}
                    captures.append(_run_worker(interpreters[label], request, root))
                if all(capture.get("status") == "complete" for capture in captures):
                    compared["captured_cost_comparison"] = compare_captured_costs(Path(captures[0]["arrays_path"]), Path(captures[1]["arrays_path"]), case["method"])
                    compared["diagnostic_arrays"] = [capture["arrays_path"] for capture in captures]
            row["comparisons"][name] = compared
        rows.append(row)
        _write(output_dir / "partial_audit.json", {"status": "running", "completed_cases": len(rows), "cases": rows})
    payload = {"status": "complete", "selection_rule": "two largest absolute accuracy discrepancies plus lexicographically first identity-metric-exact case per baseline",
               "source_sha256": hashes, "historical_results_sha256": sha256_file(historic_path),
               "followup_results_sha256": sha256_file(followup_path), "environments": environments,
               "historical_source_checks": historical_source_checks,
               "followup_source_checks": followup_source_checks,
               "all_recorded_source_hashes_match": all(historical_source_checks.values()) and all(followup_source_checks.values()),
               "historical_assignment_arrays_available": False,
               "historical_reconstruction_limitations": "Historical environment pins do not recover original host/BLAS dispatch. Matching historical aggregate metrics is not proof of identical historical assignment arrays.",
               "cases": rows, "scoring_fixtures": scoring_fixture_audit(),
               "wall_seconds": time.perf_counter() - started,
               "all_available_environments_deterministic": all(run.get("deterministic_assignments", False)
                                                              for row in rows for run in row["runs"].values()),
               "all_requested_environments_available": all("unavailable" not in item for item in environments.values()),
               "all_input_provenance_checks_pass": all(row["all_input_provenance_checks_pass"] for row in rows),
               "no_frozen_sources_changed": hashes == {name: sha256_file(root / name) for name in source_files}}
    payload["file_manifest"] = [{"path": str(path.relative_to(output_dir)), "bytes": path.stat().st_size,
                                 "sha256": sha256_file(path)} for path in sorted(output_dir.rglob("*")) if path.is_file()]
    _write(output_dir / "numerical_audit.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--historical-python", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--environment", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.environment:
        print(json.dumps(environment_provenance()))
    elif args.worker:
        _worker(args.worker)
    else:
        if args.cache_dir is None or args.output_dir is None:
            parser.error("--cache-dir and --output-dir are required")
        result = run_audit(args.repository_root, args.cache_dir, args.output_dir, args.historical_python)
        print(json.dumps({key: result[key] for key in ("status", "wall_seconds", "all_available_environments_deterministic")}))


if __name__ == "__main__":
    main()
