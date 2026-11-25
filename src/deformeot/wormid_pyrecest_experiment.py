"""Resumable PyRecEst comparisons on the original WormID cohort.

Hyperparameters are selected using the single original tuning recording.  The
other twenty recordings are follow-up comparisons: their original outcomes
have already been inspected and they are not a new untouched replication.
Only anonymous coordinate clouds enter the adapters; hidden ROI identities
enter the common scoring function after association has finished.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from .wormid import identity_blind_sequence
from .wormid_benchmark import ReferenceTrackerConfig, TrackResult, score_tracks, track_sequence
from .wormid_dandi import DandiAsset, extract_coordinate_cache, load_coordinate_cache, sha256_file
from .wormid_prospective import verify_frozen_protocol
from .wormid_switch_strata import benchmark_frame_indices


BASELINES = ("dual_anchor", "first_affine")
PYRECEST_METHODS = ("pyrecest_gnn", "pyrecest_mht")
IDENTITY_SEMANTICS = "persistent initialized ROI/neuron-instance identity"


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_mapping(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    payload = json.loads(content) if path.suffix.lower() == ".json" else yaml.safe_load(content)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"unsafe or empty candidate name: {name!r}")
    return name


def validate_candidates(candidates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if not candidates:
        raise ValueError("at least one PyRecEst candidate is required")
    validated = []
    names = set(BASELINES)
    for candidate in candidates:
        name = _safe_name(str(candidate["name"]))
        method = str(candidate["method"])
        config = candidate.get("config", {})
        if method not in PYRECEST_METHODS or not isinstance(config, dict):
            raise ValueError(f"invalid PyRecEst candidate {name}")
        if name in names:
            raise ValueError(f"duplicate or reserved candidate name {name}")
        names.add(name)
        specification: dict[str, object] = {"name": name, "method": method, "config": dict(config)}
        if "family" in candidate:
            specification["family"] = _safe_name(str(candidate["family"]))
        if "control" in candidate:
            if not isinstance(candidate["control"], bool):
                raise ValueError("candidate control flag must be boolean")
            specification["control"] = candidate["control"]
        validated.append(specification)
    return validated


def _source_signature(root: Path) -> dict[str, str]:
    files_to_hash = (
        "src/deformeot/wormid.py", "src/deformeot/wormid_benchmark.py",
        "src/deformeot/wormid_dandi.py", "src/deformeot/tracker.py",
        "src/deformeot/wormid_switch_strata.py",
        "src/deformeot/wormid_pyrecest_experiment.py",
        "src/deformeot/wormid_pyrecest_gnn.py", "src/deformeot/wormid_pyrecest_mht.py",
    )
    return {relative: sha256_file(root / relative) for relative in files_to_hash if (root / relative).is_file()}


def _package_provenance(root: Path, protocol: Mapping[str, object]) -> dict[str, object]:
    versions = {}
    for package in ("numpy", "scipy", "h5py", "fsspec", "pyrecest", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    package_section = protocol.get("pyrecest", {})
    if not isinstance(package_section, dict):
        raise ValueError("pyrecest provenance section must be a mapping")
    vendor = root / str(package_section.get("vendor_path", "work/vendor/PyRecEst"))
    actual_commit = None
    if vendor.is_dir():
        result = subprocess.run(["git", "-C", str(vendor), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            actual_commit = result.stdout.strip()
    expected_commit = package_section.get("commit")
    if expected_commit is not None and actual_commit != str(expected_commit):
        raise RuntimeError(f"PyRecEst vendor commit mismatch: expected {expected_commit}, found {actual_commit}")
    vendor_sources = {}
    for relative in ("src/pyrecest/filters/global_nearest_neighbor.py", "src/pyrecest/filters/multi_hypothesis_tracker.py"):
        if (vendor / relative).is_file():
            vendor_sources[relative] = sha256_file(vendor / relative)
    return {"python": sys.version, "platform": platform.platform(), "hostname": socket.gethostname(),
            "packages": versions, "vendor_commit": actual_commit, "expected_vendor_commit": expected_commit,
            "vendor_source_sha256": vendor_sources}


def load_experiment_context(repository_root: str | Path, protocol_path: str | Path) -> dict[str, object]:
    """Validate the original freeze and the follow-up protocol before outcomes."""
    root = Path(repository_root).resolve()
    source = Path(protocol_path).resolve()
    freeze = verify_frozen_protocol(root)
    protocol = _read_mapping(source)
    if int(protocol.get("version", -1)) != 1:
        raise ValueError("unsupported PyRecEst protocol version")
    dataset = protocol.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("dataset section must be a mapping")
    frozen_path = root / str(dataset.get("frozen_config", "configs/wormid_prospective.yaml"))
    frozen = _read_mapping(frozen_path)
    original_dataset = frozen["dataset"]
    original_association = frozen["association"]
    tuning = str(original_dataset["tuning_recording"])
    legacy = tuple(str(value) for value in original_dataset["legacy_evaluation_recordings"])
    previous = tuple(str(value) for value in original_dataset["untouched_recordings"])
    inventory = tuple(sorted((tuning, *legacy, *previous)))
    if len(inventory) != 21 or len(set(inventory)) != 21 or len(legacy) != 4 or len(previous) != 16:
        raise ValueError("original frozen split must be exactly one tuning, four legacy, sixteen previous prospective")
    association = protocol.get("association", {})
    if not isinstance(association, dict):
        raise ValueError("association section must be a mapping")
    strides = tuple(int(value) for value in association.get("strides", original_association["strides"]))
    base_seed = int(association.get("base_seed", original_association["base_seed"]))
    if strides != tuple(original_association["strides"]) or base_seed != int(original_association["base_seed"]):
        raise ValueError("follow-up must use the original frozen strides and seed rule")
    geometry = asdict(ReferenceTrackerConfig(**original_association["geometry"]))
    supplied_geometry = asdict(ReferenceTrackerConfig(**association.get("geometry", geometry)))
    if supplied_geometry != geometry:
        raise ValueError("follow-up baseline geometry must match the original freeze")
    candidates = validate_candidates(protocol.get("candidates", []))
    tuning_section = protocol.get("tuning", {})
    if not isinstance(tuning_section, dict) or tuning_section.get("objective", "mean_identity_accuracy") != "mean_identity_accuracy":
        raise ValueError("only tuning-recording mean identity accuracy is a supported primary selection objective")
    evaluation = protocol.get("evaluation", {})
    if not isinstance(evaluation, dict) or tuple(evaluation.get("baselines", BASELINES)) != BASELINES:
        raise ValueError("evaluation must retain dual_anchor and first_affine baselines")
    if int(evaluation.get("bootstrap_samples", 20_000)) < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    code_sha = _source_signature(root)
    provenance = _package_provenance(root, protocol)
    signature = _canonical_digest({"protocol_sha256": sha256_file(source), "code_sha256": code_sha,
                                   "vendor_commit": provenance["vendor_commit"],
                                   "vendor_source_sha256": provenance["vendor_source_sha256"],
                                   "packages": provenance["packages"], "python": platform.python_version()})
    return {"root": str(root), "protocol_path": str(source), "protocol": protocol,
            "protocol_sha256": sha256_file(source), "original_frozen_config_sha256": sha256_file(frozen_path),
            "original_freeze_integrity": freeze, "code_sha256": code_sha, "provenance": provenance,
            "implementation_signature": signature, "inventory": list(inventory),
            "tuning_recording": tuning, "legacy_recordings": list(legacy), "previous_prospective_recordings": list(previous),
            "strides": list(strides), "base_seed": base_seed, "geometry": geometry, "candidates": candidates}


def _asset_lookup(context: Mapping[str, object]) -> dict[str, DandiAsset]:
    protocol = context["protocol"]
    assert isinstance(protocol, dict)
    dataset = protocol.get("dataset", {})
    assert isinstance(dataset, dict)
    source = Path(str(context["root"])) / str(dataset.get("frozen_inventory", "outputs/paper/source-results/prospective.json"))
    if not source.is_file():
        return {}
    payload = _read_mapping(source)
    asset_rows = payload.get("assets", payload.get("dandiset", {}).get("assets", []))
    allowed = {field.name for field in fields(DandiAsset)}
    return {str(row["recording_id"]): DandiAsset(**{name: row[name] for name in allowed}) for row in asset_rows}


def ensure_coordinate_cache(recording_id: str, cache_dir: Path, context: Mapping[str, object]) -> dict[str, object]:
    path = cache_dir / f"{recording_id}.npz"
    if not path.is_file():
        assets = _asset_lookup(context)
        if recording_id not in assets:
            raise FileNotFoundError(f"missing cache {path}; no immutable asset metadata is available for extraction")
        _, audit = extract_coordinate_cache(assets[recording_id], cache_dir)
        return audit
    recording, metadata = load_coordinate_cache(path)
    if recording.recording_id != recording_id:
        raise ValueError(f"cache recording identity differs from filename: {path}")
    assets = _asset_lookup(context)
    asset = assets.get(recording_id)
    if asset is not None and (metadata.get("source_asset_id") != asset.asset_id or metadata.get("source_sha256_declared_by_dandi") != asset.sha256):
        raise ValueError(f"cache provenance differs from immutable asset {recording_id}")
    return {**metadata, "cache_path": str(path.resolve()), "cache_sha256": sha256_file(path), "cache_size": path.stat().st_size, "cache_reused": True}


def _run_method(sequence: Any, specification: Mapping[str, object], geometry: ReferenceTrackerConfig) -> TrackResult:
    method = str(specification["method"])
    config = specification.get("config", {})
    assert isinstance(config, dict)
    if method in BASELINES:
        return track_sequence(sequence, method, config=geometry)
    if method == "pyrecest_gnn":
        from .wormid_pyrecest_gnn import PyRecEstGNNConfig, track_pyrecest_gnn
        return track_pyrecest_gnn(sequence, config=PyRecEstGNNConfig(**config), geometry=geometry)
    if method == "pyrecest_mht":
        from .wormid_pyrecest_mht import PyRecEstMHTConfig, track_pyrecest_mht
        return track_pyrecest_mht(sequence, config=PyRecEstMHTConfig(**config), geometry=geometry)
    raise ValueError(f"unknown method {method}")


def _validate_exhaustive_result(result: TrackResult, neurons: int, frames: int) -> None:
    if len(result.assignments) != frames:
        raise ValueError("tracker must return one assignment for every selected scan")
    expected = np.arange(neurons)
    for assignment in result.assignments:
        observed = np.asarray(assignment)
        if observed.shape != (neurons,) or not np.issubdtype(observed.dtype, np.integer) or not np.array_equal(np.sort(observed), expected):
            raise ValueError("tracker assignments must be exhaustive one-to-one permutations")


def checkpoint_path(output_dir: Path, phase: str, recording_id: str, stride: int, candidate: str) -> Path:
    return output_dir / "checkpoints" / phase / f"{_safe_name(recording_id)}__s{stride}__{_safe_name(candidate)}.json"


def read_checkpoint(path: Path, task_signature: str) -> dict[str, object] | None:
    if not path.is_file():
        return None
    row = _read_mapping(path)
    if row.get("task_signature") != task_signature:
        raise RuntimeError(f"checkpoint provenance changed; use a fresh output directory: {path}")
    return row if row.get("status") == "complete" else None


def _latency_metadata(times: np.ndarray, selected: Sequence[int], specification: Mapping[str, object]) -> dict[str, object]:
    config = specification.get("config", {})
    assert isinstance(config, dict)
    lag = int(config.get("fixed_lag", 0)) if specification["method"] == "pyrecest_mht" else 0
    if lag < 0:
        raise ValueError("fixed lag cannot be negative")
    selected_times = times[np.asarray(selected)]
    indices = np.arange(len(selected_times))
    future = np.minimum(indices + lag, len(selected_times) - 1)
    delays = selected_times[future] - selected_times
    delays[0] = 0.0  # first scan initializes known track identities immediately
    return {"latency_selected_scans": lag, "maximum_actual_lookahead_seconds": float(np.max(delays)),
            "mean_actual_lookahead_seconds": float(np.mean(delays)),
            "latency_semantics": "future selected scans used before commitment; shorter available lookahead at recording tail"}


def _benchmark_recording_task(task: Mapping[str, object]) -> list[dict[str, object]]:
    """One worker owns a recording and checkpoints every method/stride result."""
    context = task["context"]
    assert isinstance(context, dict)
    recording_id = str(task["recording_id"])
    phase = str(task["phase"])
    cache_dir = Path(str(task["cache_dir"]))
    output_dir = Path(str(task["output_dir"]))
    audit = ensure_coordinate_cache(recording_id, cache_dir, context)
    recording, _ = load_coordinate_cache(cache_dir / f"{recording_id}.npz")
    inventory_index = list(context["inventory"]).index(recording_id)
    geometry = ReferenceTrackerConfig(**context["geometry"])
    rows = []
    for stride in context["strides"]:
        selected = benchmark_frame_indices(recording.frames, int(stride))
        seed = int(context["base_seed"]) + 1000 * inventory_index + int(stride)
        sequence = identity_blind_sequence(recording, frame_indices=selected, seed=seed)
        for specification in task["specifications"]:
            name = str(specification["name"])
            signature = _canonical_digest({"implementation_signature": context["implementation_signature"],
                "coordinate_sha256": audit["semantic_coordinate_sha256"], "recording_id": recording_id,
                "inventory_index": inventory_index, "stride": stride, "permutation_seed": seed,
                "specification": specification, "phase": phase})
            path = checkpoint_path(output_dir, phase, recording_id, int(stride), name)
            previous = read_checkpoint(path, signature)
            if previous is not None:
                rows.append(previous)
                continue
            started = time.perf_counter()
            row: dict[str, object] = {"schema_version": 1, "task_signature": signature,
                "implementation_signature": context["implementation_signature"], "phase": phase,
                "recording_id": recording_id, "cohort": task["cohort"], "full_inventory_index": inventory_index,
                "acquisition_date": recording_id.split("_ses-")[1][:8], "stride": int(stride), "permutation_seed": seed,
                "candidate": name, "method": specification["method"], "config": specification.get("config", {}),
                "cache_sha256": audit["cache_sha256"], "semantic_coordinate_sha256": audit["semantic_coordinate_sha256"],
                "frames": len(selected), "neurons": recording.neurons, "first_source_frame": int(selected[0]),
                "last_source_frame": int(selected[-1]), "identity_semantics": IDENTITY_SEMANTICS,
                "semantic_labels_used_for_association": False, "fluorescence_used_for_association": False,
                **_latency_metadata(recording.times_seconds, selected, specification)}
            try:
                result = _run_method(sequence, specification, geometry)
                _validate_exhaustive_result(result, recording.neurons, len(selected))
                metrics = score_tracks(sequence, result)
                opportunities = (metrics.frames - 1) * metrics.neurons
                row.update({"status": "complete", "correct": metrics.correct, "total": metrics.total,
                    "identity_accuracy": metrics.identity_accuracy, "seeded_idf1": metrics.seeded_idf1,
                    "identity_switches": metrics.identity_switches, "identity_switch_opportunities": opportunities,
                    "identity_switch_rate": metrics.identity_switches / opportunities if opportunities else 0.0,
                    "high_motion_identity_accuracy": metrics.severe_identity_accuracy,
                    "runtime_seconds": metrics.runtime_seconds,
                    "assignment_sha256": hashlib.sha256(np.asarray(result.assignments, dtype="<i8").tobytes()).hexdigest()})
            except Exception as error:
                row.update({"status": "failed", "error_type": type(error).__name__, "error": str(error),
                            "traceback": traceback.format_exc()})
            row["wall_seconds"] = time.perf_counter() - started
            _atomic_json(path, row)
            print(f"{phase} {recording_id} stride={stride} {name} {row['status']} "
                  f"accuracy={row.get('identity_accuracy')} seconds={row['wall_seconds']:.2f}", flush=True)
            rows.append(row)
    return rows


def _index_rows(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, int, str], Mapping[str, object]]:
    index = {}
    for row in rows:
        key = (str(row["recording_id"]), int(row["stride"]), str(row["candidate"]))
        if key in index:
            raise ValueError(f"duplicate result row {key}")
        index[key] = row
    return index


def select_tuning_candidates(rows: Sequence[Mapping[str, object]], candidates: Sequence[Mapping[str, object]],
                             *, tuning_recording: str, strides: Sequence[int]) -> dict[str, object]:
    """Select one complete candidate per family without looking at evaluation."""
    if any(row["recording_id"] != tuning_recording or row.get("phase") != "tune" for row in rows):
        raise ValueError("selection may consume only the single tuning recording")
    indexed = _index_rows(rows)
    leaderboard = []
    for specification in candidates:
        name = str(specification["name"])
        selected = [indexed.get((tuning_recording, int(stride), name)) for stride in strides]
        complete = all(row is not None and row.get("status") == "complete" for row in selected)
        entry: dict[str, object] = {"candidate": name, "method": specification["method"], "config": specification.get("config", {}),
            "family": specification.get("family", specification["method"]), "control": specification.get("control", False),
            "complete": complete, "expected_tasks": len(strides),
            "completed_tasks": sum(row is not None and row.get("status") == "complete" for row in selected)}
        if complete:
            entry.update({"mean_identity_accuracy": float(np.mean([row["identity_accuracy"] for row in selected])),
                          "mean_identity_switch_rate": float(np.mean([row["identity_switch_rate"] for row in selected])),
                          "runtime_seconds": float(sum(float(row["runtime_seconds"]) for row in selected))})
        leaderboard.append(entry)
    rank = lambda entry: (-float(entry["mean_identity_accuracy"]), float(entry["mean_identity_switch_rate"]),
                          float(entry["runtime_seconds"]), str(entry["candidate"]))
    complete_entries = sorted([entry for entry in leaderboard if entry["complete"] and not entry["control"]], key=rank)
    family_winners = []
    families = dict.fromkeys(str(candidate.get("family", candidate["method"])) for candidate in candidates if not candidate.get("control", False))
    for family_name in families:
        family = [entry for entry in complete_entries if entry["family"] == family_name]
        if not family:
            raise RuntimeError(f"no complete tuning candidate is available for {family_name}; inspect failed checkpoints")
        winner = family[0]
        family_winners.append(next(dict(candidate) for candidate in candidates if candidate["name"] == winner["candidate"]))
    return {"schema_version": 1, "selection_recording": tuning_recording,
            "selection_rule": "highest equal-stride mean identity accuracy; ties: lowest mean switch rate, lowest runtime, lexicographic name",
            "selected_candidates": family_winners, "global_winner": complete_entries[0]["candidate"],
            "leaderboard": sorted(leaderboard, key=lambda entry: (not entry["complete"], rank(entry) if entry["complete"] else (str(entry["candidate"]),))),
            "selection_uses_evaluation_outcomes": False, "strides": list(strides)}


def _bootstrap_difference(values: Sequence[float], samples: int, seed: int) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    if not len(array) or array.ndim != 1 or np.any(~np.isfinite(array)):
        raise ValueError("paired recording differences must be finite and nonempty")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return {"mean": float(array.mean()), "ci95_low": float(np.quantile(draws, .025)),
            "ci95_high": float(np.quantile(draws, .975)), "recordings": len(array),
            "bootstrap_samples": samples, "bootstrap_seed": seed, "per_recording_values": array.tolist()}


def summarize_comparisons(rows: Sequence[Mapping[str, object]], *, recording_ids: Sequence[str],
                          strides: Sequence[int], candidate_names: Sequence[str], bootstrap_samples: int,
                          bootstrap_seed: int) -> dict[str, object]:
    """Equal-stride, equal-recording estimands and explicit complete-pair coverage."""
    indexed = _index_rows(rows)
    method_summaries = {}
    recording_summary = {}
    for candidate in (*BASELINES, *candidate_names):
        recording_summary[candidate] = {}
        for recording in recording_ids:
            selected = [indexed.get((recording, int(stride), candidate)) for stride in strides]
            if not all(row is not None and row.get("status") == "complete" for row in selected):
                continue
            recording_summary[candidate][recording] = {
                "mean_identity_accuracy": float(np.mean([row["identity_accuracy"] for row in selected])),
                "mean_identity_switch_rate": float(np.mean([row["identity_switch_rate"] for row in selected])),
                "identity_switches": sum(int(row["identity_switches"]) for row in selected),
                "runtime_seconds": sum(float(row["runtime_seconds"]) for row in selected),
                "maximum_latency_selected_scans": max(int(row["latency_selected_scans"]) for row in selected),
            }
        complete_records = recording_summary[candidate]
        summary: dict[str, object] = {"expected_recordings": len(recording_ids), "complete_recordings": len(complete_records),
            "complete": len(complete_records) == len(recording_ids),
            "completed_tasks": sum(indexed.get((recording, int(stride), candidate), {}).get("status") == "complete"
                                   for recording in recording_ids for stride in strides),
            "expected_tasks": len(recording_ids) * len(strides), "per_recording": complete_records}
        if complete_records:
            summary.update({"recording_macro_mean_accuracy": float(np.mean([row["mean_identity_accuracy"] for row in complete_records.values()])),
                            "recording_macro_mean_switch_rate": float(np.mean([row["mean_identity_switch_rate"] for row in complete_records.values()])),
                            "total_runtime_seconds": float(sum(row["runtime_seconds"] for row in complete_records.values()))})
        method_summaries[candidate] = summary
    comparisons = {}
    for candidate in candidate_names:
        for baseline in BASELINES:
            paired_ids = [recording for recording in recording_ids if recording in recording_summary[candidate] and recording in recording_summary[baseline]]
            comparison: dict[str, object] = {"candidate": candidate, "baseline": baseline,
                "expected_recordings": len(recording_ids), "paired_complete_recordings": len(paired_ids),
                "complete": len(paired_ids) == len(recording_ids), "recording_ids": paired_ids,
                "estimand": "equal-recording mean of within-recording equal-stride differences",
                "inferential_unit": "recording; stride observations overlap and are not independent replicates"}
            if paired_ids:
                candidate_values = [recording_summary[candidate][recording] for recording in paired_ids]
                baseline_values = [recording_summary[baseline][recording] for recording in paired_ids]
                accuracy = [a["mean_identity_accuracy"] - b["mean_identity_accuracy"] for a, b in zip(candidate_values, baseline_values)]
                switch_rate = [b["mean_identity_switch_rate"] - a["mean_identity_switch_rate"] for a, b in zip(candidate_values, baseline_values)]
                comparison["accuracy_difference"] = _bootstrap_difference(accuracy, bootstrap_samples, bootstrap_seed)
                comparison["switch_rate_reduction"] = _bootstrap_difference(switch_rate, bootstrap_samples, bootstrap_seed)
                comparison["recordings_with_positive_accuracy_difference"] = sum(value > 0 for value in accuracy)
                comparison["recordings_with_negative_accuracy_difference"] = sum(value < 0 for value in accuracy)
                comparison["pooled_identity_switches"] = {"candidate": sum(row["identity_switches"] for row in candidate_values),
                                                          "baseline": sum(row["identity_switches"] for row in baseline_values),
                                                          "interpretation": "descriptive totals over overlapping stride sequences"}
                baseline_runtime = sum(row["runtime_seconds"] for row in baseline_values)
                comparison["runtime_ratio"] = sum(row["runtime_seconds"] for row in candidate_values) / baseline_runtime if baseline_runtime else None
                comparison["extra_future_scan_information"] = any(row["maximum_latency_selected_scans"] > 0 for row in candidate_values)
                comparison["improvement_criterion_met"] = bool(comparison["complete"] and comparison["accuracy_difference"]["ci95_low"] > 0)
            comparisons[f"{candidate}_vs_{baseline}"] = comparison
    return {"methods": method_summaries, "comparisons": comparisons}


def _baseline_reproduction(rows: Sequence[Mapping[str, object]], context: Mapping[str, object]) -> dict[str, object]:
    protocol = context["protocol"]
    assert isinstance(protocol, dict)
    dataset = protocol.get("dataset", {})
    assert isinstance(dataset, dict)
    path = Path(str(context["root"])) / str(dataset.get("frozen_results", "outputs/paper/source-results/prospective.json"))
    if not path.is_file():
        return {"available": False}
    payload = _read_mapping(path)
    original = {(str(row["recording_id"]), int(row["stride"]), str(row["method"])): row for row in payload["results"]}
    mismatches = []
    compared = 0
    for row in rows:
        if row["candidate"] not in BASELINES or row.get("status") != "complete":
            continue
        key = (str(row["recording_id"]), int(row["stride"]), str(row["method"]))
        if key not in original:
            continue
        compared += 1
        for field in ("permutation_seed", "correct", "total", "identity_switches"):
            if row[field] != original[key][field]:
                mismatches.append({"key": list(key), "field": field, "expected": original[key][field], "observed": row[field]})
    return {"available": True, "source_sha256": sha256_file(path), "compared_rows": compared,
            "all_compared_identity_outcomes_match": not mismatches, "mismatches": mismatches,
            "runtime_compared": False}


def _execute_tasks(tasks: Sequence[Mapping[str, object]], workers: int,
                   progress: Callable[[str], None] | None = print) -> list[dict[str, object]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    rows = []
    expanded_tasks = []
    for task in tasks:
        context = task["context"]
        assert isinstance(context, dict)
        # Extraction precedes parallel dispatch so workers never race to create
        # the same compact cache. Individual method/stride tasks then distribute
        # the expensive full-rate scans across available server CPUs.
        ensure_coordinate_cache(str(task["recording_id"]), Path(str(task["cache_dir"])), context)
        for stride in context["strides"]:
            for specification in task["specifications"]:
                expanded_tasks.append({**task, "context": {**context, "strides": [stride]},
                                       "specifications": [specification]})
    if workers == 1:
        for task in expanded_tasks:
            rows.extend(_benchmark_recording_task(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(_benchmark_recording_task, task): task for task in expanded_tasks}
            for future in as_completed(pending):
                rows.extend(future.result())
                if progress is not None:
                    task = pending[future]
                    progress(f"completed {task['recording_id']} stride={task['context']['strides'][0]} {task['specifications'][0]['name']}")
    return sorted(rows, key=lambda row: (int(row["full_inventory_index"]), int(row["stride"]), str(row["candidate"])))


def _load_phase_rows(output_dir: Path, phase: str, implementation_signature: str) -> list[dict[str, object]]:
    rows = []
    for path in sorted((output_dir / "checkpoints" / phase).glob("*.json")):
        row = _read_mapping(path)
        if row.get("implementation_signature") != implementation_signature:
            raise RuntimeError(f"checkpoint implementation provenance changed: {path}")
        rows.append(row)
    return rows


def run_experiment(*, repository_root: str | Path, protocol_path: str | Path,
                   cache_dir: str | Path, output_dir: str | Path, phase: str = "all", workers: int = 1) -> dict[str, object]:
    context = load_experiment_context(repository_root, protocol_path)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    context_path = destination / "experiment_context.json"
    if context_path.is_file() and _read_mapping(context_path).get("implementation_signature") != context["implementation_signature"]:
        raise RuntimeError("experiment provenance changed; preserve existing outcomes and use a new output directory")
    _atomic_json(context_path, context)
    baseline_specs = [{"name": method, "method": method, "config": {}} for method in BASELINES]
    selection_path = destination / "selection.json"
    if phase in ("tune", "all"):
        tuning_tasks = [{"context": context, "phase": "tune", "recording_id": context["tuning_recording"],
            "cohort": "tuning", "cache_dir": str(Path(cache_dir).resolve()), "output_dir": str(destination),
            "specifications": [*baseline_specs, *context["candidates"]]}]
        tuning_rows = _execute_tasks(tuning_tasks, workers)
        _atomic_json(destination / "tuning_results.json", {"context": context, "results": tuning_rows})
        selection = select_tuning_candidates(tuning_rows, context["candidates"],
            tuning_recording=str(context["tuning_recording"]), strides=context["strides"])
        selection["implementation_signature"] = context["implementation_signature"]
        selection["protocol_sha256"] = context["protocol_sha256"]
        _atomic_json(selection_path, selection)
        if phase == "tune":
            return selection
    if not selection_path.is_file():
        raise FileNotFoundError("evaluation requires a frozen tuning selection.json; run phase tune first")
    selection = _read_mapping(selection_path)
    if selection.get("implementation_signature") != context["implementation_signature"]:
        raise RuntimeError("frozen selection provenance differs from the current implementation")
    expected_candidates = {str(candidate["name"]): candidate for candidate in context["candidates"]}
    selected = selection["selected_candidates"]
    if any(expected_candidates.get(str(candidate["name"])) != candidate for candidate in selected):
        raise RuntimeError("selected candidate configurations do not match the frozen protocol")
    if phase in ("evaluate", "all"):
        evaluation_tasks = []
        for recording in sorted((*context["legacy_recordings"], *context["previous_prospective_recordings"])):
            evaluation_tasks.append({"context": context, "phase": "evaluate", "recording_id": recording,
                "cohort": "legacy_evaluation" if recording in context["legacy_recordings"] else "previous_prospective",
                "cache_dir": str(Path(cache_dir).resolve()), "output_dir": str(destination),
                "specifications": [*baseline_specs, *selected]})
        rows = _execute_tasks(evaluation_tasks, workers)
    else:
        rows = _load_phase_rows(destination, "evaluate", str(context["implementation_signature"]))
    evaluation = context["protocol"].get("evaluation", {})
    samples = int(evaluation.get("bootstrap_samples", 20_000))
    seed = int(evaluation.get("bootstrap_seed", 20261002))
    names = [str(candidate["name"]) for candidate in selected]
    groups = {"legacy_evaluation": context["legacy_recordings"], "previous_prospective": context["previous_prospective_recordings"],
              "all_nontuning": sorted((*context["legacy_recordings"], *context["previous_prospective_recordings"]))}
    summaries = {group: summarize_comparisons(rows, recording_ids=recordings, strides=context["strides"],
                    candidate_names=names, bootstrap_samples=samples, bootstrap_seed=seed)
                 for group, recordings in groups.items()}
    reproduction = _baseline_reproduction(rows, context)
    payload = {"schema_version": 1, "analysis_name": "PyRecEst follow-up comparison on previously examined WormID recordings",
        "context": context, "selection": selection, "selection_sha256": sha256_file(selection_path), "results": rows,
        "cohort_summaries": summaries, "baseline_reproduction": reproduction,
        "outcome_audit": {"new_untouched_replication": False, "selection_uses_evaluation_outcomes": False,
            "expected_evaluation_tasks": 20 * len(context["strides"]) * (len(BASELINES) + len(selected)),
            "complete_evaluation_tasks": sum(row.get("status") == "complete" for row in rows),
            "failed_tasks": [{key: row.get(key) for key in ("recording_id", "stride", "candidate", "error_type", "error")} for row in rows if row.get("status") == "failed"],
            "baseline_identity_outcomes_reproduced": reproduction.get("all_compared_identity_outcomes_match"),
            "downstream_calcium_evaluated": False,
            "downstream_omission_reason": "compact coordinate caches do not contain fluorescence traces",
            "latency_warning": "MHT with positive fixed_lag uses future selected scans; interpret separately from zero-lag comparisons"}}
    _atomic_json(destination / "evaluation_results.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("tune", "evaluate", "all", "summary"), default="all")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    result = run_experiment(repository_root=args.repository_root, protocol_path=args.protocol,
        cache_dir=args.cache_dir, output_dir=args.output_dir, phase=args.phase, workers=args.workers)
    print(json.dumps({"phase": args.phase, "output_dir": str(args.output_dir.resolve()),
                      "selected_candidates": result.get("selected_candidates", result.get("selection", {}).get("selected_candidates")),
                      "outcome_audit": result.get("outcome_audit")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
