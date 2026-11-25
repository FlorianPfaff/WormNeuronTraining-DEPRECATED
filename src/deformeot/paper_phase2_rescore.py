"""Independently rescore archived, initialized fixed-cohort assignments.

This module intentionally imports neither tracking implementations nor their
evaluators. The algorithm uses explicit per-track/per-frame counting, making it
independent of the vectorized historical scoring implementation.

NPZ contract (pickle is never enabled): ``assignments[T,N]`` and
``truth_indices[T,N]`` are signed int64 permutations, ``frame_indices[T]`` are
strictly increasing nonnegative signed int64 source indices, and
``frame_times[T]`` are finite strictly increasing seconds. The assignment maps
persistent track rows to anonymous observation rows; truth maps observations to
canonical ROI indices. Initialization must assign track i to observation i.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


CONTRACT_VERSION = 1
REQUIRED_ARRAYS = ("assignments", "truth_indices", "frame_indices", "frame_times")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(array: np.ndarray, *, integer: bool) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<i8" if integer else "<f8").tobytes()).hexdigest()


def _int64(array: np.ndarray, name: str, ndim: int) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != ndim or value.dtype.kind != "i" or value.dtype.itemsize != 8:
        raise ValueError(f"{name} must be a {ndim}-dimensional signed int64 array")
    return value


def validate_assignment_arrays(assignments: np.ndarray, truth_indices: np.ndarray,
                               frame_indices: np.ndarray, frame_times: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assignments = _int64(assignments, "assignments", 2)
    truth_indices = _int64(truth_indices, "truth_indices", 2)
    frame_indices = _int64(frame_indices, "frame_indices", 1)
    if not assignments.shape[0] or not assignments.shape[1]:
        raise ValueError("at least one initialized frame and one track are required")
    if assignments.shape != truth_indices.shape:
        raise ValueError("truth_indices and assignments must have the same [T,N] shape")
    frames, neurons = assignments.shape
    times = np.asarray(frame_times)
    if times.shape != (frames,) or times.dtype.kind not in "fi" or not np.isfinite(times).all():
        raise ValueError("frame_times must contain one finite numeric time per frame")
    times = np.asarray(times, dtype=np.float64)
    if not np.isfinite(times).all() or any(float(right) <= float(left) for left, right in zip(times, times[1:])):
        raise ValueError("frame_times must be strictly increasing in seconds")
    if frame_indices.shape != (frames,) or any(int(index) < 0 for index in frame_indices):
        raise ValueError("frame_indices must contain one nonnegative source index per frame")
    if any(int(right) <= int(left) for left, right in zip(frame_indices, frame_indices[1:])):
        raise ValueError("frame_indices must be strictly increasing")
    expected = list(range(neurons))
    for name, values in (("assignments", assignments), ("truth_indices", truth_indices)):
        for index, row in enumerate(values):
            if sorted(int(value) for value in row) != expected:
                raise ValueError(f"{name} frame {index} is not an exhaustive permutation of 0..N-1")
    if [int(value) for value in assignments[0]] != expected:
        raise ValueError("initialized frame assignments must equal arange(N)")
    return assignments, truth_indices, frame_indices, times


def score_assignment_arrays(assignments: np.ndarray, truth_indices: np.ndarray,
                            frame_indices: np.ndarray, frame_times: np.ndarray) -> dict[str, Any]:
    """Count identity agreement, changes, and maximal runs of wrong identities.

    Wrong-episode duration in selected scans is the number of consecutive wrong
    observations. Secondary elapsed-time summaries use two explicit conventions:
    observed span (last wrong minus first wrong time), and right-endpoint-weighted
    exposure (sum preceding sampling intervals for wrong observations). Neither
    imputes a latent biological error onset or extrapolates beyond the final scan.
    """
    assignments, truth, indices, times = validate_assignment_arrays(assignments, truth_indices, frame_indices, frame_times)
    frames, neurons = assignments.shape
    correct = switches = wrong_frames = 0
    episode_lengths: list[int] = []
    episode_spans: list[float] = []
    episode_exposures: list[float] = []
    for track in range(neurons):
        initial_identity = int(truth[0, track])
        previous_identity: int | None = None
        episode_start: int | None = None
        episode_exposure = 0.0
        for frame in range(frames):
            identity = int(truth[frame, int(assignments[frame, track])])
            if previous_identity is not None and identity != previous_identity:
                switches += 1
            previous_identity = identity
            if identity == initial_identity:
                correct += 1
                if episode_start is not None:
                    episode_lengths.append(frame - episode_start)
                    episode_spans.append(float(times[frame - 1] - times[episode_start]))
                    episode_exposures.append(episode_exposure)
                    episode_start = None
                    episode_exposure = 0.0
            else:
                wrong_frames += 1
                if episode_start is None:
                    episode_start = frame
                # Legal initialization guarantees that a wrong frame is >0.
                episode_exposure += float(times[frame] - times[frame - 1])
        if episode_start is not None:
            episode_lengths.append(frames - episode_start)
            episode_spans.append(float(times[-1] - times[episode_start]))
            episode_exposures.append(episode_exposure)

    total = frames * neurons
    excluded_total = (frames - 1) * neurons
    excluded_correct = correct - neurons
    episodes = len(episode_lengths)
    if sum(episode_lengths) != wrong_frames or correct + wrong_frames != total:
        raise AssertionError("independent episode accounting does not conserve observations")
    return {
        "contract_version": CONTRACT_VERSION,
        "frames": frames, "neurons": neurons,
        "correct": correct, "total": total, "identity_accuracy": correct / total,
        "correct_excluding_initialization": excluded_correct,
        "total_excluding_initialization": excluded_total,
        "accuracy_excluding_initialization": excluded_correct / excluded_total if excluded_total else None,
        "identity_switches": switches, "identity_switch_opportunities": excluded_total,
        "identity_switch_rate": switches / excluded_total if excluded_total else 0.0,
        "coverage": 1.0, "assignment_entries": total, "unassigned_entries": 0,
        "initialized_frame_correct": neurons, "initialized_frame_total": neurons,
        "wrong_identity_frames": wrong_frames, "wrong_identity_episode_count": episodes,
        "mean_wrong_identity_duration_selected_scans": sum(episode_lengths) / episodes if episodes else 0.0,
        "maximum_wrong_identity_duration_selected_scans": max(episode_lengths, default=0),
        "sum_wrong_identity_duration_selected_scans": sum(episode_lengths),
        "mean_wrong_identity_observed_span_seconds": sum(episode_spans) / episodes if episodes else 0.0,
        "maximum_wrong_identity_observed_span_seconds": max(episode_spans, default=0.0),
        "right_endpoint_weighted_wrong_exposure_seconds": sum(episode_exposures),
        "mean_wrong_episode_right_endpoint_exposure_seconds": sum(episode_exposures) / episodes if episodes else 0.0,
        "recording_observed_span_seconds": float(times[-1] - times[0]),
        "source_first_frame": int(indices[0]), "source_last_frame": int(indices[-1]),
        "assignment_sha256": _array_hash(assignments, integer=True),
        "truth_indices_sha256": _array_hash(truth, integer=True),
        "frame_indices_sha256": _array_hash(indices, integer=True),
        "frame_times_sha256": _array_hash(times, integer=False),
        "identity_semantics": "agreement with initialized canonical ROI identities; not semantic neuron names",
        "switch_semantics": "number of tracks changing assigned identity across adjacent selected scans, not swap events",
        "time_semantics": "observed wrong spans and right-endpoint-weighted exposure; no inferred within-interval onset or extrapolated tail",
        "scorer_uses_tracking_implementation": False,
    }


def rescore_file(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    actual_hash = file_sha256(path)
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(expected_sha256)):
            raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
        if actual_hash != expected_sha256.lower():
            raise ValueError(f"assignment artifact SHA256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED_ARRAYS) - set(archive.files)
        if missing:
            raise ValueError(f"assignment artifact missing arrays {sorted(missing)}: {path}")
        metrics = score_assignment_arrays(*(archive[name] for name in REQUIRED_ARRAYS))
    return {"path": str(path), "sha256": actual_hash,
            "manifest_hash_verified": expected_sha256 is not None, "metrics": metrics}


def _manifest_entries(path: Path) -> dict[Path, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("files"), list):
        raise ValueError("manifest must be a JSON object with a files list of path/sha256 entries")
    entries: dict[Path, str] = {}
    for row in document["files"]:
        if not isinstance(row, dict) or "path" not in row or "sha256" not in row:
            raise ValueError("each manifest entry requires path and sha256")
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest paths must be relative without parent traversal")
        artifact = (path.parent / relative).resolve()
        if not artifact.is_relative_to(path.parent.resolve()):
            raise ValueError("manifest path resolves outside its directory")
        if artifact in entries:
            raise ValueError(f"duplicate manifest artifact: {relative}")
        expected = str(row["sha256"])
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"invalid manifest SHA256: {relative}")
        entries[artifact] = expected.lower()
    return entries


def rescore_path(path: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Rescore one NPZ or all NPZs recursively, optionally enforcing a manifest.

    Manifest paths are relative to its containing directory. When rescoring a
    directory, all discovered NPZs require manifest entries and any declared
    in-scope NPZ that is missing raises rather than silently reducing the cohort.
    """
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    entries = _manifest_entries(Path(manifest_path).resolve()) if manifest_path is not None else None
    artifacts = sorted(source.rglob("*.npz")) if source.is_dir() else [source]
    artifacts = [artifact.resolve() for artifact in artifacts]
    if not artifacts or any(artifact.suffix != ".npz" for artifact in artifacts):
        raise ValueError("input must be an NPZ artifact or a directory containing NPZ artifacts")
    if len(set(artifacts)) != len(artifacts):
        raise ValueError("duplicate assignment artifacts through aliased paths")
    if entries is not None:
        for artifact in artifacts:
            if artifact not in entries:
                raise ValueError(f"assignment artifact is missing from manifest: {artifact}")
        declared = [artifact for artifact in entries if artifact.suffix == ".npz"
                    and (artifact == source or source.is_dir() and artifact.is_relative_to(source))]
        for artifact in declared:
            if not artifact.is_file():
                raise FileNotFoundError(f"declared assignment artifact missing: {artifact}")
    rows = [rescore_file(artifact, entries[artifact] if entries is not None else None) for artifact in artifacts]
    return {"schema_version": 1, "status": "complete", "artifacts": rows,
            "artifact_count": len(rows), "manifest": str(Path(manifest_path).resolve()) if manifest_path else None,
            "manifest_sha256": file_sha256(manifest_path) if manifest_path else None,
            "aggregation": "per-artifact metrics only; animal-level analysis belongs to the frozen study protocol"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="assignment NPZ or directory")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = rescore_path(args.path, args.manifest)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        destination = args.output.resolve()
        input_paths = {Path(row["path"]) for row in result["artifacts"]}
        if destination in input_paths or destination == args.path.resolve() or args.manifest is not None and destination == args.manifest.resolve():
            parser.error("output must not overwrite the assignment artifact or integrity manifest")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(destination)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
