"""Reproducible point-tracking baselines for the fDNC/NeRVE sequence.

The released ``real_jeff_all`` arrays contain one acquired volume per file.  A
row is ``(x, y, z, manual_identity)`` and a negative identity means that the
detection is not manually identified.  This module never exposes identities to
the trackers; they are used only by :func:`evaluate_partial_identity_tracks`.

Two kinds of result are deliberately kept separate:

* ``link_*`` evaluates association between adjacent *available* volumes.  This
  is a pairwise correspondence result.
* ``partial_idf1`` requires one globally consistent predicted identity over the
  complete sequence.  It is a temporal tracking result, evaluated only on the
  manually labelled subset.

The baselines are intentionally small and label-free.  ``nearest`` greedily
selects short one-to-one links, ``hungarian`` globally optimizes Euclidean
links, and ``affine_hungarian`` first estimates a trimmed, ridge-regularized
affine ICP warp before applying the same gated Hungarian association.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


_FRAME_NUMBER = re.compile(r"(?:^|_)(\d+)$")
_METHODS = ("nearest", "hungarian", "affine_hungarian")


@dataclass(frozen=True)
class FDNCFrame:
    """One fDNC point-cloud volume in the released normalized coordinates."""

    frame_id: int
    points_xyz: np.ndarray
    labels: np.ndarray
    path: Path


@dataclass(frozen=True)
class SequentialTracks:
    """Persistent track IDs and adjacent-frame parent links.

    ``parent_indices[k][j]`` is the index in frame ``k - 1`` associated with
    detection ``j`` in frame ``k``.  A value of ``-1`` denotes a birth.  Thus
    these records describe sequential association, not independent matching to
    a labelled template.
    """

    method: str
    frame_ids: tuple[int, ...]
    track_ids: tuple[np.ndarray, ...]
    parent_indices: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PartialIdentityMetrics:
    """Tracking metrics restricted to detections with manual identities."""

    method: str
    frames: int
    first_frame: int
    last_frame: int
    raw_frame_gaps_gt_one: int
    detections: int
    labeled_detections: int
    gt_identities: int
    predicted_tracks: int
    true_adjacent_links: int
    correct_adjacent_links: int
    wrong_labeled_adjacent_links: int
    missed_adjacent_links: int
    link_precision: float
    link_recall: float
    link_f1: float
    partial_idtp: int
    partial_idfp_labeled: int
    partial_idfn: int
    partial_idf1: float
    mean_track_fragments_per_identity: float
    median_track_fragments_per_identity: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _frame_id_from_path(path: Path) -> int:
    match = _FRAME_NUMBER.search(path.stem)
    if match is None:
        raise ValueError(f"cannot parse a numeric frame ID from {path.name!r}")
    return int(match.group(1))


def read_fdnc_frame(path: str | Path) -> FDNCFrame:
    """Read and validate one released fDNC ``.npy`` point cloud.

    The first three columns are normalized XYZ coordinates (not micrometres).
    Non-negative integer values in column four are manual identities; all
    negative values are treated as unlabelled.
    """

    path = Path(path)
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"{path} must have shape (N, 4), got {values.shape}")
    if len(values) == 0:
        raise ValueError(f"{path} contains no detections")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{path} must contain only finite numeric values")

    points = np.asarray(values[:, :3], dtype=np.float64)
    raw_labels = np.asarray(values[:, 3], dtype=np.float64)
    rounded_labels = np.rint(raw_labels)
    if not np.allclose(raw_labels, rounded_labels, rtol=0.0, atol=1e-9):
        raise ValueError(f"{path} contains non-integer manual identities")
    labels = rounded_labels.astype(np.int64)
    labeled = labels[labels >= 0]
    if len(np.unique(labeled)) != len(labeled):
        raise ValueError(f"{path} repeats a manual identity within one frame")

    return FDNCFrame(
        frame_id=_frame_id_from_path(path),
        points_xyz=points,
        labels=labels,
        path=path,
    )


def load_fdnc_sequence(
    directory: str | Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[FDNCFrame]:
    """Load a sequence in numeric frame order, never lexicographic order."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when supplied")
    directory = Path(directory)
    candidates = list(directory.glob("*.npy"))
    if not candidates:
        raise FileNotFoundError(f"no .npy point clouds found in {directory}")

    numbered = sorted((_frame_id_from_path(path), path) for path in candidates)
    frame_ids = [item[0] for item in numbered]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"{directory} contains duplicate numeric frame IDs")
    if start_frame is not None:
        numbered = [item for item in numbered if item[0] >= start_frame]
    if end_frame is not None:
        numbered = [item for item in numbered if item[0] <= end_frame]
    numbered = numbered[::stride]
    if max_frames is not None:
        numbered = numbered[:max_frames]
    if not numbered:
        raise ValueError("frame filters selected an empty sequence")
    return [read_fdnc_frame(path) for _, path in numbered]


def summarize_fdnc_sequence(frames: Sequence[FDNCFrame]) -> dict[str, float | int]:
    """Return compact integrity statistics without using manual labels to track."""

    _validate_sequence(frames)
    counts = np.asarray([len(frame.points_xyz) for frame in frames])
    labeled_counts = np.asarray([np.count_nonzero(frame.labels >= 0) for frame in frames])
    frame_ids = np.asarray([frame.frame_id for frame in frames])
    labels = np.concatenate([frame.labels[frame.labels >= 0] for frame in frames])
    return {
        "frames": int(len(frames)),
        "first_frame": int(frame_ids[0]),
        "last_frame": int(frame_ids[-1]),
        "raw_frame_gaps_gt_one": int(np.count_nonzero(np.diff(frame_ids) > 1)),
        "detections": int(counts.sum()),
        "labeled_detections": int(labeled_counts.sum()),
        "gt_identities": int(len(np.unique(labels))),
        "mean_detections_per_frame": float(counts.mean()),
        "median_detections_per_frame": float(np.median(counts)),
        "mean_labeled_per_frame": float(labeled_counts.mean()),
        "median_labeled_per_frame": float(np.median(labeled_counts)),
    }


def _validate_sequence(frames: Sequence[FDNCFrame]) -> None:
    if not frames:
        raise ValueError("at least one frame is required")
    frame_ids = [frame.frame_id for frame in frames]
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise ValueError("frames must be in strictly increasing numeric order")
    for frame in frames:
        if frame.points_xyz.ndim != 2 or frame.points_xyz.shape[1] != 3:
            raise ValueError("each points_xyz array must have shape (N, 3)")
        if frame.labels.shape != (len(frame.points_xyz),):
            raise ValueError("each labels array must align with points_xyz")


def _greedy_nearest_assignment(
    distances: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily take shortest admissible edges with one-to-one constraints."""

    candidate_rows, candidate_cols = np.nonzero(distances <= max_distance)
    if not len(candidate_rows):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    order = np.argsort(distances[candidate_rows, candidate_cols], kind="stable")
    used_rows = np.zeros(distances.shape[0], dtype=bool)
    used_cols = np.zeros(distances.shape[1], dtype=bool)
    selected_rows: list[int] = []
    selected_cols: list[int] = []
    for edge_index in order:
        row = int(candidate_rows[edge_index])
        col = int(candidate_cols[edge_index])
        if not used_rows[row] and not used_cols[col]:
            selected_rows.append(row)
            selected_cols.append(col)
            used_rows[row] = True
            used_cols[col] = True
    return np.asarray(selected_rows), np.asarray(selected_cols)


def _gated_hungarian_assignment(
    distances: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a partial one-to-one assignment with a geometry-only gate.

    One dummy destination per source gives every source an unmatched option.
    A real edge has benefit ``max_distance - distance`` over that option, which
    yields a maximum-weight matching over exactly the admissible edges.
    """

    if distances.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    n_source, n_destination = distances.shape
    augmented = np.full((n_source, n_destination + n_source), max_distance, dtype=np.float64)
    augmented[:, :n_destination] = np.where(
        distances <= max_distance, distances, max_distance + 1.0
    )
    rows, cols = linear_sum_assignment(augmented)
    real = cols < n_destination
    rows = rows[real]
    cols = cols[real]
    admissible = distances[rows, cols] <= max_distance
    return rows[admissible], cols[admissible]


def _robust_affine_warp(
    source_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    *,
    iterations: int,
    trim_fraction: float,
    ridge: float,
) -> np.ndarray:
    """Warp ``source_xyz`` using correspondence-free, trimmed affine ICP.

    Ridge regularization shrinks the affine map toward the identity, while the
    lowest-residual fraction of provisional Hungarian pairs is used to suppress
    missing detections and incorrect correspondences.  This is an association
    preprocessing step, not a use of manual identities.
    """

    if iterations < 0:
        raise ValueError("affine_iterations cannot be negative")
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError("affine_trim_fraction must be in (0, 1]")
    if ridge < 0:
        raise ValueError("affine_ridge cannot be negative")
    if len(source_xyz) < 4 or len(destination_xyz) < 4 or iterations == 0:
        return np.asarray(source_xyz, dtype=np.float64).copy()

    affine = np.eye(3)
    translation = np.median(destination_xyz, axis=0) - np.median(source_xyz, axis=0)
    identity_parameters = np.vstack((np.eye(3), np.zeros((1, 3))))
    regularizer = np.diag([ridge, ridge, ridge, 0.0])

    for _ in range(iterations):
        warped = source_xyz @ affine + translation
        distances = cdist(warped, destination_xyz)
        rows, cols = linear_sum_assignment(distances)
        residuals = distances[rows, cols]
        cutoff = np.quantile(residuals, trim_fraction)
        keep = residuals <= cutoff
        if np.count_nonzero(keep) < 4:
            break
        design = np.column_stack((source_xyz[rows[keep]], np.ones(np.count_nonzero(keep))))
        targets = destination_xyz[cols[keep]]
        normal = design.T @ design + regularizer
        right_hand_side = design.T @ targets + regularizer @ identity_parameters
        try:
            parameters = np.linalg.solve(normal, right_hand_side)
        except np.linalg.LinAlgError:
            parameters = np.linalg.lstsq(normal, right_hand_side, rcond=None)[0]
        candidate_affine = parameters[:3]
        if not np.isfinite(parameters).all() or abs(np.linalg.det(candidate_affine)) < 1e-6:
            break
        affine = candidate_affine
        translation = parameters[3]
    return source_xyz @ affine + translation


def track_fdnc_sequence(
    frames: Sequence[FDNCFrame],
    *,
    method: str = "hungarian",
    max_distance: float = 0.15,
    affine_iterations: int = 4,
    affine_trim_fraction: float = 0.70,
    affine_ridge: float = 0.03,
) -> SequentialTracks:
    """Associate detections sequentially without reading their manual labels.

    Track state is propagated only from the immediately previous available
    volume.  Consequently this compact baseline has no explicit re-identification
    after a missed detection and no multiple-hypothesis logic.
    """

    _validate_sequence(frames)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be finite and positive")

    next_track_id = len(frames[0].points_xyz)
    track_ids: list[np.ndarray] = [np.arange(next_track_id, dtype=np.int64)]
    parent_indices: list[np.ndarray] = [np.full(len(frames[0].points_xyz), -1, dtype=np.int64)]

    for previous, current in zip(frames, frames[1:]):
        source = previous.points_xyz
        destination = current.points_xyz
        if method == "affine_hungarian":
            association_source = _robust_affine_warp(
                source,
                destination,
                iterations=affine_iterations,
                trim_fraction=affine_trim_fraction,
                ridge=affine_ridge,
            )
        else:
            association_source = source
        distances = cdist(association_source, destination)
        if method == "nearest":
            source_indices, destination_indices = _greedy_nearest_assignment(
                distances, max_distance
            )
        else:
            source_indices, destination_indices = _gated_hungarian_assignment(
                distances, max_distance
            )

        current_track_ids = np.full(len(destination), -1, dtype=np.int64)
        current_parents = np.full(len(destination), -1, dtype=np.int64)
        current_track_ids[destination_indices] = track_ids[-1][source_indices]
        current_parents[destination_indices] = source_indices
        births = np.flatnonzero(current_track_ids < 0)
        current_track_ids[births] = np.arange(
            next_track_id, next_track_id + len(births), dtype=np.int64
        )
        next_track_id += len(births)
        track_ids.append(current_track_ids)
        parent_indices.append(current_parents)

    return SequentialTracks(
        method=method,
        frame_ids=tuple(frame.frame_id for frame in frames),
        track_ids=tuple(track_ids),
        parent_indices=tuple(parent_indices),
    )


def evaluate_partial_identity_tracks(
    frames: Sequence[FDNCFrame], tracks: SequentialTracks
) -> PartialIdentityMetrics:
    """Evaluate adjacent links and temporal identities on manual labels only."""

    _validate_sequence(frames)
    if tuple(frame.frame_id for frame in frames) != tracks.frame_ids:
        raise ValueError("track frame IDs do not match the evaluated sequence")
    if len(tracks.track_ids) != len(frames) or len(tracks.parent_indices) != len(frames):
        raise ValueError("tracks must contain one record per frame")
    for frame, ids, parents in zip(frames, tracks.track_ids, tracks.parent_indices):
        if ids.shape != (len(frame.points_xyz),) or parents.shape != (len(frame.points_xyz),):
            raise ValueError("track arrays must align with frame detections")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("a predicted track occurs more than once in one frame")

    true_links = 0
    correct_links = 0
    wrong_labeled_links = 0
    for previous, current, parents in zip(frames, frames[1:], tracks.parent_indices[1:]):
        previous_labeled = set(previous.labels[previous.labels >= 0].tolist())
        current_labeled = set(current.labels[current.labels >= 0].tolist())
        true_links += len(previous_labeled & current_labeled)
        for current_index, previous_index in enumerate(parents):
            if previous_index < 0:
                continue
            previous_label = int(previous.labels[previous_index])
            current_label = int(current.labels[current_index])
            if previous_label < 0 or current_label < 0:
                continue
            if previous_label == current_label:
                correct_links += 1
            else:
                wrong_labeled_links += 1
    missed_links = true_links - correct_links
    evaluated_predicted_links = correct_links + wrong_labeled_links
    link_precision = correct_links / evaluated_predicted_links if evaluated_predicted_links else 1.0
    link_recall = correct_links / true_links if true_links else 1.0
    link_denominator = 2 * correct_links + wrong_labeled_links + missed_links
    link_f1 = 2 * correct_links / link_denominator if link_denominator else 1.0

    all_labels = np.concatenate([frame.labels for frame in frames])
    all_track_ids = np.concatenate(tracks.track_ids)
    labeled_mask = all_labels >= 0
    labels = all_labels[labeled_mask]
    predicted = all_track_ids[labeled_mask]
    gt_values, gt_inverse = np.unique(labels, return_inverse=True)
    predicted_values, predicted_inverse = np.unique(predicted, return_inverse=True)
    overlap = np.zeros((len(gt_values), len(predicted_values)), dtype=np.int64)
    np.add.at(overlap, (gt_inverse, predicted_inverse), 1)
    gt_assignment, predicted_assignment = linear_sum_assignment(-overlap)
    idtp = int(overlap[gt_assignment, predicted_assignment].sum())
    total_labeled = int(len(labels))
    idfp_labeled = total_labeled - idtp
    idfn = total_labeled - idtp
    id_denominator = 2 * idtp + idfp_labeled + idfn
    partial_idf1 = 2 * idtp / id_denominator if id_denominator else 1.0

    fragment_counts = np.asarray(
        [len(np.unique(predicted[labels == identity])) for identity in gt_values],
        dtype=np.float64,
    )
    frame_ids = np.asarray([frame.frame_id for frame in frames])
    return PartialIdentityMetrics(
        method=tracks.method,
        frames=len(frames),
        first_frame=int(frame_ids[0]),
        last_frame=int(frame_ids[-1]),
        raw_frame_gaps_gt_one=int(np.count_nonzero(np.diff(frame_ids) > 1)),
        detections=int(sum(len(frame.points_xyz) for frame in frames)),
        labeled_detections=total_labeled,
        gt_identities=int(len(gt_values)),
        predicted_tracks=int(len(np.unique(all_track_ids))),
        true_adjacent_links=int(true_links),
        correct_adjacent_links=int(correct_links),
        wrong_labeled_adjacent_links=int(wrong_labeled_links),
        missed_adjacent_links=int(missed_links),
        link_precision=float(link_precision),
        link_recall=float(link_recall),
        link_f1=float(link_f1),
        partial_idtp=idtp,
        partial_idfp_labeled=idfp_labeled,
        partial_idfn=idfn,
        partial_idf1=float(partial_idf1),
        mean_track_fragments_per_identity=float(fragment_counts.mean()),
        median_track_fragments_per_identity=float(np.median(fragment_counts)),
    )


def benchmark_fdnc(
    directory: str | Path,
    *,
    methods: Iterable[str] = _METHODS,
    start_frame: int | None = None,
    end_frame: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    max_distance: float = 0.15,
    affine_iterations: int = 4,
    affine_trim_fraction: float = 0.70,
    affine_ridge: float = 0.03,
) -> dict[str, object]:
    """Load once, run requested baselines, and return a JSON-safe result."""

    frames = load_fdnc_sequence(
        directory,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        max_frames=max_frames,
    )
    results: dict[str, object] = {
        "dataset": summarize_fdnc_sequence(frames),
        "protocol": {
            "coordinate_units": "released normalized XYZ units (not micrometres)",
            "labels_used_for_tracking": False,
            "negative_labels": "ignored only during evaluation",
            "association": "sequential between adjacent available volumes",
            "max_distance": max_distance,
            "partial_idf1_scope": "all manually labelled detections in the loaded sequence",
        },
        "methods": {},
    }
    method_results: dict[str, object] = results["methods"]  # type: ignore[assignment]
    for method in methods:
        started = time.perf_counter()
        tracks = track_fdnc_sequence(
            frames,
            method=method,
            max_distance=max_distance,
            affine_iterations=affine_iterations,
            affine_trim_fraction=affine_trim_fraction,
            affine_ridge=affine_ridge,
        )
        metrics = evaluate_partial_identity_tracks(frames, tracks).to_dict()
        metrics["runtime_seconds"] = float(time.perf_counter() - started)
        method_results[method] = metrics
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run label-free sequential baselines on released fDNC tracking points."
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--method",
        choices=("all", *_METHODS),
        default="all",
        help="baseline to run (default: all)",
    )
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-distance", type=float, default=0.15)
    parser.add_argument("--affine-iterations", type=int, default=4)
    parser.add_argument("--affine-trim-fraction", type=float, default=0.70)
    parser.add_argument("--affine-ridge", type=float, default=0.03)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON destination; written atomically after a successful run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; invoke with ``python -m deformeot.fdnc``."""

    args = _parse_args(argv)
    methods = _METHODS if args.method == "all" else (args.method,)
    result = benchmark_fdnc(
        args.data_dir,
        methods=methods,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        max_frames=args.max_frames,
        max_distance=args.max_distance,
        affine_iterations=args.affine_iterations,
        affine_trim_fraction=args.affine_trim_fraction,
        affine_ridge=args.affine_ridge,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
