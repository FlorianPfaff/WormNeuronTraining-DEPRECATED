"""Tracking metrics with explicit support for partially labeled scenes."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class PartialTrackingMetrics:
    """Metrics for a labeled subset, ignoring unmatched predicted detections.

    ``modified_mota`` follows ASCENT's published protocol: false positives are not
    counted because ground truth covers only a known subset of visible neurons.
    ``partial_idf1`` similarly counts identity errors only among predictions that
    match a labeled ground-truth point.
    """

    modified_mota: float
    partial_idf1: float
    total_gt: int
    matched_detections: int
    misses: int
    mismatches: int
    idtp: int
    idfp_labeled: int
    idfn: int
    localization_rmse_um: float
    localization_median_um: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _validate_track_frame(frame: pd.DataFrame, name: str) -> None:
    required = {"TrackID", "ObjectID", "t", "z", "y", "x"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame.duplicated(["TrackID", "t"]).any():
        raise ValueError(f"{name} contains more than one detection per track and frame")


def mutual_nearest_matches(
    gt_zyx: np.ndarray,
    pred_zyx: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match mutually nearest 3D points within ``threshold``.

    This intentionally reproduces the correspondence rule described by ASCENT
    rather than substituting a globally optimal Hungarian match.
    """

    gt_zyx = np.asarray(gt_zyx, dtype=float)
    pred_zyx = np.asarray(pred_zyx, dtype=float)
    if gt_zyx.ndim != 2 or gt_zyx.shape[1] != 3:
        raise ValueError("gt_zyx must have shape (N, 3)")
    if pred_zyx.ndim != 2 or pred_zyx.shape[1] != 3:
        raise ValueError("pred_zyx must have shape (M, 3)")
    if len(gt_zyx) == 0 or len(pred_zyx) == 0:
        empty_i = np.empty(0, dtype=int)
        return empty_i, empty_i.copy(), np.empty(0, dtype=float)

    distances = cdist(gt_zyx, pred_zyx)
    gt_to_pred = distances.argmin(axis=1)
    pred_to_gt = distances.argmin(axis=0)
    gt_indices = np.arange(len(gt_zyx), dtype=int)
    mutual = pred_to_gt[gt_to_pred] == gt_indices
    within = distances[gt_indices, gt_to_pred] <= threshold
    keep = mutual & within
    matched_gt = gt_indices[keep]
    matched_pred = gt_to_pred[keep]
    return matched_gt, matched_pred, distances[matched_gt, matched_pred]


def _frame_matches(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    spacing_zyx: Iterable[float],
    threshold_um: float,
) -> pd.DataFrame:
    spacing = np.asarray(tuple(spacing_zyx), dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three positive values")

    records: list[dict[str, object]] = []
    pred_frames = {int(t): part for t, part in pred.groupby("t", sort=False)}
    for t, gt_t in gt.groupby("t", sort=True):
        pred_t = pred_frames.get(int(t))
        if pred_t is None:
            continue
        gt_pos = gt_t[["z", "y", "x"]].to_numpy(float) * spacing
        pred_pos = pred_t[["z", "y", "x"]].to_numpy(float) * spacing
        gi, pi, distances = mutual_nearest_matches(gt_pos, pred_pos, threshold=threshold_um)
        for g_idx, p_idx, distance_um in zip(gi, pi, distances):
            g = gt_t.iloc[int(g_idx)]
            p = pred_t.iloc[int(p_idx)]
            records.append(
                {
                    "t": int(t),
                    "gt_object": str(g["ObjectID"]),
                    "gt_track": str(g["TrackID"]),
                    "pred_object": str(p["ObjectID"]),
                    "pred_track": str(p["TrackID"]),
                    "distance_um": float(distance_um),
                }
            )
    return pd.DataFrame.from_records(records)


def _ascent_mismatch_count(gt: pd.DataFrame, matches: pd.DataFrame) -> int:
    """Reproduce ASCENT's history-based mismatch counting convention."""

    matches_by_frame = {int(t): part for t, part in matches.groupby("t", sort=True)}
    track_map: dict[str, str] = {}
    seen_gt: set[str] = set()
    mismatches = 0

    for t in sorted(int(x) for x in gt["t"].unique()):
        current = matches_by_frame.get(t)
        if current is None:
            continue
        updates: dict[str, str] = {}
        for row in current.itertuples(index=False):
            pred_id = str(row.pred_track)
            gt_id = str(row.gt_track)
            previous_gt = track_map.get(pred_id)
            if gt_id in seen_gt:
                if previous_gt != gt_id:
                    mismatches += 1
            elif previous_gt is not None:
                mismatches += 1
            updates[pred_id] = gt_id
            seen_gt.add(gt_id)
        track_map.update(updates)
    return mismatches


def _global_identity_counts(
    gt: pd.DataFrame, matches: pd.DataFrame
) -> tuple[int, int, int]:
    """Compute ID counts after the optimal global track-ID permutation."""

    gt_ids = sorted(gt["TrackID"].astype(str).unique())
    pred_ids = sorted(matches["pred_track"].astype(str).unique()) if len(matches) else []
    if not gt_ids or not pred_ids:
        return 0, int(len(matches)), int(len(gt))

    gt_index = {key: i for i, key in enumerate(gt_ids)}
    pred_index = {key: i for i, key in enumerate(pred_ids)}
    overlap = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    for (gt_id, pred_id), count in Counter(
        zip(matches["gt_track"].astype(str), matches["pred_track"].astype(str))
    ).items():
        overlap[gt_index[gt_id], pred_index[pred_id]] = count
    row_ind, col_ind = linear_sum_assignment(-overlap)
    idtp = int(overlap[row_ind, col_ind].sum())
    idfp_labeled = int(len(matches) - idtp)
    idfn = int(len(gt) - idtp)
    return idtp, idfp_labeled, idfn


def evaluate_partial_tracks(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    spacing_zyx: Iterable[float],
    threshold_um: float = 5.0,
) -> tuple[PartialTrackingMetrics, pd.DataFrame]:
    """Evaluate point tracks when only a subset of scene objects are labeled."""

    _validate_track_frame(gt, "ground truth")
    _validate_track_frame(pred, "prediction")
    if threshold_um <= 0:
        raise ValueError("threshold_um must be positive")

    matches = _frame_matches(
        gt,
        pred,
        spacing_zyx=spacing_zyx,
        threshold_um=threshold_um,
    )
    misses = int(len(gt) - len(matches))
    mismatches = _ascent_mismatch_count(gt, matches)
    idtp, idfp_labeled, idfn = _global_identity_counts(gt, matches)
    denominator = 2 * idtp + idfp_labeled + idfn
    partial_idf1 = 2 * idtp / denominator if denominator else 1.0
    distances = matches["distance_um"].to_numpy(float) if len(matches) else np.array([])
    localization_rmse = float(np.sqrt(np.mean(distances**2))) if len(distances) else float("nan")
    localization_median = float(np.median(distances)) if len(distances) else float("nan")
    result = PartialTrackingMetrics(
        modified_mota=1.0 - (misses + mismatches) / len(gt) if len(gt) else 1.0,
        partial_idf1=float(partial_idf1),
        total_gt=int(len(gt)),
        matched_detections=int(len(matches)),
        misses=misses,
        mismatches=int(mismatches),
        idtp=idtp,
        idfp_labeled=idfp_labeled,
        idfn=idfn,
        localization_rmse_um=localization_rmse,
        localization_median_um=localization_median,
    )
    return result, matches

