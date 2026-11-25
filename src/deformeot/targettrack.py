"""Reproducible evaluation of the public Targettrack volumetric demo.

The Targettrack HDF5 file stores identity-labelled 3-D ground-truth masks in
``/<frame>/mask`` and network outputs in
``/net/<network>/<frame>/predmask``.  This module deliberately evaluates those
mask outputs as *segmentation and extent measurements*.  It does not infer
end-to-end temporal tracking performance from them.

The public demo has three useful recorded partitions: five training frames,
100 validation frames, and annotated frames that appear in neither recorded
set.  The last partition is called ``unseen`` here rather than ``test`` because
the archive does not declare it as a formal blind test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_NETWORK = "CZANet_Final"
DEFAULT_VOXEL_SPACING = (1.0, 1.0, 1.0)


def _float_tuple(values: np.ndarray | Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _matrix_tuple(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(_float_tuple(row) for row in values)


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _finite_median(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _relative_l2(estimate: Sequence[float], truth: Sequence[float]) -> float:
    estimate_array = np.asarray(estimate, dtype=float)
    truth_array = np.asarray(truth, dtype=float)
    denominator = float(np.linalg.norm(truth_array))
    numerator = float(np.linalg.norm(estimate_array - truth_array))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _sha256(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MaskObject:
    """Spatial summary of one positive label in a 3-D mask.

    Coordinates follow the HDF5 array axes.  Targettrack names those axes
    ``W, H, D``; no physical voxel spacing is stored in the public demo, so
    callers should provide spacing if physical distances are required.
    ``covariance`` is the population second central moment of foreground voxel
    centres and is the extended-object scatter descriptor used here.
    """

    identity: int
    voxel_count: int
    centroid: tuple[float, float, float]
    bbox_extent: tuple[float, float, float]
    covariance: tuple[tuple[float, float, float], ...]
    principal_std: tuple[float, float, float]


@dataclass(frozen=True)
class AnonymousObservation:
    """One prediction-mask instance with its numerical label made frame-local.

    ``source_instance_label`` is retained solely so the evaluator can recover
    the pixels belonging to this per-frame instance.  The temporal tracker
    never reads it.  In particular, a value such as 7 in adjacent frames is
    not treated as the same identity.
    """

    local_index: int
    source_instance_label: int
    voxel_count: int
    centroid: tuple[float, float, float]
    bbox_extent: tuple[float, float, float]
    log_extent: tuple[float, float, float, float]


@dataclass(frozen=True)
class SpatialMatch:
    """Identity-blind per-frame evaluation match between prediction and GT."""

    local_index: int
    gt_identity: int
    iou: float


@dataclass(frozen=True)
class TemporalTrackerParameters:
    """Parameters shared by the centroid and centroid+extent trackers."""

    centroid_scale: float = 4.0
    velocity_alpha: float = 0.3
    velocity_damping_on_miss: float = 0.5
    gap_scale_growth: float = 0.1
    unmatched_cost: float = 4.5
    extent_weight: float = 0.0
    extent_alpha: float = 0.5


@dataclass(frozen=True)
class TemporalAssignment:
    """A canonical track assigned to one anonymous per-frame observation."""

    frame_id: int
    local_index: int
    track_identity: int
    pre_assignment_miss_streak: int
    association_cost: float


@dataclass(frozen=True)
class ObjectMetric:
    """Identity-conditioned prediction metrics for one GT object instance."""

    frame_id: int
    split: str
    identity: int
    gt_voxels: int
    pred_voxels: int
    intersection_voxels: int
    union_voxels: int
    detected: bool
    iou: float
    dice: float
    centroid_error: float
    volume_absolute_relative_error: float
    bbox_relative_l2_error: float
    covariance_relative_l2_error: float
    principal_std_relative_l2_error: float
    gt_centroid: tuple[float, float, float]
    pred_centroid: tuple[float, float, float] | None
    gt_bbox_extent: tuple[float, float, float]
    pred_bbox_extent: tuple[float, float, float]
    gt_principal_std: tuple[float, float, float]
    pred_principal_std: tuple[float, float, float]


@dataclass(frozen=True)
class TargettrackMetadata:
    """Validated layout and recorded data partitions of one HDF5 file."""

    path: str
    network: str
    frame_ids: tuple[int, ...]
    annotated_frame_ids: tuple[int, ...]
    prediction_frame_ids: tuple[int, ...]
    training_frame_ids: tuple[int, ...]
    validation_frame_ids: tuple[int, ...]
    unseen_frame_ids: tuple[int, ...]
    volume_shape: tuple[int, int, int]
    n_neurons_attribute: int


class TargettrackH5:
    """Validated, read-only access to a Targettrack-style HDF5 archive."""

    def __init__(self, path: str | Path, network: str | None = None):
        self.path = Path(path)
        self.requested_network = network
        self._h5: h5py.File | None = None
        self._metadata: TargettrackMetadata | None = None

    def __enter__(self) -> "TargettrackH5":
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._h5 = h5py.File(self.path, "r")
        try:
            self._metadata = self._inspect()
        except Exception:
            self._h5.close()
            self._h5 = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._metadata = None

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            raise RuntimeError("TargettrackH5 must be used as a context manager")
        return self._h5

    @property
    def metadata(self) -> TargettrackMetadata:
        if self._metadata is None:
            raise RuntimeError("TargettrackH5 must be used as a context manager")
        return self._metadata

    def _inspect(self) -> TargettrackMetadata:
        h5 = self.h5
        frame_ids = tuple(sorted(int(key) for key in h5.keys() if key.isdigit()))
        if not frame_ids:
            raise ValueError(f"{self.path} has no numeric frame groups")

        expected_t = int(h5.attrs.get("T", len(frame_ids)))
        if expected_t != len(frame_ids):
            raise ValueError(
                f"HDF5 T={expected_t}, but {len(frame_ids)} numeric frame groups were found"
            )

        volume_shape: tuple[int, int, int] | None = None
        annotated: list[int] = []
        for frame_id in frame_ids:
            group = h5[str(frame_id)]
            if "frame" not in group:
                raise ValueError(f"frame {frame_id} has no frame dataset")
            frame_shape = tuple(int(value) for value in group["frame"].shape)
            candidate = frame_shape[1:] if len(frame_shape) == 4 else frame_shape
            if len(candidate) != 3:
                raise ValueError(f"frame {frame_id} is not a 3-D volume: {frame_shape}")
            if volume_shape is None:
                volume_shape = candidate
            elif candidate != volume_shape:
                raise ValueError(
                    f"frame {frame_id} shape {candidate} differs from {volume_shape}"
                )
            if "mask" in group:
                if tuple(group["mask"].shape) != volume_shape:
                    raise ValueError(f"mask shape mismatch in frame {frame_id}")
                annotated.append(frame_id)

        if "net" not in h5:
            raise ValueError(f"{self.path} has no /net group")
        candidates = [
            name
            for name, group in h5["net"].items()
            if isinstance(group, h5py.Group)
            and any(
                key.isdigit() and isinstance(group[key], h5py.Group) and "predmask" in group[key]
                for key in group.keys()
            )
        ]
        if self.requested_network is None:
            if len(candidates) != 1:
                raise ValueError(
                    "network is ambiguous; choose one of " + ", ".join(sorted(candidates))
                )
            network = candidates[0]
        else:
            network = self.requested_network
            if network not in candidates:
                raise ValueError(
                    f"network {network!r} has no predictions; choices: {sorted(candidates)}"
                )

        network_group = h5[f"net/{network}"]
        predictions = tuple(
            sorted(
                int(key)
                for key in network_group.keys()
                if key.isdigit()
                and isinstance(network_group[key], h5py.Group)
                and "predmask" in network_group[key]
            )
        )
        for frame_id in predictions:
            if frame_id not in frame_ids:
                raise ValueError(f"prediction exists for unknown frame {frame_id}")
            if tuple(network_group[f"{frame_id}/predmask"].shape) != volume_shape:
                raise ValueError(f"prediction shape mismatch in frame {frame_id}")

        training = tuple(sorted(int(value) for value in network_group.attrs.get("traininds", ())))
        validation = tuple(
            sorted(int(value) for value in network_group.attrs.get("Validinds", ()))
        )
        annotated_set = set(annotated)
        if not set(training).issubset(annotated_set):
            raise ValueError("recorded training indices are not all annotated")
        if not set(validation).issubset(annotated_set):
            raise ValueError("recorded validation indices are not all annotated")
        if set(training).intersection(validation):
            raise ValueError("recorded training and validation indices overlap")
        unseen = tuple(sorted(annotated_set.difference(training).difference(validation)))

        assert volume_shape is not None
        return TargettrackMetadata(
            path=str(self.path.resolve()),
            network=network,
            frame_ids=frame_ids,
            annotated_frame_ids=tuple(annotated),
            prediction_frame_ids=predictions,
            training_frame_ids=training,
            validation_frame_ids=validation,
            unseen_frame_ids=unseen,
            volume_shape=volume_shape,
            n_neurons_attribute=int(h5.attrs.get("N_neurons", 0)),
        )

    def read_mask(self, frame_id: int, *, predicted: bool = False) -> np.ndarray:
        """Read and validate one identity-labelled mask as ``int32``."""

        metadata = self.metadata
        if frame_id not in metadata.frame_ids:
            raise KeyError(f"unknown frame {frame_id}")
        if predicted:
            key = f"net/{metadata.network}/{frame_id}/predmask"
        else:
            key = f"{frame_id}/mask"
        if key not in self.h5:
            kind = "prediction" if predicted else "ground-truth mask"
            raise KeyError(f"frame {frame_id} has no {kind}")
        mask = np.asarray(self.h5[key], dtype=np.int32)
        if mask.shape != metadata.volume_shape:
            raise ValueError(f"{key} has shape {mask.shape}, expected {metadata.volume_shape}")
        if np.any(mask < 0):
            raise ValueError(f"{key} contains negative identity labels")
        return mask


def extract_mask_objects(
    mask: np.ndarray,
    *,
    voxel_spacing: Sequence[float] = DEFAULT_VOXEL_SPACING,
) -> dict[int, MaskObject]:
    """Extract centroid and dynamic-extent statistics for all positive labels.

    The implementation touches only foreground voxels after the initial
    ``nonzero`` call, which keeps evaluation fast for sparse neural masks.
    """

    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"mask must be 3-D, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("mask labels must use an integer dtype")
    if np.any(array < 0):
        raise ValueError("mask contains negative labels")
    spacing = np.asarray(voxel_spacing, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("voxel_spacing must contain three positive finite values")

    coordinates = np.argwhere(array > 0)
    if not coordinates.size:
        return {}
    labels = array[tuple(coordinates.T)].astype(np.int64, copy=False)
    objects: dict[int, MaskObject] = {}
    for identity in np.unique(labels):
        object_coordinates = coordinates[labels == identity].astype(float) * spacing
        centroid = object_coordinates.mean(axis=0)
        centred = object_coordinates - centroid
        covariance = centred.T @ centred / len(object_coordinates)
        covariance = (covariance + covariance.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(covariance)
        principal_std = np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]
        bbox_extent = (
            object_coordinates.max(axis=0) - object_coordinates.min(axis=0) + spacing
        )
        objects[int(identity)] = MaskObject(
            identity=int(identity),
            voxel_count=int(len(object_coordinates)),
            centroid=_float_tuple(centroid),  # type: ignore[arg-type]
            bbox_extent=_float_tuple(bbox_extent),  # type: ignore[arg-type]
            covariance=_matrix_tuple(covariance),
            principal_std=_float_tuple(principal_std),  # type: ignore[arg-type]
        )
    return objects


def anonymize_prediction_mask(
    mask: np.ndarray,
    *,
    voxel_spacing: Sequence[float] = DEFAULT_VOXEL_SPACING,
) -> tuple[AnonymousObservation, ...]:
    """Convert identity-labelled output into an unordered instance set.

    Targettrack predicts a semantic channel per neuron, and those channels are
    needed to keep touching masks separate.  We therefore use each positive
    label only to partition pixels *within the current frame*.  Instances are
    then re-indexed by spatial order; the original numerical value is excluded
    from every temporal cost and state update.
    """

    objects = extract_mask_objects(mask, voxel_spacing=voxel_spacing)
    ordered = sorted(
        objects.items(),
        key=lambda item: (*item[1].centroid, item[1].voxel_count),
    )
    observations = []
    for local_index, (source_label, mask_object) in enumerate(ordered):
        log_extent = np.log(
            np.asarray(
                [mask_object.voxel_count, *mask_object.bbox_extent], dtype=float
            )
            + 0.5
        )
        observations.append(
            AnonymousObservation(
                local_index=local_index,
                source_instance_label=int(source_label),
                voxel_count=mask_object.voxel_count,
                centroid=mask_object.centroid,
                bbox_extent=mask_object.bbox_extent,
                log_extent=_float_tuple(log_extent),  # type: ignore[arg-type]
            )
        )
    return tuple(observations)


def match_anonymous_observations_to_gt(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    observations: Sequence[AnonymousObservation],
    *,
    iou_threshold: float = 1.0 / 3.0,
) -> tuple[SpatialMatch, ...]:
    """Match anonymous instances to GT with Hungarian IoU, without label equality.

    The prediction's numerical labels are used only to address the pixels of a
    local instance.  The cost matrix compares every such instance with every
    positive GT object, so a prediction labelled ``7`` can match any GT
    identity.  Matches must have IoU strictly above ``iou_threshold``.
    """

    gt = np.asarray(ground_truth)
    pred = np.asarray(prediction)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError("ground-truth and prediction masks must be equal-shape 3-D arrays")
    if not 0 <= iou_threshold < 1:
        raise ValueError("iou_threshold must lie in [0, 1)")
    gt_identities = sorted(int(value) for value in np.unique(gt) if value > 0)
    if not gt_identities or not observations:
        return ()

    gt_counts = {identity: int(np.count_nonzero(gt == identity)) for identity in gt_identities}
    iou = np.zeros((len(observations), len(gt_identities)), dtype=float)
    for observation_index, observation in enumerate(observations):
        pred_region = pred == observation.source_instance_label
        pred_count = observation.voxel_count
        overlapping_gt = gt[pred_region]
        identities, intersections = np.unique(overlapping_gt, return_counts=True)
        intersection_by_identity = {
            int(identity): int(count)
            for identity, count in zip(identities, intersections, strict=True)
            if identity > 0
        }
        for gt_index, identity in enumerate(gt_identities):
            intersection = intersection_by_identity.get(identity, 0)
            union = pred_count + gt_counts[identity] - intersection
            iou[observation_index, gt_index] = intersection / union if union else 0.0

    observation_indices, gt_indices = linear_sum_assignment(-iou)
    matches = [
        SpatialMatch(
            local_index=observations[observation_index].local_index,
            gt_identity=gt_identities[gt_index],
            iou=float(iou[observation_index, gt_index]),
        )
        for observation_index, gt_index in zip(
            observation_indices, gt_indices, strict=True
        )
        if iou[observation_index, gt_index] > iou_threshold
    ]
    return tuple(sorted(matches, key=lambda match: match.local_index))


def compare_identity_masks(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    frame_id: int,
    split: str,
    voxel_spacing: Sequence[float] = DEFAULT_VOXEL_SPACING,
) -> tuple[list[ObjectMetric], tuple[int, ...]]:
    """Compare same-identity objects and return false-positive identities.

    A prediction is evaluated against its explicit label, rather than being
    rematched by spatial overlap.  Missing predictions receive zero IoU/Dice
    and a zero extent; centroid error is undefined (NaN).
    """

    gt = np.asarray(ground_truth)
    pred = np.asarray(prediction)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError(f"mask shapes must be equal and 3-D, got {gt.shape} and {pred.shape}")
    if not np.issubdtype(gt.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer):
        raise TypeError("mask labels must use integer dtypes")
    if np.any(gt < 0) or np.any(pred < 0):
        raise ValueError("mask labels must be nonnegative")

    gt_objects = extract_mask_objects(gt, voxel_spacing=voxel_spacing)
    pred_objects = extract_mask_objects(pred, voxel_spacing=voxel_spacing)
    false_positive_identities = tuple(sorted(set(pred_objects).difference(gt_objects)))
    rows: list[ObjectMetric] = []
    zero_extent = (0.0, 0.0, 0.0)
    zero_covariance = np.zeros((3, 3), dtype=float)

    for identity, gt_object in sorted(gt_objects.items()):
        pred_object = pred_objects.get(identity)
        gt_region = gt == identity
        pred_region = pred == identity
        intersection = int(np.count_nonzero(gt_region & pred_region))
        union = gt_object.voxel_count + int(np.count_nonzero(pred_region)) - intersection
        pred_voxels = int(np.count_nonzero(pred_region))
        iou = intersection / union if union else 1.0
        dice_denominator = gt_object.voxel_count + pred_voxels
        dice = 2.0 * intersection / dice_denominator if dice_denominator else 1.0
        if pred_object is None:
            pred_centroid = None
            pred_bbox = zero_extent
            pred_principal = zero_extent
            pred_covariance = zero_covariance
            centroid_error = float("nan")
        else:
            pred_centroid = pred_object.centroid
            pred_bbox = pred_object.bbox_extent
            pred_principal = pred_object.principal_std
            pred_covariance = np.asarray(pred_object.covariance)
            centroid_error = float(
                np.linalg.norm(np.asarray(gt_object.centroid) - np.asarray(pred_centroid))
            )

        rows.append(
            ObjectMetric(
                frame_id=int(frame_id),
                split=str(split),
                identity=int(identity),
                gt_voxels=gt_object.voxel_count,
                pred_voxels=pred_voxels,
                intersection_voxels=intersection,
                union_voxels=union,
                detected=pred_object is not None,
                iou=float(iou),
                dice=float(dice),
                centroid_error=centroid_error,
                volume_absolute_relative_error=abs(pred_voxels - gt_object.voxel_count)
                / gt_object.voxel_count,
                bbox_relative_l2_error=_relative_l2(pred_bbox, gt_object.bbox_extent),
                covariance_relative_l2_error=_relative_l2(
                    pred_covariance.ravel(), np.asarray(gt_object.covariance).ravel()
                ),
                principal_std_relative_l2_error=_relative_l2(
                    pred_principal, gt_object.principal_std
                ),
                gt_centroid=gt_object.centroid,
                pred_centroid=pred_centroid,
                gt_bbox_extent=gt_object.bbox_extent,
                pred_bbox_extent=pred_bbox,
                gt_principal_std=gt_object.principal_std,
                pred_principal_std=pred_principal,
            )
        )
    return rows, false_positive_identities


def _split_name(frame_id: int, metadata: TargettrackMetadata) -> str:
    if frame_id in metadata.training_frame_ids:
        return "train"
    if frame_id in metadata.validation_frame_ids:
        return "validation"
    return "unseen"


def _cluster_bootstrap_difference(
    rows: Sequence[ObjectMetric],
    differences: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if n_bootstrap <= 0 or not rows:
        return (float("nan"), float("nan"))
    frame_ids = np.asarray([row.frame_id for row in rows], dtype=int)
    clusters = np.unique(frame_ids)
    by_cluster = [np.flatnonzero(frame_ids == frame_id) for frame_id in clusters]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([by_cluster[item] for item in selected])
        samples[index] = float(np.mean(differences[indices]))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return (float(lower), float(upper))


def _within_identity_volume_correlation(rows: Sequence[ObjectMetric]) -> float:
    gt_residuals: list[float] = []
    pred_residuals: list[float] = []
    identities = sorted({row.identity for row in rows})
    for identity in identities:
        selected = [row for row in rows if row.identity == identity]
        if len(selected) < 2:
            continue
        gt = np.asarray([row.gt_voxels for row in selected], dtype=float)
        pred = np.asarray([row.pred_voxels for row in selected], dtype=float)
        gt_residuals.extend(gt - gt.mean())
        pred_residuals.extend(pred - pred.mean())
    if len(gt_residuals) < 2:
        return float("nan")
    gt_array = np.asarray(gt_residuals)
    pred_array = np.asarray(pred_residuals)
    if np.std(gt_array) == 0 or np.std(pred_array) == 0:
        return float("nan")
    return float(np.corrcoef(gt_array, pred_array)[0, 1])


def _static_extent_baseline(
    rows: Sequence[ObjectMetric],
) -> dict[int, dict[str, tuple[float, ...] | float]]:
    baseline: dict[int, dict[str, tuple[float, ...] | float]] = {}
    for identity in sorted({row.identity for row in rows if row.split == "train"}):
        selected = [row for row in rows if row.split == "train" and row.identity == identity]
        baseline[identity] = {
            "volume": float(np.median([row.gt_voxels for row in selected])),
            "bbox_extent": _float_tuple(
                np.median(np.asarray([row.gt_bbox_extent for row in selected]), axis=0)
            ),
            "principal_std": _float_tuple(
                np.median(np.asarray([row.gt_principal_std for row in selected]), axis=0)
            ),
        }
    return baseline


def summarize_extent_information(
    rows: Sequence[ObjectMetric],
    static_baseline: Mapping[int, Mapping[str, tuple[float, ...] | float]],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | list[float]]:
    """Compare predicted dynamic extent to a per-identity static train template."""

    selected = [row for row in rows if row.identity in static_baseline]
    if not selected:
        return {"instances": 0}
    gt_volume = np.asarray([row.gt_voxels for row in selected], dtype=float)
    pred_volume = np.asarray([row.pred_voxels for row in selected], dtype=float)
    static_volume = np.asarray(
        [float(static_baseline[row.identity]["volume"]) for row in selected], dtype=float
    )
    dynamic_volume_error = np.abs(pred_volume - gt_volume) / gt_volume
    static_volume_error = np.abs(static_volume - gt_volume) / gt_volume
    volume_improvement = static_volume_error - dynamic_volume_error

    dynamic_log_error = np.abs(np.log((pred_volume + 0.5) / (gt_volume + 0.5)))
    static_log_error = np.abs(np.log((static_volume + 0.5) / (gt_volume + 0.5)))
    dynamic_bbox_error = np.asarray(
        [row.bbox_relative_l2_error for row in selected], dtype=float
    )
    static_bbox_error = np.asarray(
        [
            _relative_l2(
                static_baseline[row.identity]["bbox_extent"],  # type: ignore[arg-type]
                row.gt_bbox_extent,
            )
            for row in selected
        ],
        dtype=float,
    )
    ci = _cluster_bootstrap_difference(
        selected, volume_improvement, n_bootstrap=n_bootstrap, seed=seed
    )

    cvs = []
    for identity in sorted({row.identity for row in selected}):
        volumes = np.asarray(
            [row.gt_voxels for row in selected if row.identity == identity], dtype=float
        )
        if len(volumes) >= 2 and volumes.mean() > 0:
            cvs.append(float(volumes.std(ddof=0) / volumes.mean()))

    return {
        "instances": len(selected),
        "identities": len({row.identity for row in selected}),
        "detected_fraction": float(np.mean([row.detected for row in selected])),
        "gt_volume_median_within_identity_cv": float(np.median(cvs)) if cvs else float("nan"),
        "dynamic_volume_mean_absolute_relative_error": float(dynamic_volume_error.mean()),
        "static_train_median_volume_mean_absolute_relative_error": float(
            static_volume_error.mean()
        ),
        "dynamic_minus_static_volume_mare": float(
            dynamic_volume_error.mean() - static_volume_error.mean()
        ),
        "static_minus_dynamic_volume_mare": float(volume_improvement.mean()),
        "static_minus_dynamic_volume_mare_frame_bootstrap_ci95": [ci[0], ci[1]],
        "fraction_instances_dynamic_volume_better": float(np.mean(volume_improvement > 0)),
        "dynamic_volume_mean_absolute_log_error": float(dynamic_log_error.mean()),
        "static_train_median_volume_mean_absolute_log_error": float(static_log_error.mean()),
        "dynamic_bbox_mean_relative_l2_error": _finite_mean(dynamic_bbox_error),
        "static_train_median_bbox_mean_relative_l2_error": _finite_mean(
            static_bbox_error
        ),
        "within_identity_gt_pred_volume_correlation": _within_identity_volume_correlation(
            selected
        ),
    }


def summarize_object_metrics(
    rows: Sequence[ObjectMetric], false_positive_instances: int
) -> dict[str, float | int]:
    """Aggregate object metrics without hiding same-label misses."""

    if not rows:
        return {"frames": 0, "gt_instances": 0}
    intersections = sum(row.intersection_voxels for row in rows)
    unions = sum(row.union_voxels for row in rows)
    return {
        "frames": len({row.frame_id for row in rows}),
        "gt_instances": len(rows),
        "identities": len({row.identity for row in rows}),
        "detected_instances": sum(row.detected for row in rows),
        "same_label_misses": sum(not row.detected for row in rows),
        "false_positive_identity_instances": int(false_positive_instances),
        "detection_fraction": float(np.mean([row.detected for row in rows])),
        "mean_iou": float(np.mean([row.iou for row in rows])),
        "median_iou": float(np.median([row.iou for row in rows])),
        "micro_iou": float(intersections / unions) if unions else float("nan"),
        "mean_dice": float(np.mean([row.dice for row in rows])),
        "iou_above_one_third_fraction": float(np.mean([row.iou > 1.0 / 3.0 for row in rows])),
        "iou_at_least_half_fraction": float(np.mean([row.iou >= 0.5 for row in rows])),
        "centroid_error_mean_detected": _finite_mean(row.centroid_error for row in rows),
        "centroid_error_median_detected": _finite_median(row.centroid_error for row in rows),
        "volume_absolute_relative_error_mean": _finite_mean(
            row.volume_absolute_relative_error for row in rows
        ),
        "bbox_relative_l2_error_mean": _finite_mean(
            row.bbox_relative_l2_error for row in rows
        ),
        "covariance_relative_l2_error_mean": _finite_mean(
            row.covariance_relative_l2_error for row in rows
        ),
        "principal_std_relative_l2_error_mean": _finite_mean(
            row.principal_std_relative_l2_error for row in rows
        ),
    }


def _feature_vector(mask_object: MaskObject) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(mask_object.centroid, dtype=float),
            np.log(np.asarray([mask_object.voxel_count], dtype=float) + 0.5),
            np.log(np.asarray(mask_object.bbox_extent, dtype=float) + 0.5),
            np.log(np.asarray(mask_object.principal_std, dtype=float) + 0.5),
        ]
    )


def _association_accuracy_difference_ci(
    frame_records: Sequence[dict[str, int]],
    model: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if not frame_records or n_bootstrap <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    n_frames = len(frame_records)
    for bootstrap_index in range(n_bootstrap):
        selected = rng.integers(0, n_frames, size=n_frames)
        correct_model = sum(frame_records[index][model] for index in selected)
        correct_centroid = sum(frame_records[index]["centroid"] for index in selected)
        total = sum(frame_records[index]["total"] for index in selected)
        samples[bootstrap_index] = (correct_model - correct_centroid) / total
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def _association_temporal_block_ci(
    frame_records: Sequence[dict[str, int]],
    model: str,
    *,
    block_width_frames: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[tuple[float, float], int]:
    """Bootstrap fixed temporal blocks as a dependence-sensitivity analysis."""

    blocks: dict[int, dict[str, int]] = {}
    for record in frame_records:
        block_id = record["frame_id"] // block_width_frames
        aggregate = blocks.setdefault(
            block_id, {"centroid": 0, model: 0, "total": 0}
        )
        aggregate["centroid"] += record["centroid"]
        if model != "centroid":
            aggregate[model] += record[model]
        aggregate["total"] += record["total"]
    interval = _association_accuracy_difference_ci(
        list(blocks.values()), model, n_bootstrap=n_bootstrap, seed=seed
    )
    return interval, len(blocks)


def evaluate_extent_association(
    objects_by_frame: Mapping[int, Mapping[int, MaskObject]],
    metadata: TargettrackMetadata,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Ablate extent cues in identity assignment on identity-stripped GT masks.

    Per-identity prototypes are fitted on the recorded training frames.  Feature
    scales are supervised residual standard deviations on the recorded
    validation frames.  At each evaluation frame, Hungarian assignment maps
    the unlabeled GT objects to the training identities.  This isolates whether
    observed object extent can help association; it is not an evaluation of the
    supplied network's already identity-labelled predictions.
    """

    train_features: dict[int, list[np.ndarray]] = {}
    for frame_id in metadata.training_frame_ids:
        for identity, mask_object in objects_by_frame.get(frame_id, {}).items():
            train_features.setdefault(identity, []).append(_feature_vector(mask_object))
    if len(train_features) < 2:
        raise ValueError("extent association requires at least two training identities")
    prototypes = {
        identity: np.mean(features, axis=0) for identity, features in train_features.items()
    }

    calibration_residuals = []
    for frame_id in metadata.validation_frame_ids:
        for identity, mask_object in objects_by_frame.get(frame_id, {}).items():
            if identity in prototypes:
                calibration_residuals.append(_feature_vector(mask_object) - prototypes[identity])
    if not calibration_residuals:
        raise ValueError("extent association requires recorded validation objects for scaling")
    scales = np.std(np.asarray(calibration_residuals), axis=0, ddof=0)
    scales = np.maximum(scales, 1e-6)

    models = {
        "centroid": slice(0, 3),
        "centroid_plus_volume": slice(0, 4),
        "centroid_plus_volume_bbox": slice(0, 7),
        "centroid_plus_volume_bbox_principal_std": slice(0, 10),
    }
    prototype_ids = sorted(prototypes)
    prototype_matrix = np.asarray([prototypes[identity] for identity in prototype_ids])
    output: dict[str, object] = {
        "method": (
            "Hungarian assignment of identity-stripped GT objects to per-identity train "
            "prototypes; feature scales calibrated on recorded validation labels"
        ),
        "uses_network_predictions": False,
        "training_frames": list(metadata.training_frame_ids),
        "calibration_frames": list(metadata.validation_frame_ids),
        "feature_scale": scales.tolist(),
        "splits": {},
    }

    for split, frame_ids in {
        "validation": metadata.validation_frame_ids,
        "unseen": metadata.unseen_frame_ids,
    }.items():
        correct = {model: 0 for model in models}
        correct_by_identity = {
            model: {identity: 0 for identity in prototype_ids} for model in models
        }
        total_by_identity = {identity: 0 for identity in prototype_ids}
        total = 0
        frame_records: list[dict[str, int]] = []
        for frame_id in frame_ids:
            frame_objects = {
                identity: mask_object
                for identity, mask_object in objects_by_frame.get(frame_id, {}).items()
                if identity in prototypes
            }
            if not frame_objects:
                continue
            true_ids = sorted(frame_objects)
            feature_matrix = np.asarray(
                [_feature_vector(frame_objects[identity]) for identity in true_ids]
            )
            record = {model: 0 for model in models}
            record["frame_id"] = frame_id
            record["total"] = len(true_ids)
            for identity in true_ids:
                total_by_identity[identity] += 1
            for model, feature_slice in models.items():
                residual = (
                    feature_matrix[:, None, feature_slice]
                    - prototype_matrix[None, :, feature_slice]
                ) / scales[feature_slice]
                cost = np.sum(residual**2, axis=2)
                object_indices, prototype_indices = linear_sum_assignment(cost)
                model_correct = 0
                for object_index, prototype_index in zip(
                    object_indices, prototype_indices, strict=True
                ):
                    identity = true_ids[object_index]
                    is_correct = identity == prototype_ids[prototype_index]
                    model_correct += is_correct
                    correct_by_identity[model][identity] += int(is_correct)
                record[model] = int(model_correct)
                correct[model] += int(model_correct)
            total += len(true_ids)
            frame_records.append(record)

        model_output = {}
        for model in models:
            accuracy = correct[model] / total if total else float("nan")
            delta = (correct[model] - correct["centroid"]) / total if total else float("nan")
            ci = _association_accuracy_difference_ci(
                frame_records, model, n_bootstrap=n_bootstrap, seed=seed
            )
            block_ci, n_blocks = _association_temporal_block_ci(
                frame_records,
                model,
                block_width_frames=100,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            model_output[model] = {
                "correct": correct[model],
                "total": total,
                "accuracy": float(accuracy),
                "accuracy_delta_vs_centroid": float(delta),
                "accuracy_delta_vs_centroid_frame_bootstrap_ci95": [ci[0], ci[1]],
                "accuracy_delta_vs_centroid_100_frame_block_bootstrap_ci95": [
                    block_ci[0],
                    block_ci[1],
                ],
                "temporal_blocks": n_blocks,
            }
        identity_output = {}
        for identity in prototype_ids:
            identity_total = total_by_identity[identity]
            if not identity_total:
                continue
            identity_output[str(identity)] = {
                "total": identity_total,
                **{
                    model: {
                        "correct": correct_by_identity[model][identity],
                        "accuracy": correct_by_identity[model][identity] / identity_total,
                    }
                    for model in models
                },
            }
        output["splits"][split] = {  # type: ignore[index]
            "frames": len(frame_records),
            "objects": total,
            "models": model_output,
            "by_identity": identity_output,
        }
    return output


@dataclass
class _TemporalTrackState:
    identity: int
    position: np.ndarray
    velocity: np.ndarray
    log_extent: np.ndarray
    miss_streak: int = 0


def _initial_temporal_states(
    initial_objects: Mapping[int, MaskObject],
) -> list[_TemporalTrackState]:
    states = []
    for identity, mask_object in sorted(initial_objects.items()):
        log_extent = np.log(
            np.asarray(
                [mask_object.voxel_count, *mask_object.bbox_extent], dtype=float
            )
            + 0.5
        )
        states.append(
            _TemporalTrackState(
                identity=identity,
                position=np.asarray(mask_object.centroid, dtype=float),
                velocity=np.zeros(3, dtype=float),
                log_extent=log_extent,
            )
        )
    return states


def _track_one_direction(
    frame_ids: Sequence[int],
    observations_by_frame: Mapping[int, Sequence[AnonymousObservation]],
    initial_objects: Mapping[int, MaskObject],
    parameters: TemporalTrackerParameters,
    extent_scales: np.ndarray,
) -> dict[int, tuple[TemporalAssignment, ...]]:
    """Track in one chronological direction from a common initialized state."""

    states = _initial_temporal_states(initial_objects)
    output: dict[int, tuple[TemporalAssignment, ...]] = {}
    for frame_id in frame_ids:
        observations = tuple(observations_by_frame.get(frame_id, ()))
        predicted_positions = [state.position + state.velocity for state in states]
        n_tracks = len(states)
        n_observations = len(observations)
        cost = np.empty((n_tracks, n_observations), dtype=float)
        for track_index, state in enumerate(states):
            gap_multiplier = np.sqrt(
                1.0 + parameters.gap_scale_growth * state.miss_streak
            )
            centroid_scale = parameters.centroid_scale * gap_multiplier
            for observation_index, observation in enumerate(observations):
                centroid_residual = (
                    np.asarray(observation.centroid) - predicted_positions[track_index]
                )
                centroid_cost = float(
                    np.sum((centroid_residual / centroid_scale) ** 2)
                )
                extent_residual = (
                    np.asarray(observation.log_extent) - state.log_extent
                ) / extent_scales
                extent_cost = parameters.extent_weight * float(
                    np.mean(extent_residual**2)
                )
                cost[track_index, observation_index] = centroid_cost + extent_cost

        matched: list[tuple[int, int, float]] = []
        if n_observations:
            size = n_tracks + n_observations
            augmented = np.full((size, size), 1e9, dtype=float)
            augmented[:n_tracks, :n_observations] = cost
            for track_index in range(n_tracks):
                augmented[track_index, n_observations + track_index] = (
                    parameters.unmatched_cost
                )
            for observation_index in range(n_observations):
                augmented[n_tracks + observation_index, observation_index] = (
                    parameters.unmatched_cost
                )
            augmented[n_tracks:, n_observations:] = 0.0
            row_indices, column_indices = linear_sum_assignment(augmented)
            matched = [
                (int(row), int(column), float(cost[row, column]))
                for row, column in zip(row_indices, column_indices, strict=True)
                if row < n_tracks and column < n_observations
            ]

        matched_track_indices = {track_index for track_index, _, _ in matched}
        assignments = []
        for track_index, observation_index, association_cost in matched:
            state = states[track_index]
            observation = observations[observation_index]
            pre_miss_streak = state.miss_streak
            observation_position = np.asarray(observation.centroid, dtype=float)
            displacement = observation_position - state.position
            state.velocity = (
                (1.0 - parameters.velocity_alpha) * state.velocity
                + parameters.velocity_alpha * displacement
            )
            state.position = observation_position
            state.log_extent = (
                (1.0 - parameters.extent_alpha) * state.log_extent
                + parameters.extent_alpha * np.asarray(observation.log_extent)
            )
            state.miss_streak = 0
            assignments.append(
                TemporalAssignment(
                    frame_id=frame_id,
                    local_index=observation.local_index,
                    track_identity=state.identity,
                    pre_assignment_miss_streak=pre_miss_streak,
                    association_cost=association_cost,
                )
            )

        for track_index, state in enumerate(states):
            if track_index in matched_track_indices:
                continue
            state.position = predicted_positions[track_index]
            state.velocity *= parameters.velocity_damping_on_miss
            state.miss_streak += 1
        output[frame_id] = tuple(
            sorted(assignments, key=lambda assignment: assignment.local_index)
        )
    return output


def run_anonymous_temporal_tracker(
    observations_by_frame: Mapping[int, Sequence[AnonymousObservation]],
    initial_objects: Mapping[int, MaskObject],
    *,
    anchor_frame: int,
    frame_ids: Sequence[int],
    parameters: TemporalTrackerParameters,
    extent_scales: Sequence[float],
    anchor_spatial_matches: Sequence[SpatialMatch] = (),
) -> dict[int, tuple[TemporalAssignment, ...]]:
    """Run bidirectionally from one declared GT-initialized anchor frame."""

    scales = np.asarray(extent_scales, dtype=float)
    if scales.shape != (4,) or np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("extent_scales must contain four positive finite values")
    ordered_frames = tuple(sorted(int(frame_id) for frame_id in frame_ids))
    if anchor_frame not in ordered_frames:
        raise ValueError("anchor_frame is absent from frame_ids")
    if not initial_objects:
        raise ValueError("at least one canonical identity is required at the anchor")

    backward_frames = tuple(
        frame_id for frame_id in reversed(ordered_frames) if frame_id < anchor_frame
    )
    forward_frames = tuple(frame_id for frame_id in ordered_frames if frame_id > anchor_frame)
    backward = _track_one_direction(
        backward_frames,
        observations_by_frame,
        initial_objects,
        parameters,
        scales,
    )
    forward = _track_one_direction(
        forward_frames,
        observations_by_frame,
        initial_objects,
        parameters,
        scales,
    )

    anchor_match_by_local = {
        match.local_index: match.gt_identity for match in anchor_spatial_matches
    }
    anchor_assignments = tuple(
        TemporalAssignment(
            frame_id=anchor_frame,
            local_index=observation.local_index,
            track_identity=anchor_match_by_local[observation.local_index],
            pre_assignment_miss_streak=0,
            association_cost=0.0,
        )
        for observation in observations_by_frame.get(anchor_frame, ())
        if observation.local_index in anchor_match_by_local
        and anchor_match_by_local[observation.local_index] in initial_objects
    )
    return {**backward, anchor_frame: anchor_assignments, **forward}


def _global_identity_true_positives(confusion: Mapping[tuple[int, int], int]) -> int:
    if not confusion:
        return 0
    gt_ids = sorted({key[0] for key in confusion})
    track_ids = sorted({key[1] for key in confusion})
    counts = np.zeros((len(gt_ids), len(track_ids)), dtype=int)
    gt_index = {identity: index for index, identity in enumerate(gt_ids)}
    track_index = {identity: index for index, identity in enumerate(track_ids)}
    for (gt_identity, track_identity), count in confusion.items():
        counts[gt_index[gt_identity], track_index[track_identity]] = count
    rows, columns = linear_sum_assignment(-counts)
    return int(counts[rows, columns].sum())


def _temporal_frame_records(
    assignments_by_frame: Mapping[int, Sequence[TemporalAssignment]],
    spatial_matches_by_frame: Mapping[int, Sequence[SpatialMatch]],
    gt_objects_by_frame: Mapping[int, Mapping[int, MaskObject]],
    observations_by_frame: Mapping[int, Sequence[AnonymousObservation]],
    frame_ids: Sequence[int],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for frame_id in frame_ids:
        assignments = {
            assignment.local_index: assignment
            for assignment in assignments_by_frame.get(frame_id, ())
        }
        spatial_matches = {
            match.local_index: match for match in spatial_matches_by_frame.get(frame_id, ())
        }
        correct = 0
        assigned_spatial_matches = 0
        confusion: dict[tuple[int, int], int] = {}
        gaps: list[tuple[int | None, bool]] = []
        identities: list[tuple[int, int | None]] = []
        for local_index, spatial_match in spatial_matches.items():
            assignment = assignments.get(local_index)
            assigned_spatial_matches += assignment is not None
            assigned_identity = assignment.track_identity if assignment else None
            is_correct = assigned_identity == spatial_match.gt_identity
            correct += is_correct
            identities.append((spatial_match.gt_identity, assigned_identity))
            gaps.append(
                (
                    assignment.pre_assignment_miss_streak if assignment else None,
                    is_correct,
                )
            )
            if assigned_identity is not None:
                key = (spatial_match.gt_identity, assigned_identity)
                confusion[key] = confusion.get(key, 0) + 1
        records.append(
            {
                "frame_id": frame_id,
                "gt_total": len(gt_objects_by_frame.get(frame_id, {})),
                "observation_total": len(observations_by_frame.get(frame_id, ())),
                "assigned_total": len(assignments),
                "spatial_matches": len(spatial_matches),
                "assigned_spatial_matches": assigned_spatial_matches,
                "canonical_correct": correct,
                "confusion": confusion,
                "gaps": gaps,
                "identities": identities,
            }
        )
    return records


def _summarize_temporal_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    gt_total = sum(int(record["gt_total"]) for record in records)
    observation_total = sum(int(record["observation_total"]) for record in records)
    assigned_total = sum(int(record["assigned_total"]) for record in records)
    spatial_matches = sum(int(record["spatial_matches"]) for record in records)
    assigned_spatial_matches = sum(
        int(record["assigned_spatial_matches"]) for record in records
    )
    canonical_correct = sum(int(record["canonical_correct"]) for record in records)
    canonical_idfp = assigned_total - canonical_correct
    canonical_idfn = gt_total - canonical_correct
    canonical_denominator = 2 * canonical_correct + canonical_idfp + canonical_idfn

    confusion: dict[tuple[int, int], int] = {}
    for record in records:
        for key, count in record["confusion"].items():  # type: ignore[union-attr]
            confusion[key] = confusion.get(key, 0) + int(count)
    global_idtp = _global_identity_true_positives(confusion)
    global_idfp = assigned_total - global_idtp
    global_idfn = gt_total - global_idtp
    global_denominator = 2 * global_idtp + global_idfp + global_idfn

    previous_track_by_gt: dict[int, int] = {}
    identity_switches = 0
    for record in sorted(records, key=lambda item: int(item["frame_id"])):
        for gt_identity, track_identity in record["identities"]:  # type: ignore[assignment]
            if track_identity is None:
                continue
            previous = previous_track_by_gt.get(gt_identity)
            if previous is not None and previous != track_identity:
                identity_switches += 1
            previous_track_by_gt[gt_identity] = track_identity

    gap_bins: dict[str, list[int]] = {
        "no_gap": [0, 0],
        "gap_1_2": [0, 0],
        "gap_3_9": [0, 0],
        "gap_10_plus": [0, 0],
        "spatial_match_unassigned": [0, 0],
    }
    for record in records:
        for miss_streak, is_correct in record["gaps"]:  # type: ignore[assignment]
            if miss_streak is None:
                name = "spatial_match_unassigned"
            elif miss_streak == 0:
                name = "no_gap"
            elif miss_streak <= 2:
                name = "gap_1_2"
            elif miss_streak <= 9:
                name = "gap_3_9"
            else:
                name = "gap_10_plus"
            gap_bins[name][0] += int(is_correct)
            gap_bins[name][1] += 1

    return {
        "frames": len(records),
        "gt_instances": gt_total,
        "anonymous_observations": observation_total,
        "assigned_track_detections": assigned_total,
        "identity_blind_spatial_matches": spatial_matches,
        "assigned_identity_blind_spatial_matches": assigned_spatial_matches,
        "spatial_detection_recall": spatial_matches / gt_total if gt_total else float("nan"),
        "spatial_match_assignment_coverage": (
            assigned_spatial_matches / spatial_matches
            if spatial_matches
            else float("nan")
        ),
        "canonical_correct": canonical_correct,
        "canonical_assignment_accuracy_on_spatial_matches": (
            canonical_correct / spatial_matches if spatial_matches else float("nan")
        ),
        "canonical_accuracy_on_assigned_spatial_matches": (
            canonical_correct / assigned_spatial_matches
            if assigned_spatial_matches
            else float("nan")
        ),
        "canonical_precision_over_all_assigned_track_detections": (
            canonical_correct / assigned_total if assigned_total else float("nan")
        ),
        "canonical_idtp": canonical_correct,
        "canonical_idfp": canonical_idfp,
        "canonical_idfn": canonical_idfn,
        "canonical_idf1": (
            2 * canonical_correct / canonical_denominator
            if canonical_denominator
            else float("nan")
        ),
        "global_mapping_idtp": global_idtp,
        "global_mapping_idf1": (
            2 * global_idtp / global_denominator
            if global_denominator
            else float("nan")
        ),
        "identity_switches_between_spatially_matched_annotated_events": identity_switches,
        "gap_recovery": {
            name: {
                "correct": values[0],
                "events": values[1],
                "accuracy": values[0] / values[1] if values[1] else float("nan"),
            }
            for name, values in gap_bins.items()
        },
    }


def _compact_temporal_frame_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, float | int]]:
    """Expose auditable per-frame counts without internal tuple-key confusion maps."""

    output = []
    for record in records:
        gt_total = int(record["gt_total"])
        assigned_total = int(record["assigned_total"])
        spatial_matches = int(record["spatial_matches"])
        assigned_spatial = int(record["assigned_spatial_matches"])
        correct = int(record["canonical_correct"])
        denominator = gt_total + assigned_total
        output.append(
            {
                "frame_id": int(record["frame_id"]),
                "gt_instances": gt_total,
                "anonymous_observations": int(record["observation_total"]),
                "assigned_track_detections": assigned_total,
                "identity_blind_spatial_matches": spatial_matches,
                "assigned_identity_blind_spatial_matches": assigned_spatial,
                "canonical_correct": correct,
                "canonical_assignment_accuracy_on_spatial_matches": (
                    correct / spatial_matches if spatial_matches else float("nan")
                ),
                "canonical_idf1": (
                    2 * correct / denominator if denominator else float("nan")
                ),
            }
        )
    return output


def summarize_temporal_tracking(
    assignments_by_frame: Mapping[int, Sequence[TemporalAssignment]],
    spatial_matches_by_frame: Mapping[int, Sequence[SpatialMatch]],
    gt_objects_by_frame: Mapping[int, Mapping[int, MaskObject]],
    observations_by_frame: Mapping[int, Sequence[AnonymousObservation]],
    frame_ids: Sequence[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = _temporal_frame_records(
        assignments_by_frame,
        spatial_matches_by_frame,
        gt_objects_by_frame,
        observations_by_frame,
        frame_ids,
    )
    return _summarize_temporal_records(records), records


def _bootstrap_temporal_delta(
    centroid_records: Sequence[dict[str, object]],
    extent_records: Sequence[dict[str, object]],
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
    temporal_block_width: int | None = None,
) -> tuple[float, float]:
    if len(centroid_records) != len(extent_records) or not centroid_records:
        return (float("nan"), float("nan"))
    if temporal_block_width is None:
        centroid_units = [[record] for record in centroid_records]
        extent_units = [[record] for record in extent_records]
    else:
        centroid_blocks: dict[int, list[dict[str, object]]] = {}
        extent_blocks: dict[int, list[dict[str, object]]] = {}
        for record in centroid_records:
            block = int(record["frame_id"]) // temporal_block_width
            centroid_blocks.setdefault(block, []).append(record)
        for record in extent_records:
            block = int(record["frame_id"]) // temporal_block_width
            extent_blocks.setdefault(block, []).append(record)
        common_blocks = sorted(set(centroid_blocks).intersection(extent_blocks))
        centroid_units = [centroid_blocks[block] for block in common_blocks]
        extent_units = [extent_blocks[block] for block in common_blocks]
    if not centroid_units or n_bootstrap <= 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        selected = rng.integers(0, len(centroid_units), size=len(centroid_units))
        centroid_sample = [record for unit in selected for record in centroid_units[unit]]
        extent_sample = [record for unit in selected for record in extent_units[unit]]
        centroid_value = float(_summarize_temporal_records(centroid_sample)[metric])
        extent_value = float(_summarize_temporal_records(extent_sample)[metric])
        values[index] = extent_value - centroid_value
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def _robust_extent_scales(
    observations_by_frame: Mapping[int, Sequence[AnonymousObservation]],
) -> np.ndarray:
    features = np.asarray(
        [
            observation.log_extent
            for observations in observations_by_frame.values()
            for observation in observations
        ],
        dtype=float,
    )
    if len(features) < 2:
        return np.ones(4, dtype=float)
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    return np.maximum(1.4826 * mad, 0.1)


def _choose_anchor_frame(
    metadata: TargettrackMetadata,
    gt_objects_by_frame: Mapping[int, Mapping[int, MaskObject]],
) -> int:
    candidates = [
        frame_id
        for frame_id in metadata.training_frame_ids
        if frame_id in gt_objects_by_frame
    ]
    if not candidates:
        candidates = list(metadata.annotated_frame_ids[:1])
    if not candidates:
        raise ValueError("no annotated frame is available for temporal initialization")
    return max(candidates, key=lambda frame_id: (len(gt_objects_by_frame[frame_id]), frame_id))


def _paired_temporal_uncertainty(
    records_by_split: Mapping[str, Mapping[str, Sequence[dict[str, object]]]],
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    uncertainty: dict[str, object] = {}
    for split in ("validation", "unseen"):
        centroid_records = records_by_split[split]["centroid_only"]
        extent_records = records_by_split[split]["centroid_plus_dynamic_extent"]
        split_output = {}
        for metric in (
            "canonical_idf1",
            "canonical_assignment_accuracy_on_spatial_matches",
            "global_mapping_idf1",
        ):
            frame_ci = _bootstrap_temporal_delta(
                centroid_records,
                extent_records,
                metric=metric,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            block_ci = _bootstrap_temporal_delta(
                centroid_records,
                extent_records,
                metric=metric,
                n_bootstrap=n_bootstrap,
                seed=seed,
                temporal_block_width=100,
            )
            split_output[f"{metric}_extent_minus_centroid"] = {
                "point_estimate": float(
                    summaries[split]["centroid_plus_dynamic_extent"][metric]
                )
                - float(summaries[split]["centroid_only"][metric]),
                "frame_bootstrap_ci95": [frame_ci[0], frame_ci[1]],
                "100_frame_block_bootstrap_ci95": [block_ci[0], block_ci[1]],
                "bootstrap_replicates": n_bootstrap,
                "seed": seed,
            }
        uncertainty[split] = split_output
    return uncertainty


def evaluate_anonymous_temporal_tracking(
    observations_by_frame: Mapping[int, Sequence[AnonymousObservation]],
    spatial_matches_by_frame: Mapping[int, Sequence[SpatialMatch]],
    gt_objects_by_frame: Mapping[int, Mapping[int, MaskObject]],
    metadata: TargettrackMetadata,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Tune on recorded validation frames, then compare on unseen annotations."""

    anchor_frame = _choose_anchor_frame(metadata, gt_objects_by_frame)
    initial_objects = gt_objects_by_frame[anchor_frame]
    extent_scales = _robust_extent_scales(observations_by_frame)
    anchor_matches = spatial_matches_by_frame.get(anchor_frame, ())

    common_grid = [
        TemporalTrackerParameters(
            centroid_scale=centroid_scale,
            velocity_alpha=velocity_alpha,
            gap_scale_growth=gap_growth,
            extent_weight=0.0,
        )
        for centroid_scale in (2.0, 4.0, 8.0, 12.0, 16.0, 24.0)
        for velocity_alpha in (0.0, 0.3, 0.7)
        for gap_growth in (0.0, 0.1)
    ]
    tuning_rows = []
    best_parameters = common_grid[0]
    best_score = (-float("inf"), -float("inf"))
    for parameters in common_grid:
        assignments = run_anonymous_temporal_tracker(
            observations_by_frame,
            initial_objects,
            anchor_frame=anchor_frame,
            frame_ids=metadata.frame_ids,
            parameters=parameters,
            extent_scales=extent_scales,
            anchor_spatial_matches=anchor_matches,
        )
        summary, _ = summarize_temporal_tracking(
            assignments,
            spatial_matches_by_frame,
            gt_objects_by_frame,
            observations_by_frame,
            metadata.validation_frame_ids,
        )
        score = (
            float(summary["canonical_idf1"]),
            -float(summary["identity_switches_between_spatially_matched_annotated_events"]),
        )
        tuning_rows.append(
            {
                "parameters": asdict(parameters),
                "validation_canonical_idf1": summary["canonical_idf1"],
                "validation_identity_switches": summary[
                    "identity_switches_between_spatially_matched_annotated_events"
                ],
            }
        )
        if score > best_score:
            best_score = score
            best_parameters = parameters

    centroid_parameters = best_parameters
    extent_tuning_rows = []
    best_extent_parameters = TemporalTrackerParameters(
        **{
            **asdict(centroid_parameters),
            "extent_weight": 0.05,
            "extent_alpha": 0.5,
        }
    )
    best_extent_score = (-float("inf"), -float("inf"))
    for extent_weight in (0.1, 0.5, 2.0, 8.0, 32.0, 64.0, 128.0, 256.0, 512.0):
        for extent_alpha in (0.1, 0.25, 0.4, 0.5, 1.0):
            parameters = TemporalTrackerParameters(
                **{
                    **asdict(centroid_parameters),
                    "extent_weight": extent_weight,
                    "extent_alpha": extent_alpha,
                }
            )
            assignments = run_anonymous_temporal_tracker(
                observations_by_frame,
                initial_objects,
                anchor_frame=anchor_frame,
                frame_ids=metadata.frame_ids,
                parameters=parameters,
                extent_scales=extent_scales,
                anchor_spatial_matches=anchor_matches,
            )
            summary, _ = summarize_temporal_tracking(
                assignments,
                spatial_matches_by_frame,
                gt_objects_by_frame,
                observations_by_frame,
                metadata.validation_frame_ids,
            )
            score = (
                float(summary["canonical_idf1"]),
                -float(
                    summary[
                        "identity_switches_between_spatially_matched_annotated_events"
                    ]
                ),
            )
            extent_tuning_rows.append(
                {
                    "parameters": asdict(parameters),
                    "validation_canonical_idf1": summary["canonical_idf1"],
                    "validation_identity_switches": summary[
                        "identity_switches_between_spatially_matched_annotated_events"
                    ],
                }
            )
            if score > best_extent_score:
                best_extent_score = score
                best_extent_parameters = parameters

    tracker_assignments = {}
    for name, parameters in {
        "centroid_only": centroid_parameters,
        "centroid_plus_dynamic_extent": best_extent_parameters,
    }.items():
        tracker_assignments[name] = run_anonymous_temporal_tracker(
            observations_by_frame,
            initial_objects,
            anchor_frame=anchor_frame,
            frame_ids=metadata.frame_ids,
            parameters=parameters,
            extent_scales=extent_scales,
            anchor_spatial_matches=anchor_matches,
        )

    summaries: dict[str, dict[str, object]] = {"validation": {}, "unseen": {}}
    records_by_split: dict[str, dict[str, list[dict[str, object]]]] = {
        "validation": {},
        "unseen": {},
    }
    for split, frame_ids in {
        "validation": metadata.validation_frame_ids,
        "unseen": metadata.unseen_frame_ids,
    }.items():
        for name, assignments in tracker_assignments.items():
            summary, records = summarize_temporal_tracking(
                assignments,
                spatial_matches_by_frame,
                gt_objects_by_frame,
                observations_by_frame,
                frame_ids,
            )
            summaries[split][name] = summary
            records_by_split[split][name] = records

    uncertainty = _paired_temporal_uncertainty(
        records_by_split,
        summaries,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    forced_assignments = {}
    for name, parameters in {
        "centroid_only": centroid_parameters,
        "centroid_plus_dynamic_extent": best_extent_parameters,
    }.items():
        forced_parameters = TemporalTrackerParameters(
            **{**asdict(parameters), "unmatched_cost": 1e6}
        )
        forced_assignments[name] = run_anonymous_temporal_tracker(
            observations_by_frame,
            initial_objects,
            anchor_frame=anchor_frame,
            frame_ids=metadata.frame_ids,
            parameters=forced_parameters,
            extent_scales=extent_scales,
            anchor_spatial_matches=anchor_matches,
        )
    forced_summaries: dict[str, dict[str, object]] = {
        "validation": {},
        "unseen": {},
    }
    forced_records: dict[str, dict[str, list[dict[str, object]]]] = {
        "validation": {},
        "unseen": {},
    }
    for split, frame_ids in {
        "validation": metadata.validation_frame_ids,
        "unseen": metadata.unseen_frame_ids,
    }.items():
        for name, assignments in forced_assignments.items():
            summary, records = summarize_temporal_tracking(
                assignments,
                spatial_matches_by_frame,
                gt_objects_by_frame,
                observations_by_frame,
                frame_ids,
            )
            forced_summaries[split][name] = summary
            forced_records[split][name] = records
    forced_uncertainty = _paired_temporal_uncertainty(
        forced_records,
        forced_summaries,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return {
        "method": (
            "Supplied positive mask labels partition touching pixels only within each "
            "frame. Numerical labels are discarded, instances are spatially re-indexed, "
            "and two recursive Hungarian trackers consume identical centroid/extent "
            "observations bidirectionally from one declared training-frame anchor."
        ),
        "initialization": {
            "anchor_frame": anchor_frame,
            "anchor_is_recorded_training_frame": anchor_frame
            in metadata.training_frame_ids,
            "canonical_identities_initialized_from_gt": sorted(initial_objects),
            "gt_used_after_anchor_for_tracking": False,
        },
        "evaluation_matching": (
            "Per annotated frame, Hungarian matching maximizes mask IoU between anonymous "
            "prediction instances and all GT identities; only IoU > 1/3 is accepted. "
            "Prediction label equality is never used."
        ),
        "extent_feature": "log(voxel count + 0.5) and three log(bbox side + 0.5) values",
        "metric_definition": (
            "Canonical IDF1 is 2*correct/(assigned track detections + GT instances), so "
            "wrong assignments, unmatched assigned observations, and unassigned GT "
            "objects are penalized. Global-mapping IDF1 first optimally permutes track "
            "IDs. Identity switches are changes between successive spatially matched "
            "annotated events for a GT identity."
        ),
        "gap_definition": (
            "Gap bins use the assigned track's number of consecutive unassigned movie "
            "frames immediately before the evaluated assignment; spatially matched "
            "objects rejected as clutter are reported separately."
        ),
        "extent_scale": extent_scales.tolist(),
        "extent_scale_source": (
            "Robust MAD over all anonymous observations in the movie, without GT labels "
            "or supplied numerical identity values (offline/transductive scaling)."
        ),
        "parameter_selection": {
            "common_motion_and_gating": (
                "centroid-only grid selected by recorded-validation canonical IDF1; "
                "then frozen for both trackers"
            ),
            "extent_weight_and_update": (
                "positive extent weight/EMA grid selected by recorded-validation "
                "canonical IDF1"
            ),
            "centroid_grid": tuning_rows,
            "extent_grid": extent_tuning_rows,
            "selected_centroid_only": asdict(centroid_parameters),
            "selected_centroid_plus_dynamic_extent": asdict(best_extent_parameters),
        },
        "metrics_by_split": summaries,
        "per_frame_metrics": {
            split: {
                name: _compact_temporal_frame_records(records)
                for name, records in split_records.items()
            }
            for split, split_records in records_by_split.items()
        },
        "paired_uncertainty": uncertainty,
        "forced_assignment_sensitivity": {
            "method": (
                "Set the same effectively infinite unmatched cost for both selected "
                "trackers so every anonymous observation is assigned when tracks are "
                "available; no parameters are retuned."
            ),
            "metrics_by_split": forced_summaries,
            "per_frame_metrics": {
                split: {
                    name: _compact_temporal_frame_records(records)
                    for name, records in split_records.items()
                }
                for split, split_records in forced_records.items()
            },
            "paired_uncertainty": forced_uncertainty,
        },
        "caveats": [
            "Identity is supervised at one recorded training anchor; this is not fully "
            "unsupervised discovery.",
            "The tracker runs backward and forward from the anchor, so it is an offline "
            "bidirectional benchmark rather than a forward-only online system.",
            "Mask-channel labels are retained only as anonymous per-frame instance "
            "boundaries because binarization would merge touching neurons.",
            "Validation labels select parameters; unseen annotations are excluded from "
            "selection but are not guaranteed prospectively blind.",
            "All data come from one recording; frame/block bootstraps do not provide "
            "between-animal uncertainty.",
        ],
    }


def evaluate_targettrack(
    path: str | Path,
    *,
    network: str | None = None,
    voxel_spacing: Sequence[float] = DEFAULT_VOXEL_SPACING,
    n_bootstrap: int = 2000,
    seed: int = 0,
    include_object_rows: bool = False,
    include_association_ablation: bool = True,
    include_temporal_benchmark: bool = True,
) -> dict[str, object]:
    """Run the full real-data Targettrack mask and extent benchmark."""

    source_path = Path(path)
    all_rows: list[ObjectMetric] = []
    false_positives_by_split = {"train": 0, "validation": 0, "unseen": 0}
    gt_objects_by_frame: dict[int, dict[int, MaskObject]] = {}
    observations_by_frame: dict[int, tuple[AnonymousObservation, ...]] = {}
    spatial_matches_by_frame: dict[int, tuple[SpatialMatch, ...]] = {}
    stored_iou_differences: list[float] = []

    with TargettrackH5(source_path, network=network) as dataset:
        metadata = dataset.metadata
        annotated_with_predictions = sorted(
            set(metadata.annotated_frame_ids).intersection(metadata.prediction_frame_ids)
        )
        for frame_id in annotated_with_predictions:
            split = _split_name(frame_id, metadata)
            ground_truth = dataset.read_mask(frame_id)
            prediction = dataset.read_mask(frame_id, predicted=True)
            rows, false_positive_identities = compare_identity_masks(
                ground_truth,
                prediction,
                frame_id=frame_id,
                split=split,
                voxel_spacing=voxel_spacing,
            )
            all_rows.extend(rows)
            false_positives_by_split[split] += len(false_positive_identities)
            gt_objects_by_frame[frame_id] = extract_mask_objects(
                ground_truth, voxel_spacing=voxel_spacing
            )
            observations = anonymize_prediction_mask(
                prediction, voxel_spacing=voxel_spacing
            )
            observations_by_frame[frame_id] = observations
            spatial_matches_by_frame[frame_id] = match_anonymous_observations_to_gt(
                ground_truth,
                prediction,
                observations,
            )

            stored_key = f"net/{metadata.network}/pred_iou"
            if stored_key in dataset.h5:
                stored = np.asarray(dataset.h5[stored_key][frame_id], dtype=float)
                for row in rows:
                    if row.identity < len(stored) and np.isfinite(stored[row.identity]):
                        stored_iou_differences.append(abs(row.iou - stored[row.identity]))

        static_baseline = _static_extent_baseline(all_rows)
        summaries = {}
        extent_summaries = {}
        identity_summaries = {}
        for split in ("train", "validation", "unseen"):
            selected = [row for row in all_rows if row.split == split]
            summaries[split] = summarize_object_metrics(
                selected, false_positives_by_split[split]
            )
            extent_summaries[split] = summarize_extent_information(
                selected,
                static_baseline,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        for identity in sorted({row.identity for row in all_rows}):
            selected = [row for row in all_rows if row.identity == identity]
            identity_summaries[str(identity)] = summarize_object_metrics(
                selected, false_positive_instances=0
            )

        if include_temporal_benchmark:
            for frame_id in metadata.prediction_frame_ids:
                if frame_id in observations_by_frame:
                    continue
                prediction = dataset.read_mask(frame_id, predicted=True)
                observations_by_frame[frame_id] = anonymize_prediction_mask(
                    prediction, voxel_spacing=voxel_spacing
                )
            temporal_benchmark = evaluate_anonymous_temporal_tracking(
                observations_by_frame,
                spatial_matches_by_frame,
                gt_objects_by_frame,
                metadata,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

        result: dict[str, object] = {
            "benchmark": "Targettrack identity-labelled volumetric mask evaluation",
            "interpretation_boundary": (
                "Segmentation, centroid, and extent evaluation on annotated frames; the "
                "archive's identity-labelled masks do not by themselves establish temporal "
                "identity-switch or end-to-end tracking performance."
            ),
            "source": {
                "path": str(source_path.resolve()),
                "bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
            },
            "voxel_spacing_in_array_axis_order": list(map(float, voxel_spacing)),
            "voxel_spacing_unit": "voxel unless caller supplies physical spacing",
            "metadata": asdict(metadata),
            "split_definition": {
                "train": "recorded net/<network>.attrs['traininds']",
                "validation": "recorded net/<network>.attrs['Validinds']",
                "unseen": (
                    "annotated frames in neither recorded train nor validation; not claimed "
                    "to be a formally blinded test set"
                ),
            },
            "caveats": [
                "The public demo is one freely moving-worm recording, so no animal-level "
                "or biological-generalization uncertainty can be estimated.",
                "Validation frames were evaluated during network training; only the "
                "archive-recorded partitions are reconstructed here.",
                "The unseen partition is excluded from the recorded train/validation "
                "indices but is not claimed to have been prospectively blinded.",
                "Frame-bootstrap intervals quantify within-recording frame variation and "
                "do not account for temporal dependence or between-animal variation.",
                "Mask volume and shape variation can combine optics, pose, alignment, "
                "annotation, and biological morphology; it is not direct evidence of "
                "changing physical cell size.",
                "The extent-association ablation uses identity-stripped ground-truth masks, "
                "not the supplied network predictions, to isolate cue information.",
            ],
            "mask_metrics_by_split": summaries,
            "extent_information_by_split": extent_summaries,
            "mask_metrics_by_identity_all_splits": identity_summaries,
            "stored_pred_iou_reproduction": {
                "compared_instances": len(stored_iou_differences),
                "maximum_absolute_difference": max(stored_iou_differences)
                if stored_iou_differences
                else float("nan"),
            },
        }
        if include_association_ablation:
            result["extent_association_ablation"] = evaluate_extent_association(
                gt_objects_by_frame,
                metadata,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        if include_temporal_benchmark:
            result["anonymous_temporal_extent_benchmark"] = temporal_benchmark
        if include_object_rows:
            result["object_rows"] = [asdict(row) for row in all_rows]
        return result


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path, help="Targettrack HDF5 archive")
    parser.add_argument("--network", default=None, help="network group under /net")
    parser.add_argument(
        "--voxel-spacing",
        nargs=3,
        type=float,
        metavar=("AXIS0", "AXIS1", "AXIS2"),
        default=DEFAULT_VOXEL_SPACING,
        help="physical spacing in HDF5 array-axis order (default: one voxel each)",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-object-rows", action="store_true")
    parser.add_argument("--no-association-ablation", action="store_true")
    parser.add_argument("--no-temporal-benchmark", action="store_true")
    parser.add_argument("--output", type=Path, help="write JSON to this path")
    args = parser.parse_args(argv)

    result = evaluate_targettrack(
        args.h5,
        network=args.network,
        voxel_spacing=args.voxel_spacing,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        include_object_rows=args.include_object_rows,
        include_association_ablation=not args.no_association_ablation,
        include_temporal_benchmark=not args.no_temporal_benchmark,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=True, default=_json_default)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
