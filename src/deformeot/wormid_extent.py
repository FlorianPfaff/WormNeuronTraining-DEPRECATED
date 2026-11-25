"""Identity-blind local-shape measurements for WormID-Bench tracking.

The WormID EY recordings contain manually tracked point centres and the raw
calcium volume.  This module turns a small raw-image patch around each supplied
centre into an intensity-normalized extended-object descriptor.  Absolute
fluorescence and the stored neural traces are deliberately excluded: only the
*spatial distribution* of positive, locally background-subtracted signal is
used.

The accompanying tracker is intentionally simple.  It estimates one robust
affine deformation between consecutive point clouds, associates detections by
Hungarian assignment, and optionally adds a frozen-calibration descriptor cost.
Every frame is independently permuted before tracking so that NWB ROI row order
cannot leak identity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .tracker import AffineTransform, estimate_shared_affine
from .wormid import (
    IdentityBlindSequence,
    WormIDRecording,
    identity_blind_sequence,
    load_wormid_recording,
)
from .wormid_benchmark import (
    ReferenceTrackerConfig,
    TrackResult as ReferenceTrackResult,
    score_tracks,
    track_sequence,
)


_EPS = 1e-12


@dataclass(frozen=True)
class PatchFeatureConfig:
    """Physical patch and robust-normalization settings.

    Radii are specified in source ``(x, y, z)`` voxels because the raw NWB
    movie uses ``(time, x, y, z)`` order.  The descriptor itself uses physical
    micrometres so its geometry respects the anisotropic z spacing.
    """

    radius_xyz: tuple[int, int, int] = (5, 5, 2)
    background_quantile: float = 0.25
    upper_quantile: float = 0.995
    radial_edges_um: tuple[float, ...] = (0.45, 0.9, 1.5, 2.4)
    minimum_signal_mass: float = 1.0

    def __post_init__(self) -> None:
        if len(self.radius_xyz) != 3 or any(value < 1 for value in self.radius_xyz):
            raise ValueError("radius_xyz must contain three positive integers")
        if not 0 <= self.background_quantile < self.upper_quantile <= 1:
            raise ValueError("quantiles must satisfy 0 <= background < upper <= 1")
        if not self.radial_edges_um or any(
            right <= left
            for left, right in zip(self.radial_edges_um, self.radial_edges_um[1:])
        ):
            raise ValueError("radial_edges_um must be a nonempty increasing sequence")
        if self.minimum_signal_mass <= 0:
            raise ValueError("minimum_signal_mass must be positive")


@dataclass(frozen=True)
class PatchFeatureSet:
    """Compact patch descriptors aligned to stable NWB ROI rows."""

    descriptors: np.ndarray
    reliability: np.ndarray
    feature_names: tuple[str, ...]
    extent_indices: tuple[int, ...]
    source_frame_indices: tuple[int, ...]
    extraction_seconds: float

    def __post_init__(self) -> None:
        descriptors = np.asarray(self.descriptors, dtype=float)
        reliability = np.asarray(self.reliability, dtype=float)
        if descriptors.ndim != 3:
            raise ValueError("descriptors must have shape (frames, neurons, features)")
        if reliability.shape != descriptors.shape[:2]:
            raise ValueError("reliability must have shape (frames, neurons)")
        if len(self.feature_names) != descriptors.shape[-1]:
            raise ValueError("feature_names does not match descriptor width")
        if not np.all(np.isfinite(descriptors)) or not np.all(np.isfinite(reliability)):
            raise ValueError("features and reliabilities must be finite")
        if np.any((reliability < 0) | (reliability > 1)):
            raise ValueError("reliability must lie in [0, 1]")
        if len(self.source_frame_indices) != descriptors.shape[0]:
            raise ValueError("source_frame_indices must have one entry per frame")
        if any(index < 0 or index >= descriptors.shape[-1] for index in self.extent_indices):
            raise ValueError("extent index outside descriptor width")
        object.__setattr__(self, "descriptors", descriptors.copy())
        object.__setattr__(self, "reliability", reliability.copy())


@dataclass(frozen=True)
class RobustFeatureScaler:
    """Median/IQR normalization fitted only on the calibration recording."""

    center: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape[-1] != len(self.center):
            raise ValueError("feature width does not match scaler")
        return (values - self.center) / self.scale


@dataclass(frozen=True)
class AssociationConfig:
    """Frozen settings for point-plus-shape association."""

    appearance_weight: float = 0.0
    template_update: float = 0.0
    residual_velocity_gain: float = 0.0
    feature_indices: tuple[int, ...] | None = None
    position_power: float = 1.0
    use_reliability_gate: bool = True

    def __post_init__(self) -> None:
        if self.appearance_weight < 0:
            raise ValueError("appearance_weight cannot be negative")
        if not 0 <= self.template_update <= 1:
            raise ValueError("template_update must lie in [0, 1]")
        if not 0 <= self.residual_velocity_gain <= 1:
            raise ValueError("residual_velocity_gain must lie in [0, 1]")
        if self.position_power not in (1.0, 2.0):
            raise ValueError("position_power must be 1 or 2")


@dataclass(frozen=True)
class TrackingResult:
    """Assignments and hidden identities retained separately for evaluation."""

    assignments: np.ndarray
    assigned_truth: np.ndarray
    canonical_truth: np.ndarray
    frame_indices: tuple[int, ...]
    transition_affines: tuple[AffineTransform, ...]
    tracking_seconds: float


@dataclass(frozen=True)
class AnonymousPatchSequence:
    """Patch rows paired to anonymous detection rows before association."""

    descriptors: tuple[np.ndarray, ...]
    reliability: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class DualAnchorExtentConfig:
    """Extent settings layered onto the separately frozen dual-anchor geometry."""

    extent_weight: float = 0.0
    extent_update: float = 0.0
    feature_indices: tuple[int, ...] | None = None
    use_reliability_gate: bool = True

    def __post_init__(self) -> None:
        if self.extent_weight < 0:
            raise ValueError("extent_weight cannot be negative")
        if not 0 <= self.extent_update <= 1:
            raise ValueError("extent_update must lie in [0, 1]")


def _patch_offsets(radius_xyz: tuple[int, int, int]) -> np.ndarray:
    ranges = [np.arange(-radius, radius + 1) for radius in radius_xyz]
    mesh = np.meshgrid(*ranges, indexing="ij")
    return np.column_stack([axis.reshape(-1) for axis in mesh]).astype(int)


def _weighted_quantile_rows(
    values: np.ndarray,
    valid: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """Small row-wise masked quantile without masked-array side effects."""

    masked = np.where(valid, values, np.nan)
    with np.errstate(invalid="ignore"):
        result = np.nanquantile(masked, quantile, axis=1)
    return np.where(np.isfinite(result), result, 0.0)


def patch_shape_descriptors(
    volume_xyz: np.ndarray,
    centers_xyz_voxels: np.ndarray,
    spacing_xyz_um: Sequence[float],
    *,
    config: PatchFeatureConfig = PatchFeatureConfig(),
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[int, ...]]:
    """Measure intensity-normalized local spatial shape at supplied centres.

    This function never receives identity labels.  Additive local background
    and multiplicative intensity are removed before calculating a probability
    mass over patch voxels.  Descriptor components are spatial moments and mass
    histograms; total fluorescence is returned only through a bounded reliability
    gate and is not a matching feature.
    """

    volume = np.asarray(volume_xyz, dtype=float)
    centers = np.asarray(centers_xyz_voxels, dtype=float)
    spacing = np.asarray(spacing_xyz_um, dtype=float)
    if volume.ndim != 3:
        raise ValueError("volume_xyz must be three-dimensional")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_xyz_voxels must have shape (neurons, 3)")
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz_um must contain three positive values")
    if np.any(~np.isfinite(centers)):
        raise ValueError("centres must be finite")

    offsets = _patch_offsets(config.radius_xyz)
    rounded = np.rint(centers).astype(int)
    sample_indices = rounded[:, None, :] + offsets[None, :, :]
    shape = np.asarray(volume.shape, dtype=int)
    valid = np.all((sample_indices >= 0) & (sample_indices < shape), axis=2)
    clipped = np.clip(sample_indices, 0, shape - 1)
    intensities = volume[
        clipped[..., 0],
        clipped[..., 1],
        clipped[..., 2],
    ]

    radii = np.asarray(config.radius_xyz, dtype=float)
    normalized_edge = np.max(np.abs(offsets) / radii, axis=1)
    outer = normalized_edge >= 0.8
    background = _weighted_quantile_rows(
        intensities,
        valid & outer[None, :],
        config.background_quantile,
    )
    upper = _weighted_quantile_rows(
        intensities,
        valid,
        config.upper_quantile,
    )
    cap = np.maximum(upper, background + _EPS)
    signal = np.clip(intensities - background[:, None], 0.0, cap[:, None] - background[:, None])
    signal *= valid
    mass = np.sum(signal, axis=1)
    safe_mass = np.maximum(mass, _EPS)
    weights = signal / safe_mass[:, None]

    # Coordinates are relative to the supplied (possibly fractional) centre,
    # not merely to the rounded patch origin.
    coordinates_um = (
        sample_indices.astype(float) - centers[:, None, :]
    ) * spacing[None, None, :]
    centroid = np.einsum("nk,nkd->nd", weights, coordinates_um)
    centered = coordinates_um - centroid[:, None, :]
    covariance = np.einsum("nk,nki,nkj->nij", weights, centered, centered)
    covariance += 1e-6 * np.eye(3)[None, :, :]
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    log_std = np.log(np.sqrt(eigenvalues))
    eigen_fraction = eigenvalues / np.sum(eigenvalues, axis=1, keepdims=True)

    covariance_unique = covariance[:, (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
    radius_um = np.linalg.norm(coordinates_um, axis=2)
    radial_edges = (-np.inf, *config.radial_edges_um, np.inf)
    radial_mass = np.column_stack(
        [
            np.sum(
                weights * ((radius_um > left) & (radius_um <= right)),
                axis=1,
            )
            for left, right in zip(radial_edges, radial_edges[1:])
        ]
    )

    z_offsets = np.arange(-config.radius_xyz[2], config.radius_xyz[2] + 1)
    z_mass = np.column_stack(
        [np.sum(weights * (offsets[None, :, 2] == offset), axis=1) for offset in z_offsets]
    )
    entropy = -np.sum(np.where(weights > 0, weights * np.log(weights + _EPS), 0.0), axis=1)
    entropy /= np.log(weights.shape[1])
    concentration = np.max(weights, axis=1)
    peak_index = np.argmax(signal, axis=1)
    peak_offset = coordinates_um[np.arange(len(centers)), peak_index]

    descriptors = np.column_stack(
        [
            centroid,
            covariance_unique,
            log_std,
            eigen_fraction,
            radial_mass,
            z_mass,
            entropy,
            concentration,
            peak_offset,
        ]
    )
    descriptors[mass < config.minimum_signal_mass] = 0.0

    # Reliability is a signal-to-background gate, not an identity descriptor.
    # The saturation at one prevents high-fluorescence neurons receiving larger
    # matching influence than ordinary well-resolved neurons.
    patch_size = np.maximum(np.sum(valid, axis=1), 1)
    mean_signal = mass / patch_size
    reliability = mean_signal / (mean_signal + background + 1.0)
    reliability = np.where(mass >= config.minimum_signal_mass, reliability, 0.0)
    reliability = np.clip(reliability, 0.0, 1.0)

    names = (
        "centroid_dx_um",
        "centroid_dy_um",
        "centroid_dz_um",
        "cov_xx_um2",
        "cov_yy_um2",
        "cov_zz_um2",
        "cov_xy_um2",
        "cov_xz_um2",
        "cov_yz_um2",
        "log_std_small",
        "log_std_middle",
        "log_std_large",
        "eig_fraction_small",
        "eig_fraction_middle",
        "eig_fraction_large",
        *(f"radial_mass_{index}" for index in range(radial_mass.shape[1])),
        *(f"z_mass_{int(offset):+d}" for offset in z_offsets),
        "spatial_entropy",
        "peak_concentration",
        "peak_dx_um",
        "peak_dy_um",
        "peak_dz_um",
    )
    # Centroid, covariance, log axes, and normalized eigenvalues are the strict
    # second-moment extended-object view.  Histograms/peak add local appearance.
    extent_indices = tuple(range(15))
    return descriptors, reliability, names, extent_indices


def extract_wormid_patch_features(
    nwb_path: str | Path,
    *,
    recording: WormIDRecording | None = None,
    frame_indices: Iterable[int] | None = None,
    config: PatchFeatureConfig = PatchFeatureConfig(),
) -> PatchFeatureSet:
    """Stream raw volumes and extract descriptors without reading trace arrays."""

    source = Path(nwb_path)
    if recording is None:
        recording = load_wormid_recording(source)
    if Path(recording.path).resolve() != source.resolve():
        raise ValueError("recording and nwb_path refer to different files")
    if frame_indices is None:
        selected = tuple(range(recording.frames))
    else:
        selected = tuple(int(value) for value in frame_indices)
    if not selected or any(value < 0 or value >= recording.frames for value in selected):
        raise ValueError("frame_indices must be nonempty and inside the trajectory")
    if any(right <= left for left, right in zip(selected, selected[1:])):
        raise ValueError("frame_indices must be strictly increasing")

    descriptors: list[np.ndarray] = []
    reliabilities: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    extent_indices: tuple[int, ...] | None = None
    started = perf_counter()
    with h5py.File(source, "r") as nwb:
        data_path = "acquisition/CalciumImageSeries/data"
        if data_path not in nwb:
            raise ValueError(f"{source} is missing {data_path}")
        movie = nwb[data_path]
        if movie.ndim != 4 or tuple(movie.shape[1:]) != (256, 128, 21):
            # WormID-Bench currently uses this layout.  Requiring the dimensions
            # avoids silently transposing an unrelated NWB volume convention.
            raise ValueError("expected WormID movie layout (time, 256, 128, 21)")
        if movie.shape[0] < recording.frames:
            raise ValueError("movie is shorter than the tracked trajectory")
        spacing_xyz = np.asarray(recording.spacing_zyx_um[::-1], dtype=float)
        for frame_index in selected:
            measured, quality, current_names, current_extent = patch_shape_descriptors(
                movie[frame_index],
                recording.source_xyz_voxels[frame_index],
                spacing_xyz,
                config=config,
            )
            if names is None:
                names = current_names
                extent_indices = current_extent
            elif names != current_names or extent_indices != current_extent:
                raise AssertionError("descriptor schema changed during extraction")
            descriptors.append(measured)
            reliabilities.append(quality)

    assert names is not None and extent_indices is not None
    return PatchFeatureSet(
        descriptors=np.stack(descriptors),
        reliability=np.stack(reliabilities),
        feature_names=names,
        extent_indices=extent_indices,
        source_frame_indices=selected,
        extraction_seconds=perf_counter() - started,
    )


def fit_robust_feature_scaler(
    calibration_features: np.ndarray,
    *,
    reliability: np.ndarray | None = None,
    minimum_scale: float = 0.02,
) -> RobustFeatureScaler:
    """Fit normalization on calibration data only."""

    features = np.asarray(calibration_features, dtype=float)
    if features.ndim < 2:
        raise ValueError("calibration_features must include samples and features")
    flat = features.reshape(-1, features.shape[-1])
    if reliability is not None:
        quality = np.asarray(reliability, dtype=float).reshape(-1)
        if len(quality) != len(flat):
            raise ValueError("reliability shape does not match features")
        flat = flat[quality > 0]
    if len(flat) == 0:
        raise ValueError("no reliable calibration features")
    center = np.median(flat, axis=0)
    q25, q75 = np.quantile(flat, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, minimum_scale)
    return RobustFeatureScaler(center=center, scale=scale)


def pair_patch_features_to_anonymous_detections(
    sequence: IdentityBlindSequence,
    features: PatchFeatureSet,
) -> AnonymousPatchSequence:
    """Apply the detection simulator's random row permutation to patch rows.

    ``truth_indices`` is used here only as the simulator's source-row permutation
    to keep each raw-image measurement paired with its coordinate.  The returned
    object contains no native indices, labels, or truth and is the only feature
    object passed to the association algorithm.
    """

    if sequence.source_frame_indices != features.source_frame_indices:
        raise ValueError("feature frames do not match anonymous sequence frames")
    if len(sequence.truth_indices) != len(features.descriptors):
        raise ValueError("feature and sequence lengths differ")
    descriptors: list[np.ndarray] = []
    reliability: list[np.ndarray] = []
    for frame, order, measured, quality in zip(
        sequence.frames,
        sequence.truth_indices,
        features.descriptors,
        features.reliability,
        strict=True,
    ):
        order = np.asarray(order, dtype=int)
        if len(order) != len(frame.positions) or np.any(np.sort(order) != np.arange(len(order))):
            raise ValueError("simulator row mapping must be a complete permutation")
        descriptors.append(np.asarray(measured, dtype=float)[order].copy())
        reliability.append(np.asarray(quality, dtype=float)[order].copy())
    return AnonymousPatchSequence(tuple(descriptors), tuple(reliability))


def _reference_spatial_cost(
    predicted: np.ndarray,
    target: np.ndarray,
    floor_um: float,
) -> np.ndarray:
    distances = cdist(predicted, target)
    rows, columns = linear_sum_assignment(distances)
    scale = max(float(np.median(distances[rows, columns])), floor_um)
    return distances / scale


def _reference_affine(
    source: np.ndarray,
    target: np.ndarray,
    config: ReferenceTrackerConfig,
) -> AffineTransform:
    return estimate_shared_affine(
        source,
        target,
        iterations=config.affine_iterations,
        ridge=config.affine_ridge,
        huber_delta=config.affine_huber_delta,
        min_scale=config.affine_min_scale,
        max_scale=config.affine_max_scale,
    )


def track_dual_anchor_extent(
    sequence: IdentityBlindSequence,
    anonymous_features: AnonymousPatchSequence,
    scaler: RobustFeatureScaler,
    *,
    geometry_config: ReferenceTrackerConfig = ReferenceTrackerConfig(),
    extent_config: DualAnchorExtentConfig = DualAnchorExtentConfig(),
) -> ReferenceTrackResult:
    """Add calibrated dynamic extent to the frozen dual-anchor point tracker.

    With ``extent_weight=0`` this follows the geometry and numerical operations
    of :func:`deformeot.wormid_benchmark.track_sequence`'s ``dual_anchor`` arm.
    Labels, native ROI indices, and stored neural traces are absent from every
    argument accepted here.
    """

    if not sequence.frames:
        raise ValueError("sequence is empty")
    if len(anonymous_features.descriptors) != len(sequence.frames) or len(
        anonymous_features.reliability
    ) != len(sequence.frames):
        raise ValueError("anonymous features must contain one array per frame")
    reference = np.asarray(sequence.frames[0].positions, dtype=float)
    count = len(reference)
    if extent_config.feature_indices is None:
        indices = np.arange(len(scaler.center), dtype=int)
    else:
        indices = np.asarray(extent_config.feature_indices, dtype=int)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("feature_indices must select at least one feature")

    scaled = tuple(
        scaler.transform(np.asarray(values, dtype=float))[:, indices]
        for values in anonymous_features.descriptors
    )
    qualities = tuple(np.asarray(values, dtype=float) for values in anonymous_features.reliability)
    for frame, values, quality in zip(sequence.frames, scaled, qualities, strict=True):
        if values.shape[0] != len(frame.positions) or quality.shape != (len(frame.positions),):
            raise ValueError("anonymous feature rows must match detection rows")

    started = perf_counter()
    assignments: list[np.ndarray] = [np.arange(count, dtype=np.int64)]
    previous = reference.copy()
    template = scaled[0].copy()
    template_quality = qualities[0].copy()
    cloud_motion: list[float] = [0.0]
    for frame_index, frame in enumerate(sequence.frames[1:], start=1):
        current = np.asarray(frame.positions, dtype=float)
        forward = np.min(cdist(previous, current), axis=1)
        backward = np.min(cdist(current, previous), axis=1)
        cloud_motion.append(float(0.5 * (np.median(forward) + np.median(backward))))

        temporal_prediction = _reference_affine(previous, current, geometry_config).apply(
            previous
        )
        temporal = _reference_spatial_cost(
            temporal_prediction,
            current,
            geometry_config.spatial_scale_floor_um,
        )
        reference_prediction = _reference_affine(reference, current, geometry_config).apply(
            reference
        )
        anchored = _reference_spatial_cost(
            reference_prediction,
            current,
            geometry_config.spatial_scale_floor_um,
        )
        cost = temporal + geometry_config.reference_weight * anchored
        if extent_config.extent_weight:
            extent = cdist(template, scaled[frame_index]) / np.sqrt(len(indices))
            if extent_config.use_reliability_gate:
                pair_quality = np.sqrt(
                    template_quality[:, None] * qualities[frame_index][None, :]
                )
            else:
                pair_quality = 1.0
            cost += extent_config.extent_weight * pair_quality * extent

        track_rows, detection_rows = linear_sum_assignment(cost)
        assignment = np.full(count, -1, dtype=np.int64)
        assignment[track_rows] = detection_rows
        assignments.append(assignment)
        previous = current[assignment]
        update = extent_config.extent_update
        template = (1 - update) * template + update * scaled[frame_index][assignment]
        template_quality = (
            (1 - update) * template_quality + update * qualities[frame_index][assignment]
        )

    return ReferenceTrackResult(
        method="dual_anchor_extent" if extent_config.extent_weight else "dual_anchor",
        assignments=tuple(assignments),
        runtime_seconds=float(perf_counter() - started),
        cloud_motion_um=np.asarray(cloud_motion, dtype=float),
    )


def precompute_affines(
    positions_zyx_um: np.ndarray,
    frame_indices: Sequence[int],
    *,
    seed: int = 20260720,
) -> tuple[AffineTransform, ...]:
    """Estimate identity-blind point-cloud transport for selected transitions."""

    positions = np.asarray(positions_zyx_um, dtype=float)
    selected = tuple(int(value) for value in frame_indices)
    rng = np.random.default_rng(seed)
    transforms: list[AffineTransform] = []
    for left, right in zip(selected, selected[1:]):
        source_order = rng.permutation(positions.shape[1])
        target_order = rng.permutation(positions.shape[1])
        transforms.append(
            estimate_shared_affine(
                positions[left, source_order],
                positions[right, target_order],
                iterations=6,
                ridge=0.2,
                huber_delta=2.5,
                min_scale=0.55,
                max_scale=1.8,
            )
        )
    return tuple(transforms)


def _median_nearest_neighbor(points: np.ndarray) -> float:
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    return max(float(np.median(np.min(distances, axis=1))), 1e-3)


def track_permuted_recording(
    recording: WormIDRecording,
    features: PatchFeatureSet,
    scaler: RobustFeatureScaler,
    *,
    frame_indices: Sequence[int],
    config: AssociationConfig,
    affines: Sequence[AffineTransform] | None = None,
    seed: int = 20260720,
) -> TrackingResult:
    """Track a complete cohort after independently permuting every frame.

    Labels and trace arrays are not accepted by this function.  The only hidden
    truth is the random row permutation, retained in the return value so callers
    can score after tracking.
    """

    selected = tuple(int(value) for value in frame_indices)
    if selected != features.source_frame_indices:
        raise ValueError("feature frames do not match requested tracking frames")
    if len(selected) < 2:
        raise ValueError("tracking requires at least two frames")
    if affines is None:
        affines = precompute_affines(recording.positions_zyx_um, selected, seed=seed + 1)
    affines = tuple(affines)
    if len(affines) != len(selected) - 1:
        raise ValueError("affines must contain one transform per transition")

    if config.feature_indices is None:
        feature_indices = np.arange(features.descriptors.shape[-1], dtype=int)
    else:
        feature_indices = np.asarray(config.feature_indices, dtype=int)
    if feature_indices.ndim != 1 or len(feature_indices) == 0:
        raise ValueError("feature_indices must be a nonempty one-dimensional selection")

    scaled = scaler.transform(features.descriptors)[..., feature_indices]
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(recording.neurons) for _ in selected])
    canonical_truth = permutations[0].copy()
    track_positions = recording.positions_zyx_um[selected[0], canonical_truth].copy()
    template = scaled[0, canonical_truth].copy()
    template_quality = features.reliability[0, canonical_truth].copy()
    residual_velocity = np.zeros_like(track_positions)

    all_assignments = [np.arange(recording.neurons, dtype=int)]
    assigned_truth = [canonical_truth.copy()]
    started = perf_counter()
    for local_index, (frame_index, affine) in enumerate(
        zip(selected[1:], affines, strict=True),
        start=1,
    ):
        order = permutations[local_index]
        detections = recording.positions_zyx_um[frame_index, order]
        predicted = affine.apply(track_positions)
        predicted += config.residual_velocity_gain * residual_velocity
        spatial_scale = _median_nearest_neighbor(detections)
        position_cost = cdist(predicted, detections) / spatial_scale
        if config.position_power == 2.0:
            position_cost **= 2

        if config.appearance_weight > 0:
            detection_features = scaled[local_index, order]
            feature_cost = cdist(template, detection_features) / np.sqrt(len(feature_indices))
            if config.use_reliability_gate:
                pair_quality = np.sqrt(
                    template_quality[:, None]
                    * features.reliability[local_index, order][None, :]
                )
            else:
                pair_quality = 1.0
            # Unreliable patches contribute no appearance evidence.
            cost = position_cost + config.appearance_weight * pair_quality * feature_cost
        else:
            detection_features = scaled[local_index, order]
            cost = position_cost

        track_rows, detection_rows = linear_sum_assignment(cost)
        assignment = np.empty(recording.neurons, dtype=int)
        assignment[track_rows] = detection_rows
        selected_positions = detections[assignment]
        residual_velocity = selected_positions - affine.apply(track_positions)
        track_positions = selected_positions

        update = config.template_update
        selected_features = detection_features[assignment]
        selected_quality = features.reliability[local_index, order][assignment]
        template = (1 - update) * template + update * selected_features
        template_quality = (1 - update) * template_quality + update * selected_quality
        all_assignments.append(assignment)
        assigned_truth.append(order[assignment])

    return TrackingResult(
        assignments=np.stack(all_assignments),
        assigned_truth=np.stack(assigned_truth),
        canonical_truth=canonical_truth,
        frame_indices=selected,
        transition_affines=affines,
        tracking_seconds=perf_counter() - started,
    )


def score_tracking(
    result: TrackingResult,
    recording: WormIDRecording,
    *,
    dense_threshold_um: float | None = None,
    severe_threshold_um: float | None = None,
) -> dict[str, float | int]:
    """Score canonical identity and optional post-hoc difficulty strata."""

    correct = result.assigned_truth == result.canonical_truth[None, :]
    previous = result.assigned_truth[:-1]
    current = result.assigned_truth[1:]
    switches = int(np.sum(current != previous))
    scores: dict[str, float | int] = {
        "correct": int(np.sum(correct)),
        "total": int(correct.size),
        "identity_accuracy": float(np.mean(correct)),
        "identity_switches": switches,
        "tracked_correct": int(np.sum(correct[1:])),
        "tracked_total": int(correct[1:].size),
        "tracked_identity_accuracy": float(np.mean(correct[1:])),
    }

    if dense_threshold_um is not None:
        dense_masks = []
        for frame_index in result.frame_indices:
            points = recording.positions_zyx_um[frame_index]
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            native_dense = np.min(distances, axis=1) <= dense_threshold_um
            dense_masks.append(native_dense[result.canonical_truth])
        dense = np.stack(dense_masks)
        dense[0] = False
        scores.update(
            dense_correct=int(np.sum(correct & dense)),
            dense_total=int(np.sum(dense)),
            dense_accuracy=float(np.mean(correct[dense])) if np.any(dense) else float("nan"),
        )

    if severe_threshold_um is not None:
        severe = np.zeros_like(correct)
        for local_index, (left, right) in enumerate(
            zip(result.frame_indices, result.frame_indices[1:]),
            start=1,
        ):
            displacement = np.linalg.norm(
                recording.positions_zyx_um[right] - recording.positions_zyx_um[left],
                axis=1,
            )
            severe[local_index] = displacement[result.canonical_truth] >= severe_threshold_um
        scores.update(
            severe_correct=int(np.sum(correct & severe)),
            severe_total=int(np.sum(severe)),
            severe_accuracy=(
                float(np.mean(correct[severe])) if np.any(severe) else float("nan")
            ),
        )
    return scores


def calibration_thresholds(
    recording: WormIDRecording,
    *,
    strides: Sequence[int],
) -> tuple[float, dict[int, float]]:
    """Derive evaluation-stratum cutoffs from the calibration animal only."""

    nearest = []
    for points in recording.positions_zyx_um:
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        nearest.extend(np.min(distances, axis=1))
    dense_threshold = float(np.quantile(nearest, 0.25))
    severe: dict[int, float] = {}
    for stride in strides:
        displacement = np.linalg.norm(
            recording.positions_zyx_um[stride:] - recording.positions_zyx_um[:-stride],
            axis=2,
        )
        severe[int(stride)] = float(np.quantile(displacement, 0.75))
    return dense_threshold, severe


def save_patch_feature_cache(features: PatchFeatureSet, path: str | Path) -> Path:
    """Atomically save a compact, schema-carrying patch-feature cache."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        descriptors=features.descriptors.astype(np.float32),
        reliability=features.reliability.astype(np.float32),
        feature_names=np.asarray(features.feature_names),
        extent_indices=np.asarray(features.extent_indices, dtype=np.int64),
        source_frame_indices=np.asarray(features.source_frame_indices, dtype=np.int64),
        extraction_seconds=float(features.extraction_seconds),
    )
    temporary.replace(destination)
    return destination


def load_patch_feature_cache(path: str | Path) -> PatchFeatureSet:
    """Load a cache created by :func:`save_patch_feature_cache`."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as cache:
        required = {
            "descriptors",
            "reliability",
            "feature_names",
            "extent_indices",
            "source_frame_indices",
            "extraction_seconds",
        }
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(f"feature cache is missing {sorted(missing)}")
        return PatchFeatureSet(
            descriptors=cache["descriptors"],
            reliability=cache["reliability"],
            feature_names=tuple(str(value) for value in cache["feature_names"].tolist()),
            extent_indices=tuple(int(value) for value in cache["extent_indices"]),
            source_frame_indices=tuple(int(value) for value in cache["source_frame_indices"]),
            extraction_seconds=float(cache["extraction_seconds"]),
        )


def _benchmark_frame_indices(frame_count: int, stride: int) -> tuple[int, ...]:
    if stride < 1:
        raise ValueError("stride must be positive")
    selected = tuple(range(0, frame_count, stride))
    return selected if selected[-1] == frame_count - 1 else (*selected, frame_count - 1)


def _subset_patch_features(
    features: PatchFeatureSet,
    frame_indices: Sequence[int],
) -> PatchFeatureSet:
    selected = tuple(int(value) for value in frame_indices)
    lookup = {frame: row for row, frame in enumerate(features.source_frame_indices)}
    try:
        rows = [lookup[frame] for frame in selected]
    except KeyError as error:
        raise ValueError(f"feature cache lacks source frame {error.args[0]}") from error
    return PatchFeatureSet(
        descriptors=features.descriptors[rows],
        reliability=features.reliability[rows],
        feature_names=features.feature_names,
        extent_indices=features.extent_indices,
        source_frame_indices=selected,
        extraction_seconds=0.0,
    )


def _reference_metrics_with_strata(
    sequence: IdentityBlindSequence,
    result: ReferenceTrackResult,
    recording: WormIDRecording,
    *,
    dense_threshold_um: float,
    severe_threshold_um: float,
) -> dict[str, float | int | None]:
    """Evaluate association after tracking; no truth enters a tracking cost."""

    canonical = np.asarray(sequence.truth_indices[0], dtype=int)
    identities: list[np.ndarray] = []
    correct: list[np.ndarray] = []
    dense: list[np.ndarray] = []
    for hidden_order, assignment, frame_index in zip(
        sequence.truth_indices,
        result.assignments,
        sequence.source_frame_indices,
        strict=True,
    ):
        current_identities = np.asarray(hidden_order, dtype=int)[
            np.asarray(assignment, dtype=int)
        ]
        identities.append(current_identities)
        correct.append(current_identities == canonical)
        points = recording.positions_zyx_um[frame_index]
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        native_dense = np.min(distances, axis=1) <= dense_threshold_um
        dense.append(native_dense[canonical])

    identity_matrix = np.stack(identities)
    correct_matrix = np.stack(correct)
    dense_matrix = np.stack(dense)
    dense_matrix[0] = False
    tracked = correct_matrix[1:]
    reference_metrics = score_tracks(
        sequence,
        result,
        severe_threshold_um=severe_threshold_um,
    )
    dense_total = int(np.sum(dense_matrix))
    return {
        "correct": reference_metrics.correct,
        "total": reference_metrics.total,
        "seeded_identity_accuracy": reference_metrics.identity_accuracy,
        "tracked_correct": int(np.sum(tracked)),
        "tracked_total": int(tracked.size),
        "tracked_identity_accuracy": float(np.mean(tracked)),
        "identity_switches": int(np.count_nonzero(np.diff(identity_matrix, axis=0))),
        "switch_opportunities": int(tracked.size),
        "identity_switch_rate": float(
            np.count_nonzero(np.diff(identity_matrix, axis=0)) / tracked.size
        ),
        "dense_correct": int(np.sum(correct_matrix & dense_matrix)),
        "dense_total": dense_total,
        "dense_identity_accuracy": (
            float(np.mean(correct_matrix[dense_matrix])) if dense_total else None
        ),
        "dense_prevalence": float(dense_total / tracked.size),
        "severe_correct": reference_metrics.severe_correct,
        "severe_total": reference_metrics.severe_total,
        "severe_identity_accuracy": (
            reference_metrics.severe_identity_accuracy
            if reference_metrics.severe_total
            else None
        ),
        "severe_prevalence": float(reference_metrics.severe_total / tracked.size),
        "severe_frames": reference_metrics.severe_frames,
        "severe_transition_threshold_um": severe_threshold_um,
        "runtime_seconds": reference_metrics.runtime_seconds,
    }


def _paired_recording_bootstrap(
    differences: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("bootstrap requires finite paired recording differences")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "recordings": int(len(values)),
        "mean_difference": float(np.mean(values)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "samples": int(samples),
        "per_recording_differences": values.tolist(),
    }


def _aggregate_reference_comparison(
    rows: Sequence[dict[str, object]],
    *,
    recording_ids: Sequence[str],
    strides: Sequence[int],
    baseline: str,
    improved: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    methods = (baseline, improved)

    def aggregate(selected: Sequence[dict[str, object]]) -> dict[str, float | int]:
        tracked_total = sum(int(row["tracked_total"]) for row in selected)
        dense_total = sum(int(row["dense_total"]) for row in selected)
        severe_total = sum(int(row["severe_total"]) for row in selected)
        switches = sum(int(row["identity_switches"]) for row in selected)
        opportunities = sum(int(row["switch_opportunities"]) for row in selected)
        return {
            "tracked_identity_accuracy": sum(
                int(row["tracked_correct"]) for row in selected
            )
            / tracked_total,
            "dense_identity_accuracy": sum(int(row["dense_correct"]) for row in selected)
            / dense_total,
            "dense_prevalence": dense_total / tracked_total,
            "severe_identity_accuracy": sum(
                int(row["severe_correct"]) for row in selected
            )
            / severe_total,
            "severe_prevalence": severe_total / tracked_total,
            "identity_switch_rate": switches / opportunities,
            "identity_switches": switches,
            "switch_opportunities": opportunities,
        }

    recording_summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for recording_id in recording_ids:
        recording_summaries[recording_id] = {}
        for method in methods:
            selected = [
                row
                for row in rows
                if row["recording_id"] == recording_id and row["method"] == method
            ]
            if len(selected) != len(strides):
                raise ValueError("comparison lacks a complete recording-method block")
            recording_summaries[recording_id][method] = aggregate(selected)

    metric_differences: dict[str, list[float]] = {
        "tracked_identity_accuracy_difference": [],
        "dense_identity_accuracy_difference": [],
        "severe_identity_accuracy_difference": [],
        "identity_switch_rate_reduction": [],
    }
    for summary in recording_summaries.values():
        base = summary[baseline]
        better = summary[improved]
        metric_differences["tracked_identity_accuracy_difference"].append(
            float(better["tracked_identity_accuracy"])
            - float(base["tracked_identity_accuracy"])
        )
        metric_differences["dense_identity_accuracy_difference"].append(
            float(better["dense_identity_accuracy"])
            - float(base["dense_identity_accuracy"])
        )
        metric_differences["severe_identity_accuracy_difference"].append(
            float(better["severe_identity_accuracy"])
            - float(base["severe_identity_accuracy"])
        )
        metric_differences["identity_switch_rate_reduction"].append(
            float(base["identity_switch_rate"]) - float(better["identity_switch_rate"])
        )

    micro_by_stride: dict[str, dict[str, dict[str, float | int]]] = {}
    for stride in strides:
        micro_by_stride[str(stride)] = {}
        for method in methods:
            selected = [
                row for row in rows if row["stride"] == stride and row["method"] == method
            ]
            micro_by_stride[str(stride)][method] = aggregate(selected)

    return {
        "recording_summaries": recording_summaries,
        "micro_by_stride": micro_by_stride,
        "complete_recording_paired_bootstrap": {
            name: _paired_recording_bootstrap(
                values,
                samples=bootstrap_samples,
                seed=seed,
            )
            for name, values in metric_differences.items()
        },
    }


def _load_or_extract_benchmark_features(
    recording: WormIDRecording,
    *,
    cache_dir: Path,
    patch_config: PatchFeatureConfig,
    regenerate: bool,
) -> tuple[PatchFeatureSet, Path]:
    cache_path = cache_dir / f"{recording.recording_id}.npz"
    if regenerate or not cache_path.is_file():
        measured = extract_wormid_patch_features(
            recording.path,
            recording=recording,
            config=patch_config,
        )
        save_patch_feature_cache(measured, cache_path)
    features = load_patch_feature_cache(cache_path)
    if features.descriptors.shape[:2] != (recording.frames, recording.neurons):
        raise ValueError(f"{cache_path} dimensions do not match {recording.recording_id}")
    return features, cache_path


def run_wormid_extent_benchmark(
    calibration_path: str | Path,
    held_out_paths: Sequence[str | Path],
    *,
    feature_cache_dir: str | Path,
    output_path: str | Path,
    strides: Sequence[int] = (1, 5, 10, 20, 40),
    calibration_gap_strides: Sequence[int] = (5, 10, 20, 40),
    extent_weights: Sequence[float] = (0.25, 0.5, 1, 2, 4, 8, 16, 32),
    extent_updates: Sequence[float] = (0, 0.05, 0.1, 0.15, 0.25, 0.5, 1),
    seed: int = 20260720,
    bootstrap_samples: int = 10_000,
    regenerate_features: bool = False,
    patch_config: PatchFeatureConfig = PatchFeatureConfig(),
    geometry_config: ReferenceTrackerConfig = ReferenceTrackerConfig(
        reference_weight=6.0,
        descriptor_weight=0.0,
    ),
) -> dict[str, object]:
    """Reproduce calibration, frozen confirmation, and no-intensity robustness.

    The path ordering is part of the deterministic seed rule: calibration has
    path index zero and held-out recordings have indices one onward.  The gated
    extent arm is the confirmatory comparison.  The no-intensity arm transfers
    its selected weight/update unchanged and is explicitly labelled post-hoc
    robustness.
    """

    calibration_source = Path(calibration_path).resolve()
    held_sources = tuple(Path(path).resolve() for path in held_out_paths)
    if not held_sources:
        raise ValueError("at least one held-out recording is required")
    all_sources = (calibration_source, *held_sources)
    if len(set(all_sources)) != len(all_sources):
        raise ValueError("calibration and held-out paths must be unique")
    stride_values = tuple(int(value) for value in strides)
    gap_values = tuple(int(value) for value in calibration_gap_strides)
    if not set(gap_values).issubset(stride_values):
        raise ValueError("calibration_gap_strides must be a subset of strides")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    started = perf_counter()
    cache_dir = Path(feature_cache_dir)
    recordings = tuple(load_wormid_recording(path) for path in all_sources)
    feature_pairs = tuple(
        _load_or_extract_benchmark_features(
            recording,
            cache_dir=cache_dir,
            patch_config=patch_config,
            regenerate=regenerate_features,
        )
        for recording in recordings
    )
    features = tuple(pair[0] for pair in feature_pairs)
    cache_paths = tuple(pair[1] for pair in feature_pairs)
    calibration = recordings[0]
    calibration_features = features[0]
    gated_scaler = fit_robust_feature_scaler(
        calibration_features.descriptors,
        reliability=calibration_features.reliability,
    )
    strict_scaler = fit_robust_feature_scaler(
        calibration_features.descriptors,
        reliability=None,
    )
    dense_threshold, _ = calibration_thresholds(calibration, strides=stride_values)

    calibration_items: dict[
        int,
        tuple[
            IdentityBlindSequence,
            AnonymousPatchSequence,
            ReferenceTrackResult,
        ],
    ] = {}
    severe_thresholds: dict[int, float] = {}
    for stride in stride_values:
        selected = _benchmark_frame_indices(calibration.frames, stride)
        sequence = identity_blind_sequence(
            calibration,
            frame_indices=selected,
            seed=seed + stride,
        )
        subset = _subset_patch_features(calibration_features, selected)
        anonymous = pair_patch_features_to_anonymous_detections(sequence, subset)
        baseline_result = track_sequence(sequence, "dual_anchor", config=geometry_config)
        severe_thresholds[stride] = float(
            np.quantile(baseline_result.cloud_motion_um[1:], 0.75)
        )
        calibration_items[stride] = (sequence, anonymous, baseline_result)

    grid_rows: list[dict[str, object]] = []
    best_objective = -np.inf
    selected_extent: DualAnchorExtentConfig | None = None
    for update in extent_updates:
        for weight in extent_weights:
            candidate = DualAnchorExtentConfig(
                extent_weight=float(weight),
                extent_update=float(update),
                feature_indices=calibration_features.extent_indices,
                use_reliability_gate=True,
            )
            accuracies: dict[str, float] = {}
            candidate_runtime = 0.0
            for stride in gap_values:
                sequence, anonymous, _ = calibration_items[stride]
                result = track_dual_anchor_extent(
                    sequence,
                    anonymous,
                    gated_scaler,
                    geometry_config=geometry_config,
                    extent_config=candidate,
                )
                accuracies[str(stride)] = score_tracks(sequence, result).identity_accuracy
                candidate_runtime += result.runtime_seconds
            objective = float(np.mean(list(accuracies.values())))
            grid_rows.append(
                {
                    "extent_weight": float(weight),
                    "extent_update": float(update),
                    "macro_seeded_identity_accuracy": objective,
                    "accuracy_by_stride": accuracies,
                    "runtime_seconds": candidate_runtime,
                }
            )
            if objective > best_objective:
                best_objective = objective
                selected_extent = candidate
    assert selected_extent is not None

    baseline_gap_objective = float(
        np.mean(
            [
                score_tracks(calibration_items[stride][0], calibration_items[stride][2])
                .identity_accuracy
                for stride in gap_values
            ]
        )
    )

    calibration_rows: list[dict[str, object]] = []
    strict_calibration_rows: list[dict[str, object]] = []
    strict_extent = DualAnchorExtentConfig(
        extent_weight=selected_extent.extent_weight,
        extent_update=selected_extent.extent_update,
        feature_indices=selected_extent.feature_indices,
        use_reliability_gate=False,
    )
    for stride in stride_values:
        sequence, anonymous, baseline_result = calibration_items[stride]
        gated_result = track_dual_anchor_extent(
            sequence,
            anonymous,
            gated_scaler,
            geometry_config=geometry_config,
            extent_config=selected_extent,
        )
        strict_result = track_dual_anchor_extent(
            sequence,
            anonymous,
            strict_scaler,
            geometry_config=geometry_config,
            extent_config=strict_extent,
        )
        for method, result in (
            ("dual_anchor", baseline_result),
            ("dual_anchor_extent", gated_result),
        ):
            row = _reference_metrics_with_strata(
                sequence,
                result,
                calibration,
                dense_threshold_um=dense_threshold,
                severe_threshold_um=severe_thresholds[stride],
            )
            row.update(
                recording_id=calibration.recording_id,
                split="calibration",
                method=method,
                stride=stride,
                frames_evaluated=len(sequence.frames),
            )
            calibration_rows.append(row)
        strict_row = _reference_metrics_with_strata(
            sequence,
            strict_result,
            calibration,
            dense_threshold_um=dense_threshold,
            severe_threshold_um=severe_thresholds[stride],
        )
        strict_row.update(
            recording_id=calibration.recording_id,
            split="calibration",
            method="dual_anchor_extent_no_intensity_gate",
            stride=stride,
            frames_evaluated=len(sequence.frames),
        )
        strict_calibration_rows.append(strict_row)

    held_rows: list[dict[str, object]] = []
    strict_rows: list[dict[str, object]] = []
    for path_index, (recording, measured) in enumerate(
        zip(recordings[1:], features[1:], strict=True),
        start=1,
    ):
        for stride in stride_values:
            selected = _benchmark_frame_indices(recording.frames, stride)
            sequence = identity_blind_sequence(
                recording,
                frame_indices=selected,
                seed=seed + 1000 * path_index + stride,
            )
            subset = _subset_patch_features(measured, selected)
            anonymous = pair_patch_features_to_anonymous_detections(sequence, subset)
            baseline_result = track_sequence(
                sequence,
                "dual_anchor",
                config=geometry_config,
            )
            gated_result = track_dual_anchor_extent(
                sequence,
                anonymous,
                gated_scaler,
                geometry_config=geometry_config,
                extent_config=selected_extent,
            )
            strict_result = track_dual_anchor_extent(
                sequence,
                anonymous,
                strict_scaler,
                geometry_config=geometry_config,
                extent_config=strict_extent,
            )
            for method, result in (
                ("dual_anchor", baseline_result),
                ("dual_anchor_extent", gated_result),
            ):
                row = _reference_metrics_with_strata(
                    sequence,
                    result,
                    recording,
                    dense_threshold_um=dense_threshold,
                    severe_threshold_um=severe_thresholds[stride],
                )
                row.update(
                    recording_id=recording.recording_id,
                    split="held_out",
                    method=method,
                    stride=stride,
                    frames_evaluated=len(sequence.frames),
                    first_source_frame=selected[0],
                    last_source_frame=selected[-1],
                )
                held_rows.append(row)
            strict_row = _reference_metrics_with_strata(
                sequence,
                strict_result,
                recording,
                dense_threshold_um=dense_threshold,
                severe_threshold_um=severe_thresholds[stride],
            )
            strict_row.update(
                recording_id=recording.recording_id,
                split="held_out",
                method="dual_anchor_extent_no_intensity_gate",
                stride=stride,
                frames_evaluated=len(sequence.frames),
                first_source_frame=selected[0],
                last_source_frame=selected[-1],
            )
            strict_rows.append(strict_row)

    held_recording_ids = tuple(recording.recording_id for recording in recordings[1:])
    confirmatory = _aggregate_reference_comparison(
        held_rows,
        recording_ids=held_recording_ids,
        strides=stride_values,
        baseline="dual_anchor",
        improved="dual_anchor_extent",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    baseline_rows = [row for row in held_rows if row["method"] == "dual_anchor"]
    robustness_rows = [*baseline_rows, *strict_rows]
    robustness = _aggregate_reference_comparison(
        robustness_rows,
        recording_ids=held_recording_ids,
        strides=stride_values,
        baseline="dual_anchor",
        improved="dual_anchor_extent_no_intensity_gate",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    payload: dict[str, object] = {
        "schema_version": 2,
        "analysis_roles": {
            "confirmatory_primary": "reliability-gated extent selected on calibration only",
            "post_hoc_robustness": (
                "no-intensity gate; selected weight/update transferred unchanged after "
                "the first held-out analysis and never retuned on held-out labels"
            ),
        },
        "seed": seed,
        "seed_rule": "seed + 1000*path_index + stride; calibration path_index=0",
        "calibration_recording_id": calibration.recording_id,
        "held_out_recording_ids": list(held_recording_ids),
        "strides": list(stride_values),
        "calibration_gap_strides": list(gap_values),
        "patch_config": asdict(patch_config),
        "measurement_boundary": {
            "raw_movie_path": "acquisition/CalciumImageSeries/data",
            "stored_trace_arrays_read": False,
            "identity_labels_used_for_association": False,
            "detection_rows_independently_permuted_every_frame": True,
            "total_intensity_is_descriptor_feature": False,
        },
        "geometry_config": asdict(geometry_config),
        "feature_scalers": {
            "confirmatory": "calibration-only median/IQR over reliable patches",
            "no_intensity": (
                "calibration-only median/IQR over every patch; no reliability filtering"
            ),
        },
        "sources": [
            {
                "recording_id": recording.recording_id,
                "split": "calibration" if index == 0 else "held_out",
                "path": recording.path,
                "size_bytes": Path(recording.path).stat().st_size,
                "frames": recording.frames,
                "neurons": recording.neurons,
                "feature_cache": str(cache_paths[index].resolve()),
                "feature_extraction_seconds": features[index].extraction_seconds,
            }
            for index, recording in enumerate(recordings)
        ],
        "calibration_grid": {
            "extent_weights": [float(value) for value in extent_weights],
            "extent_updates": [float(value) for value in extent_updates],
            "baseline_macro_seeded_identity_accuracy": baseline_gap_objective,
            "selected_macro_seeded_identity_accuracy": best_objective,
            "selected_config": asdict(selected_extent),
            "candidates": grid_rows,
        },
        "evaluation_strata": {
            "dense_threshold_um": dense_threshold,
            "dense_source": "calibration-animal pooled nearest-neighbor Q25",
            "severe_threshold_um_by_stride": {
                str(key): value for key, value in severe_thresholds.items()
            },
            "severe_source": (
                "calibration-animal identity-blind cloud-motion Q75 for each stride"
            ),
        },
        "calibration_results": calibration_rows,
        "confirmatory_held_out_results": held_rows,
        "confirmatory_summary": confirmatory,
        "no_intensity_config": asdict(strict_extent),
        "no_intensity_calibration_results": strict_calibration_rows,
        "no_intensity_held_out_results": strict_rows,
        "no_intensity_summary": robustness,
        "runtime_seconds": perf_counter() - started,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return payload


def _parse_benchmark_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, help="calibration EY NWB")
    parser.add_argument("--held-out", nargs="+", required=True, help="held-out EY NWBs")
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument(
        "--calibration-gap-strides",
        nargs="+",
        type=int,
        default=[5, 10, 20, 40],
    )
    parser.add_argument(
        "--extent-weights",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1, 2, 4, 8, 16, 32],
    )
    parser.add_argument(
        "--extent-updates",
        nargs="+",
        type=float,
        default=[0, 0.05, 0.1, 0.15, 0.25, 0.5, 1],
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--regenerate-features", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the exact five-animal benchmark."""

    args = _parse_benchmark_args(argv)
    payload = run_wormid_extent_benchmark(
        args.calibration,
        args.held_out,
        feature_cache_dir=args.feature_cache_dir,
        output_path=args.output,
        strides=args.strides,
        calibration_gap_strides=args.calibration_gap_strides,
        extent_weights=args.extent_weights,
        extent_updates=args.extent_updates,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        regenerate_features=args.regenerate_features,
    )
    selected = payload["calibration_grid"]
    confirmatory = payload["confirmatory_summary"]
    robustness = payload["no_intensity_summary"]
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "selected_config": selected["selected_config"],
                "confirmatory_bootstrap": confirmatory[
                    "complete_recording_paired_bootstrap"
                ],
                "no_intensity_bootstrap": robustness[
                    "complete_recording_paired_bootstrap"
                ],
                "runtime_seconds": payload["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
