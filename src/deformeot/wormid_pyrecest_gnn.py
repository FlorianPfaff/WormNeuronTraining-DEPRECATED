"""PyRecEst GNN experiments for the initialized, exhaustive WormID cohort.

The control uses the original dual-anchor cost through PyRecEst's external
pairwise-cost API.  The Kalman variant transports means and covariances through
the same anonymous-cloud affine estimate, then combines normalized Mahalanobis
distance with the immutable first-frame anchor.  This is an online adapter:
neither future frames nor hidden identities are consumed.

PyRecEst is optional and imported on demand.  Its nearest-neighbor tracker
currently requires the NumPy backend.  Covariance and local-velocity settings
are experimental motion models; they are not calibrated identity confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .wormid import IdentityBlindSequence
from .wormid_benchmark import (
    ReferenceTrackerConfig,
    TrackResult,
    _affine,
    _spatial_cost,
    _validate_positions,
    descriptor_cost,
    distance_profile_descriptor,
)


@dataclass(frozen=True)
class PyRecEstGNNConfig:
    """Small, explicit family of online PyRecEst integration settings.

    ``equivalence`` reproduces the frozen dual-anchor assignment objective.
    ``kalman`` uses recursive position filtering and Mahalanobis association.
    Noise standard deviations are in the input coordinate unit (micrometers
    in WormID).  Local velocity is the smoothed displacement remaining after
    shared affine transport, per selected frame interval.  Its default weight
    is zero.  Adaptive process noise optionally averages displacement outer
    products, using previous assignments only.
    """

    mode: str = "kalman"
    measurement_noise_um: float = 0.35
    process_noise_um: float = 0.75
    local_velocity_weight: float = 0.0
    velocity_decay: float = 0.5
    adaptive_process_noise: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"equivalence", "kalman"}:
            raise ValueError("mode must be 'equivalence' or 'kalman'")
        values = (
            self.measurement_noise_um,
            self.process_noise_um,
            self.local_velocity_weight,
            self.velocity_decay,
            self.adaptive_process_noise,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("GNN configuration values must be finite")
        if self.measurement_noise_um <= 0 or self.process_noise_um <= 0:
            raise ValueError("measurement and process noise must be positive")
        if not 0 <= self.local_velocity_weight <= 1:
            raise ValueError("local_velocity_weight must lie in [0, 1]")
        if not 0 <= self.velocity_decay < 1:
            raise ValueError("velocity_decay must lie in [0, 1)")
        if not 0 <= self.adaptive_process_noise <= 1:
            raise ValueError("adaptive_process_noise must lie in [0, 1]")


def _make_tracker(reference: np.ndarray, config: PyRecEstGNNConfig):
    try:
        import pyrecest.backend
        from pyrecest.distributions import GaussianDistribution
        from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor
    except ImportError as exc:
        raise ImportError(
            "PyRecEst GNN experiments require PyRecEst in a separate compatible "
            "environment; install the pinned vendor checkout first."
        ) from exc
    if pyrecest.backend.__backend_name__ != "numpy":
        raise NotImplementedError("WormID PyRecEst GNN requires the NumPy backend")

    class AssignmentRecordingGNN(GlobalNearestNeighbor):
        # update_linear does not return its association. Capture that exact
        # association while preserving PyRecEst's own association and updates.
        def find_association(self, *args, **kwargs):
            association = super().find_association(*args, **kwargs)
            self.last_assignment = np.asarray(association, dtype=np.int64).copy()
            return association

    covariance = np.eye(3) * config.measurement_noise_um**2
    tracker = AssignmentRecordingGNN(
        initial_prior=[GaussianDistribution(point, covariance) for point in reference],
        association_param={
            "distance_metric_pos": "Euclidean" if config.mode == "equivalence" else "Mahalanobis",
            "square_dist": False,
            # The evaluated dataset supplies every initialized neuron each
            # frame. Infinite gates and the fully finite square real cost
            # preserve exhaustive assignment without births or abstentions.
            "gating_distance_threshold": float("inf"),
            "max_new_tracks": 0,
            # The standard route avoids cost rescaling in the maximum-
            # cardinality helper, preserving the original numerical ties.
            "maximize_cardinality": False,
            "pairwise_cost_weight": 1.0,
        },
        log_prior_estimates=False,
        log_posterior_estimates=False,
    )
    return tracker, GaussianDistribution


def _mahalanobis_cost(tracker, current: np.ndarray, measurement_covariance: np.ndarray):
    """Return PyRecEst's unsquared cost and a physical covariance scale."""

    states = tracker.filter_state
    means = np.stack([state.mu for state in states])
    covariances = np.stack([state.C for state in states])
    covariance_scale_um = float(
        np.sqrt(np.median(np.trace(covariances + measurement_covariance, axis1=1, axis2=2)) / 3)
    )
    if np.all(covariances == covariances[0]):
        cost = cdist(
            means,
            current,
            "mahalanobis",
            VI=np.linalg.inv(covariances[0] + measurement_covariance),
        )
    else:
        cost = np.stack(
            [
                cdist(
                    mean[None],
                    current,
                    "mahalanobis",
                    VI=np.linalg.inv(covariance + measurement_covariance),
                )[0]
                for mean, covariance in zip(means, covariances)
            ]
        )
    return cost, covariance_scale_um


def track_pyrecest_gnn(
    sequence: IdentityBlindSequence,
    *,
    config: PyRecEstGNNConfig = PyRecEstGNNConfig(),
    geometry: ReferenceTrackerConfig = ReferenceTrackerConfig(),
) -> TrackResult:
    """Track only ``sequence.frames`` and return exhaustive detection indices.

    Affine registration is estimated from the previous observed cloud, then
    applied to the recursive means and covariances. The reference and temporal
    cues are correlated, so their weighted association objective should not be
    interpreted as a product of independent calibrated probabilities.
    """

    if not sequence.frames:
        raise ValueError("sequence is empty")
    reference = _validate_positions(sequence.frames[0].positions)
    count = len(reference)
    if any(len(frame.positions) != count for frame in sequence.frames):
        raise ValueError("PyRecEst WormID GNN requires a fixed exhaustive cohort")

    started = time.perf_counter()
    tracker, GaussianDistribution = _make_tracker(reference, config)
    assignments = [np.arange(count, dtype=np.int64)]
    previous = reference.copy()
    measurement_covariance = np.eye(3) * config.measurement_noise_um**2
    process_covariance = np.eye(3) * config.process_noise_um**2
    local_velocity = np.zeros_like(reference)
    adaptive_covariance = np.zeros((count, 3, 3))
    reference_descriptor = distance_profile_descriptor(
        reference, quantiles=geometry.descriptor_quantiles
    )
    cloud_motion = [0.0]

    for frame in sequence.frames[1:]:
        current = _validate_positions(frame.positions)
        forward = np.min(cdist(previous, current), axis=1)
        backward = np.min(cdist(current, previous), axis=1)
        cloud_motion.append(float(0.5 * (np.median(forward) + np.median(backward))))
        transport = _affine(previous, current, geometry)
        reference_prediction = _affine(reference, current, geometry).apply(reference)
        anchor = _spatial_cost(
            reference_prediction, current, geometry.spatial_scale_floor_um
        )

        if config.mode == "equivalence":
            prediction = transport.apply(previous)
            tracker.filter_state = [
                GaussianDistribution(point, measurement_covariance) for point in prediction
            ]
            raw_geometric = cdist(prediction, current)
            temporal = _spatial_cost(prediction, current, geometry.spatial_scale_floor_um)
        else:
            previous_means = np.asarray(tracker.get_point_estimate()).T
            affine_prediction = transport.apply(previous_means)
            local_velocity = local_velocity @ transport.matrix.T
            adaptive_covariance = np.einsum(
                "ij,njk,lk->nil", transport.matrix, adaptive_covariance, transport.matrix
            )
            noises = np.moveaxis(process_covariance[None] + adaptive_covariance, 0, 2)
            deterministic_inputs = (
                transport.offset[None] + config.local_velocity_weight * local_velocity
            ).T
            tracker.predict_linear(transport.matrix, noises, deterministic_inputs)
            raw_geometric, covariance_scale_um = _mahalanobis_cost(
                tracker, current, measurement_covariance
            )
            rows, cols = linear_sum_assignment(raw_geometric)
            # Convert the frozen physical floor to Mahalanobis units. A bare
            # tiny dimensionless floor would magnify the temporal cue whenever
            # quantized coordinates give a zero median residual.
            scale = max(
                float(np.median(raw_geometric[rows, cols])),
                geometry.spatial_scale_floor_um / covariance_scale_um,
            )
            temporal = raw_geometric / scale

        # GNN adds its own geometric term to this external cost. Subtract that
        # same term so the resulting objective is exactly the intended fused
        # temporal/reference objective, with no accidental duplicate cue.
        pairwise_cost = temporal + geometry.reference_weight * anchor - raw_geometric
        if geometry.descriptor_weight:
            current_descriptor = distance_profile_descriptor(
                current, quantiles=geometry.descriptor_quantiles
            )
            pairwise_cost += geometry.descriptor_weight * descriptor_cost(
                reference_descriptor, current_descriptor
            )

        tracker.update_linear(
            current.T,
            np.eye(3),
            measurement_covariance,
            pairwise_cost_matrix=pairwise_cost,
        )
        assignment = tracker.last_assignment.copy()
        if not np.array_equal(np.sort(assignment), np.arange(count)):
            raise RuntimeError("PyRecEst GNN did not produce an exhaustive permutation")
        assignments.append(assignment)
        previous = current[assignment]

        if config.mode == "kalman":
            displacement = previous - affine_prediction
            local_velocity = (
                config.velocity_decay * local_velocity
                + (1 - config.velocity_decay) * displacement
            )
            rate = config.adaptive_process_noise
            adaptive_covariance = (
                (1 - rate) * adaptive_covariance
                + rate * np.einsum("ni,nj->nij", displacement, displacement)
            )

    return TrackResult(
        method=f"pyrecest_gnn_{config.mode}",
        assignments=tuple(assignments),
        runtime_seconds=float(time.perf_counter() - started),
        cloud_motion_um=np.asarray(cloud_motion),
    )
