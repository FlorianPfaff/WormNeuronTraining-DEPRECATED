"""Calibration and selective-risk evaluation for association posteriors.

Tracking produces dependent observations: thousands of frame-level links can come
from only a handful of animals.  This module therefore separates event-level
calibration metrics from recording-level uncertainty estimation.  The bootstrap
functions resample complete groups and never individual frames.

Binary functions consume the probability assigned to the selected association and
whether that association was correct.  :func:`credible_set_coverage` instead uses
the complete categorical posterior, including the missed-detection event when it
is part of the tracker output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Hashable, Literal, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


Array = np.ndarray
ScalarStatistic = Callable[[Array, Array], float]


def _binary_inputs(
    probabilities: Sequence[float] | Array,
    correctness: Sequence[bool | int] | Array,
) -> tuple[Array, Array]:
    probabilities = np.asarray(probabilities, dtype=float)
    correctness = np.asarray(correctness)
    if probabilities.ndim != 1 or correctness.ndim != 1:
        raise ValueError("probabilities and correctness must be one-dimensional")
    if len(probabilities) != len(correctness) or len(probabilities) == 0:
        raise ValueError("probabilities and correctness must have equal nonzero length")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if correctness.dtype == np.bool_:
        labels = correctness.astype(float)
    else:
        try:
            labels = correctness.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("correctness must contain only binary values") from error
        if not np.all(np.isfinite(labels)) or not np.all(np.isin(labels, [0.0, 1.0])):
            raise ValueError("correctness must contain only 0/1 or boolean values")
    return probabilities, labels


def negative_log_likelihood(
    probabilities: Sequence[float] | Array,
    correctness: Sequence[bool | int] | Array,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Mean Bernoulli negative log likelihood in nats."""

    probabilities, labels = _binary_inputs(probabilities, correctness)
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))


def brier_score(
    probabilities: Sequence[float] | Array,
    correctness: Sequence[bool | int] | Array,
) -> float:
    """Mean binary Brier score; lower is better."""

    probabilities, labels = _binary_inputs(probabilities, correctness)
    return float(np.mean((probabilities - labels) ** 2))


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    empirical_accuracy: float
    absolute_gap: float


@dataclass(frozen=True)
class ECEResult:
    """Expected calibration error and the reliability bins behind it."""

    ece: float
    strategy: Literal["fixed", "adaptive"]
    requested_bins: int
    occupied_bins: int
    bins: tuple[ReliabilityBin, ...]


def _ece_from_indices(
    probabilities: Array,
    labels: Array,
    indices: Array,
    bounds: Sequence[tuple[float, float]],
    *,
    strategy: Literal["fixed", "adaptive"],
    requested_bins: int,
) -> ECEResult:
    reliability_bins: list[ReliabilityBin] = []
    weighted_gap = 0.0
    for index, (lower, upper) in enumerate(bounds):
        selected = indices == index
        count = int(selected.sum())
        if count:
            mean_probability = float(np.mean(probabilities[selected]))
            accuracy = float(np.mean(labels[selected]))
            gap = abs(mean_probability - accuracy)
            weighted_gap += count * gap / len(probabilities)
        else:
            mean_probability = float("nan")
            accuracy = float("nan")
            gap = float("nan")
        reliability_bins.append(
            ReliabilityBin(
                float(lower),
                float(upper),
                count,
                mean_probability,
                accuracy,
                float(gap),
            )
        )
    return ECEResult(
        ece=float(weighted_gap),
        strategy=strategy,
        requested_bins=requested_bins,
        occupied_bins=sum(item.count > 0 for item in reliability_bins),
        bins=tuple(reliability_bins),
    )


def fixed_bin_ece(
    probabilities: Sequence[float] | Array,
    correctness: Sequence[bool | int] | Array,
    *,
    n_bins: int = 15,
) -> ECEResult:
    """ECE with equal-width bins over [0, 1].

    Empty bins are retained in ``bins`` so reliability diagrams have fixed axes.
    """

    probabilities, labels = _binary_inputs(probabilities, correctness)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.searchsorted(edges[1:-1], probabilities, side="right")
    bounds = list(zip(edges[:-1], edges[1:]))
    return _ece_from_indices(
        probabilities,
        labels,
        indices,
        bounds,
        strategy="fixed",
        requested_bins=n_bins,
    )


def adaptive_bin_ece(
    probabilities: Sequence[float] | Array,
    correctness: Sequence[bool | int] | Array,
    *,
    n_bins: int = 15,
) -> ECEResult:
    """Quantile-bin ECE without splitting tied confidence values.

    Ties are kept together to make the result invariant to input row order.  This
    can produce fewer occupied bins than requested, which is reported explicitly.
    """

    probabilities, labels = _binary_inputs(probabilities, correctness)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if n_bins == 1:
        cut_points = np.empty(0, dtype=float)
    else:
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        cut_points = np.unique(np.quantile(probabilities, quantiles, method="linear"))
        # A cut exactly equal to an endpoint creates no useful extra bin.
        cut_points = cut_points[(cut_points > 0) & (cut_points < 1)]
    indices = np.searchsorted(cut_points, probabilities, side="right")
    edges = np.concatenate([[0.0], cut_points, [1.0]])
    bounds = list(zip(edges[:-1], edges[1:]))
    return _ece_from_indices(
        probabilities,
        labels,
        indices,
        bounds,
        strategy="adaptive",
        requested_bins=n_bins,
    )


@dataclass(frozen=True)
class CredibleCoveragePoint:
    nominal_level: float
    empirical_coverage: float
    mean_posterior_mass: float
    mean_set_size: float
    nominal_gap: float
    mass_calibration_gap: float


def _posterior_rows(
    posteriors: Sequence[Sequence[float] | Array] | Array,
) -> tuple[Array, ...]:
    if isinstance(posteriors, np.ndarray) and posteriors.ndim == 2:
        rows = tuple(np.asarray(row, dtype=float) for row in posteriors)
    else:
        rows = tuple(np.asarray(row, dtype=float) for row in posteriors)
    if not rows:
        raise ValueError("posteriors cannot be empty")
    for row in rows:
        if row.ndim != 1 or len(row) == 0:
            raise ValueError("each posterior must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(row)) or np.any(row < 0):
            raise ValueError("posterior entries must be finite and nonnegative")
        if not np.isclose(row.sum(), 1.0, atol=1e-7, rtol=1e-7):
            raise ValueError("each categorical posterior must sum to one")
    return rows


def credible_set_coverage(
    posteriors: Sequence[Sequence[float] | Array] | Array,
    true_event_indices: Sequence[int] | Array,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.99),
) -> tuple[CredibleCoveragePoint, ...]:
    """Evaluate highest-posterior-density credible sets.

    All events tied at the inclusion boundary are retained, making coverage
    invariant to candidate ordering.  Because categorical sets overshoot their
    nominal mass, ``mass_calibration_gap`` (coverage minus actual included mass)
    is usually more interpretable than ``nominal_gap``.

    Rows may contain different numbers of candidates, as happens when the number
    of detections changes between frames.
    """

    rows = _posterior_rows(posteriors)
    truth = np.asarray(true_event_indices)
    if truth.ndim != 1 or len(truth) != len(rows):
        raise ValueError("true_event_indices must have one entry per posterior")
    try:
        truth = truth.astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError("true_event_indices must contain integers") from error
    if any(index < 0 or index >= len(row) for index, row in zip(truth, rows)):
        raise ValueError("a true event index is outside its posterior")
    levels_array = np.asarray(levels, dtype=float)
    if levels_array.ndim != 1 or len(levels_array) == 0:
        raise ValueError("levels must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(levels_array)) or np.any(
        (levels_array <= 0) | (levels_array > 1)
    ):
        raise ValueError("credible levels must lie in (0, 1]")

    results: list[CredibleCoveragePoint] = []
    for level in levels_array:
        covered: list[bool] = []
        set_masses: list[float] = []
        set_sizes: list[int] = []
        for row, true_index in zip(rows, truth):
            descending = np.sort(row)[::-1]
            boundary_index = int(np.searchsorted(np.cumsum(descending), level, side="left"))
            boundary_index = min(boundary_index, len(descending) - 1)
            cutoff = descending[boundary_index]
            included = row >= cutoff - 1e-14
            covered.append(bool(included[true_index]))
            set_masses.append(float(row[included].sum()))
            set_sizes.append(int(included.sum()))
        empirical = float(np.mean(covered))
        mass = float(np.mean(set_masses))
        results.append(
            CredibleCoveragePoint(
                nominal_level=float(level),
                empirical_coverage=empirical,
                mean_posterior_mass=mass,
                mean_set_size=float(np.mean(set_sizes)),
                nominal_gap=empirical - float(level),
                mass_calibration_gap=empirical - mass,
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class RiskCoverageResult:
    """Tie-invariant selective-risk curve and discrete AURC."""

    thresholds: Array
    coverage: Array
    risk: Array
    aurc: float
    total_events: int
    total_errors: int


def risk_coverage_curve(
    probabilities: Sequence[float] | Array,
    correctness: Sequence[bool | int] | Array,
) -> RiskCoverageResult:
    """Compute error risk as low-confidence associations are withheld.

    Curve points occur after complete confidence-tie blocks.  AURC is the expected
    discrete prefix risk under a random ordering within each tie block, avoiding
    arbitrary dependence on input row order.
    """

    probabilities, labels = _binary_inputs(probabilities, correctness)
    order = np.argsort(-probabilities, kind="stable")
    sorted_probability = probabilities[order]
    sorted_error = 1 - labels[order]
    thresholds: list[float] = []
    coverages: list[float] = []
    risks: list[float] = []
    selected = 0
    errors = 0.0
    aurc_sum = 0.0
    start = 0
    while start < len(probabilities):
        end = start + 1
        while end < len(probabilities) and sorted_probability[end] == sorted_probability[start]:
            end += 1
        block_size = end - start
        block_errors = float(np.sum(sorted_error[start:end]))
        # Expected cumulative errors for each possible prefix within a tied block.
        for within_block in range(1, block_size + 1):
            expected_errors = errors + within_block * block_errors / block_size
            aurc_sum += expected_errors / (selected + within_block)
        selected += block_size
        errors += block_errors
        thresholds.append(float(sorted_probability[start]))
        coverages.append(selected / len(probabilities))
        risks.append(errors / selected)
        start = end
    return RiskCoverageResult(
        thresholds=np.asarray(thresholds),
        coverage=np.asarray(coverages),
        risk=np.asarray(risks),
        aurc=float(aurc_sum / len(probabilities)),
        total_events=int(len(probabilities)),
        total_errors=int(round(errors)),
    )


@dataclass(frozen=True)
class BinaryCalibrationReport:
    events: int
    empirical_accuracy: float
    mean_probability: float
    negative_log_likelihood: float
    brier: float
    fixed_ece: ECEResult
    adaptive_ece: ECEResult
    risk_coverage: RiskCoverageResult


def evaluate_binary_calibration(
    probabilities: Sequence[float] | Array,
    correctness: Sequence[bool | int] | Array,
    *,
    n_bins: int = 15,
) -> BinaryCalibrationReport:
    """Create the standard binary association-calibration report."""

    probabilities, labels = _binary_inputs(probabilities, correctness)
    return BinaryCalibrationReport(
        events=len(probabilities),
        empirical_accuracy=float(np.mean(labels)),
        mean_probability=float(np.mean(probabilities)),
        negative_log_likelihood=negative_log_likelihood(probabilities, labels),
        brier=brier_score(probabilities, labels),
        fixed_ece=fixed_bin_ece(probabilities, labels, n_bins=n_bins),
        adaptive_ece=adaptive_bin_ece(probabilities, labels, n_bins=n_bins),
        risk_coverage=risk_coverage_curve(probabilities, labels),
    )


@dataclass(frozen=True)
class TemperatureCalibrator:
    """Held-out scalar temperature fit for binary association confidence."""

    temperature: float
    calibration_split: Hashable
    calibration_events: int
    calibration_groups: int | None
    group_balanced: bool
    optimizer_objective: float

    @classmethod
    def fit(
        cls,
        probabilities: Sequence[float] | Array,
        correctness: Sequence[bool | int] | Array,
        *,
        split_labels: Sequence[Hashable] | Array,
        calibration_split: Hashable,
        group_ids: Sequence[Hashable] | Array | None = None,
        temperature_bounds: tuple[float, float] = (0.05, 20.0),
    ) -> "TemperatureCalibrator":
        """Fit only rows explicitly assigned to ``calibration_split``.

        When ``group_ids`` are supplied, each recording receives equal total loss
        weight so a long movie cannot dominate the fitted temperature.
        """

        probabilities, labels = _binary_inputs(probabilities, correctness)
        splits = np.asarray(split_labels, dtype=object)
        if splits.ndim != 1 or len(splits) != len(probabilities):
            raise ValueError("split_labels must have one entry per event")
        selected = np.fromiter(
            (value == calibration_split for value in splits),
            dtype=bool,
            count=len(splits),
        )
        if selected.sum() < 2:
            raise ValueError("the declared calibration split needs at least two events")
        selected_probabilities = probabilities[selected]
        selected_labels = labels[selected]
        if len(np.unique(selected_labels)) < 2:
            raise ValueError("temperature fitting needs correct and incorrect calibration events")
        lower, upper = temperature_bounds
        if not (0 < lower < upper):
            raise ValueError("temperature_bounds must be positive and increasing")

        calibration_groups: int | None = None
        if group_ids is None:
            weights = np.full(len(selected_probabilities), 1 / len(selected_probabilities))
            group_balanced = False
        else:
            groups = np.asarray(group_ids, dtype=object)
            if groups.ndim != 1 or len(groups) != len(probabilities):
                raise ValueError("group_ids must have one entry per event")
            selected_groups = groups[selected]
            group_members: dict[Hashable, list[int]] = {}
            for index, group in enumerate(selected_groups):
                try:
                    group_members.setdefault(group, []).append(index)
                except TypeError as error:
                    raise ValueError("group_ids must be hashable") from error
            calibration_groups = len(group_members)
            weights = np.empty(len(selected_probabilities), dtype=float)
            for members in group_members.values():
                weights[members] = 1 / (len(group_members) * len(members))
            group_balanced = True

        clipped = np.clip(selected_probabilities, 1e-8, 1 - 1e-8)
        unscaled_logits = logit(clipped)

        def objective(log_temperature: float) -> float:
            calibrated = expit(unscaled_logits / math.exp(log_temperature))
            calibrated = np.clip(calibrated, 1e-12, 1 - 1e-12)
            losses = -(
                selected_labels * np.log(calibrated)
                + (1 - selected_labels) * np.log1p(-calibrated)
            )
            return float(np.sum(weights * losses))

        solution = minimize_scalar(
            objective,
            bounds=(math.log(lower), math.log(upper)),
            method="bounded",
            options={"xatol": 1e-10, "maxiter": 500},
        )
        if not solution.success or not np.isfinite(solution.fun):
            raise RuntimeError(f"temperature optimization failed: {solution.message}")
        return cls(
            temperature=float(math.exp(solution.x)),
            calibration_split=calibration_split,
            calibration_events=int(selected.sum()),
            calibration_groups=calibration_groups,
            group_balanced=group_balanced,
            optimizer_objective=float(solution.fun),
        )

    def transform(self, probabilities: Sequence[float] | Array) -> Array:
        """Apply the held-out temperature without refitting."""

        probabilities = np.asarray(probabilities, dtype=float)
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError("probabilities must be finite and lie in [0, 1]")
        clipped = np.clip(probabilities, 1e-8, 1 - 1e-8)
        return expit(logit(clipped) / self.temperature)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapResult:
    """Percentile interval from complete-group resampling."""

    point_estimate: float
    lower: float
    upper: float
    standard_error: float
    confidence_level: float
    n_groups: int
    n_resamples: int
    aggregation: Literal["macro", "pooled"]
    seed: int
    replicates: Array


def grouped_bootstrap(
    values: Array | Sequence[float],
    targets: Array | Sequence[float],
    group_ids: Sequence[Hashable] | Array,
    statistic: ScalarStatistic,
    *,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
    aggregation: Literal["macro", "pooled"] = "macro",
) -> BootstrapResult:
    """Bootstrap a scalar statistic by resampling complete recordings.

    ``macro`` (default) gives every recording equal weight and bootstraps the mean
    per-recording statistic.  ``pooled`` concatenates all events from each sampled
    recording, preserving recording clusters but weighting longer recordings more.
    """

    values = np.asarray(values)
    targets = np.asarray(targets)
    groups = np.asarray(group_ids, dtype=object)
    if values.ndim < 1 or targets.ndim < 1:
        raise ValueError("values and targets need an event axis")
    if len(values) == 0 or len(values) != len(targets) or len(values) != len(groups):
        raise ValueError("values, targets, and group_ids must have equal nonzero length")
    if groups.ndim != 1:
        raise ValueError("group_ids must be one-dimensional")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least two")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    if aggregation not in {"macro", "pooled"}:
        raise ValueError("aggregation must be 'macro' or 'pooled'")

    group_members: dict[Hashable, list[int]] = {}
    for event_index, group in enumerate(groups):
        try:
            group_members.setdefault(group, []).append(event_index)
        except TypeError as error:
            raise ValueError("group_ids must be hashable") from error
    if len(group_members) < 2:
        raise ValueError("grouped bootstrap needs at least two independent groups")
    members = [np.asarray(indices, dtype=int) for indices in group_members.values()]
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=float)

    if aggregation == "macro":
        group_statistics = np.asarray(
            [float(statistic(values[index], targets[index])) for index in members]
        )
        if not np.all(np.isfinite(group_statistics)):
            raise ValueError("statistic returned a nonfinite group estimate")
        point_estimate = float(np.mean(group_statistics))
        draws = rng.integers(0, len(members), size=(n_resamples, len(members)))
        replicates[:] = np.mean(group_statistics[draws], axis=1)
    else:
        point_estimate = float(statistic(values, targets))
        for bootstrap_index in range(n_resamples):
            sampled_groups = rng.integers(0, len(members), size=len(members))
            indices = np.concatenate([members[index] for index in sampled_groups])
            replicates[bootstrap_index] = float(statistic(values[indices], targets[indices]))
        if not np.isfinite(point_estimate) or not np.all(np.isfinite(replicates)):
            raise ValueError("statistic returned a nonfinite pooled estimate")

    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(replicates, [tail, 1 - tail])
    return BootstrapResult(
        point_estimate=point_estimate,
        lower=float(lower),
        upper=float(upper),
        standard_error=float(np.std(replicates, ddof=1)),
        confidence_level=confidence_level,
        n_groups=len(members),
        n_resamples=n_resamples,
        aggregation=aggregation,
        seed=seed,
        replicates=replicates,
    )


BinaryMetricName = Literal["nll", "brier", "fixed_ece", "adaptive_ece", "aurc"]


def recording_bootstrap(
    probabilities: Sequence[float] | Array,
    correctness: Sequence[bool | int] | Array,
    recording_ids: Sequence[Hashable] | Array,
    *,
    metric: BinaryMetricName = "brier",
    n_bins: int = 15,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
    aggregation: Literal["macro", "pooled"] = "macro",
) -> BootstrapResult:
    """Convenience wrapper for recording-bootstrap association metrics."""

    probabilities, labels = _binary_inputs(probabilities, correctness)
    statistics: dict[str, ScalarStatistic] = {
        "nll": negative_log_likelihood,
        "brier": brier_score,
        "fixed_ece": lambda value, target: fixed_bin_ece(
            value, target, n_bins=n_bins
        ).ece,
        "adaptive_ece": lambda value, target: adaptive_bin_ece(
            value, target, n_bins=n_bins
        ).ece,
        "aurc": lambda value, target: risk_coverage_curve(value, target).aurc,
    }
    if metric not in statistics:
        raise ValueError(f"unknown metric: {metric}")
    return grouped_bootstrap(
        probabilities,
        labels,
        recording_ids,
        statistics[metric],
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        aggregation=aggregation,
    )

