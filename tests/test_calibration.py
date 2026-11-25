import numpy as np
from scipy.special import expit, logit

from deformeot.calibration import (
    TemperatureCalibrator,
    adaptive_bin_ece,
    brier_score,
    credible_set_coverage,
    evaluate_binary_calibration,
    fixed_bin_ece,
    negative_log_likelihood,
    recording_bootstrap,
    risk_coverage_curve,
)


def _exactly_calibrated_binary_sample():
    probabilities = []
    correctness = []
    for numerator in range(1, 10):
        probability = numerator / 10
        probabilities.extend([probability] * 100)
        correctness.extend([1] * (10 * numerator) + [0] * (100 - 10 * numerator))
    return np.asarray(probabilities), np.asarray(correctness)


def test_known_calibrated_probabilities_beat_shrunk_miscalibrated_probabilities():
    probabilities, correctness = _exactly_calibrated_binary_sample()
    miscalibrated = 0.25 + 0.5 * probabilities

    calibrated_report = evaluate_binary_calibration(probabilities, correctness, n_bins=10)
    miscalibrated_report = evaluate_binary_calibration(
        miscalibrated, correctness, n_bins=10
    )
    assert calibrated_report.fixed_ece.ece < 1e-12
    assert calibrated_report.adaptive_ece.ece < 1e-12
    assert miscalibrated_report.fixed_ece.ece > 0.1
    assert calibrated_report.negative_log_likelihood < (
        miscalibrated_report.negative_log_likelihood
    )
    assert calibrated_report.brier < miscalibrated_report.brier


def test_ece_is_invariant_to_row_order_and_adaptive_bins_keep_ties_together():
    probabilities, correctness = _exactly_calibrated_binary_sample()
    rng = np.random.default_rng(7)
    order = rng.permutation(len(probabilities))
    fixed_before = fixed_bin_ece(probabilities, correctness, n_bins=13)
    fixed_after = fixed_bin_ece(probabilities[order], correctness[order], n_bins=13)
    adaptive_before = adaptive_bin_ece(probabilities, correctness, n_bins=13)
    adaptive_after = adaptive_bin_ece(
        probabilities[order], correctness[order], n_bins=13
    )
    assert np.isclose(fixed_before.ece, fixed_after.ece)
    assert np.isclose(adaptive_before.ece, adaptive_after.ece)
    assert adaptive_before.occupied_bins <= len(np.unique(probabilities))


def test_categorical_credible_sets_report_actual_mass_and_detect_miscalibration():
    posterior = np.array([0.6, 0.3, 0.1])
    calibrated_truth = np.array([0] * 6 + [1] * 3 + [2])
    calibrated = credible_set_coverage(
        np.repeat(posterior[None, :], 10, axis=0),
        calibrated_truth,
        levels=(0.5, 0.8),
    )
    assert np.isclose(calibrated[0].mean_posterior_mass, 0.6)
    assert np.isclose(calibrated[0].empirical_coverage, 0.6)
    assert np.isclose(calibrated[0].mass_calibration_gap, 0)
    assert np.isclose(calibrated[1].mean_posterior_mass, 0.9)
    assert np.isclose(calibrated[1].empirical_coverage, 0.9)

    wrong_truth = np.array([0] * 3 + [1] * 3 + [2] * 4)
    miscalibrated = credible_set_coverage(
        [posterior.copy() for _ in range(10)], wrong_truth, levels=(0.5,)
    )[0]
    assert np.isclose(miscalibrated.empirical_coverage, 0.3)
    assert np.isclose(miscalibrated.mass_calibration_gap, -0.3)


def test_risk_coverage_and_aurc_reward_ranking_errors_last_and_handle_ties():
    correctness = np.array([1, 0, 1, 0, 1, 1])
    well_ranked = risk_coverage_curve([0.99, 0.10, 0.9, 0.20, 0.8, 0.7], correctness)
    reversed_rank = risk_coverage_curve([0.10, 0.99, 0.2, 0.90, 0.8, 0.7], correctness)
    assert well_ranked.aurc < reversed_rank.aurc
    assert np.all(np.diff(well_ranked.coverage) > 0)
    assert np.isclose(well_ranked.risk[-1], 2 / 6)

    tied_a = risk_coverage_curve([0.8, 0.8, 0.4, 0.4], [1, 0, 1, 0])
    tied_b = risk_coverage_curve([0.8, 0.8, 0.4, 0.4], [0, 1, 0, 1])
    assert np.isclose(tied_a.aurc, tied_b.aurc)
    assert np.allclose(tied_a.risk, tied_b.risk)


def test_temperature_fit_uses_only_declared_split_and_repairs_known_scaling():
    base_probability, calibration_truth = _exactly_calibrated_binary_sample()
    overconfident = expit(3.0 * logit(base_probability))
    # Appended evaluation labels deliberately favor a conflicting temperature.
    probabilities = np.concatenate([overconfident, overconfident])
    correctness = np.concatenate([calibration_truth, 1 - calibration_truth])
    split = np.array(["calibration"] * len(overconfident) + ["evaluation"] * len(overconfident))

    fitted = TemperatureCalibrator.fit(
        probabilities,
        correctness,
        split_labels=split,
        calibration_split="calibration",
    )
    fitted_calibration_only = TemperatureCalibrator.fit(
        overconfident,
        calibration_truth,
        split_labels=np.array(["calibration"] * len(overconfident)),
        calibration_split="calibration",
    )
    transformed = fitted.transform(overconfident)
    assert 2.9 < fitted.temperature < 3.1
    assert np.isclose(fitted.temperature, fitted_calibration_only.temperature)
    assert fitted.calibration_events == len(overconfident)
    assert negative_log_likelihood(transformed, calibration_truth) < (
        negative_log_likelihood(overconfident, calibration_truth)
    )
    assert brier_score(transformed, calibration_truth) < (
        brier_score(overconfident, calibration_truth)
    )


def test_recording_bootstrap_is_reproducible_and_macro_weights_recordings_equally():
    # A long easy recording and a short difficult recording must each get one vote.
    probabilities = np.concatenate([np.full(100, 0.9), np.full(10, 0.9)])
    correctness = np.concatenate([np.ones(100), np.zeros(10)])
    recordings = np.array(["long"] * 100 + ["short"] * 10)
    expected_macro = np.mean(
        [
            brier_score(probabilities[:100], correctness[:100]),
            brier_score(probabilities[100:], correctness[100:]),
        ]
    )
    first = recording_bootstrap(
        probabilities,
        correctness,
        recordings,
        metric="brier",
        n_resamples=500,
        seed=23,
    )
    second = recording_bootstrap(
        probabilities,
        correctness,
        recordings,
        metric="brier",
        n_resamples=500,
        seed=23,
    )
    pooled = recording_bootstrap(
        probabilities,
        correctness,
        recordings,
        metric="brier",
        n_resamples=200,
        seed=23,
        aggregation="pooled",
    )
    assert np.isclose(first.point_estimate, expected_macro)
    assert np.array_equal(first.replicates, second.replicates)
    assert first.n_groups == 2
    assert first.lower <= first.point_estimate <= first.upper
    assert pooled.point_estimate < first.point_estimate

