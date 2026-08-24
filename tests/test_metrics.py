"""Tests for evaluation/metrics.py (threshold search, EER, calibration).

Uses small synthetic score/label arrays — no model or network access.
Requires the `calibration` extra (`pip install -e ".[calibration]"`, for
scikit-learn); skipped entirely if unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from audio_deepfake_detector.evaluation.metrics import (
    apply_temperature,
    best_balanced_accuracy_threshold,
    best_f1_threshold,
    brier_score,
    compute_eer,
    compute_roc_auc,
    compute_threshold_metrics,
    expected_calibration_error,
    fit_temperature_scaling,
    max_fpr_threshold,
    prob_to_logit,
)

# A perfectly separable toy set: bonafide scores near 0, spoof scores near 1.
Y_TRUE_SEPARABLE = np.array([0, 0, 0, 0, 1, 1, 1, 1])
Y_SCORES_SEPARABLE = np.array([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95])


def test_compute_threshold_metrics_perfect_separation():
    m = compute_threshold_metrics(Y_TRUE_SEPARABLE, Y_SCORES_SEPARABLE, 0.5)
    assert m.tpr == pytest.approx(1.0)
    assert m.fpr == pytest.approx(0.0)
    assert m.accuracy == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_compute_eer_near_zero_for_separable_data():
    eer, threshold = compute_eer(Y_TRUE_SEPARABLE, Y_SCORES_SEPARABLE)
    assert eer == pytest.approx(0.0, abs=1e-6)
    assert 0.2 < threshold <= 0.8


def test_compute_roc_auc_perfect_for_separable_data():
    assert compute_roc_auc(Y_TRUE_SEPARABLE, Y_SCORES_SEPARABLE) == pytest.approx(1.0)


def test_best_balanced_accuracy_threshold_finds_separating_threshold():
    m = best_balanced_accuracy_threshold(Y_TRUE_SEPARABLE, Y_SCORES_SEPARABLE)
    assert m.balanced_accuracy == pytest.approx(1.0)


def test_best_f1_threshold_finds_separating_threshold():
    m = best_f1_threshold(Y_TRUE_SEPARABLE, Y_SCORES_SEPARABLE)
    assert m.f1 == pytest.approx(1.0)


def test_max_fpr_threshold_returns_none_when_unachievable():
    # One bonafide sample scores 1.0 (the top of the sweep range), so for
    # EVERY threshold strictly below 1.0 that bonafide sample is a false
    # positive -> fpr >= 1/2 = 0.5 at every swept threshold, never <= 5%.
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([1.0, 0.1, 0.9, 0.95])
    result = max_fpr_threshold(y_true, y_scores, max_fpr=0.05)
    assert result is None


def test_max_fpr_threshold_finds_valid_threshold_when_achievable():
    result = max_fpr_threshold(Y_TRUE_SEPARABLE, Y_SCORES_SEPARABLE, max_fpr=0.05)
    assert result is not None
    assert result.fpr <= 0.05


def test_brier_score_zero_for_perfect_confident_predictions():
    y_true = np.array([0, 1])
    y_scores = np.array([0.0, 1.0])
    assert brier_score(y_true, y_scores) == pytest.approx(0.0)


def test_brier_score_worst_case():
    y_true = np.array([0, 1])
    y_scores = np.array([1.0, 0.0])
    assert brier_score(y_true, y_scores) == pytest.approx(1.0)


def test_expected_calibration_error_zero_for_perfectly_calibrated_bins():
    # Half the samples score 0.0 and are all bonafide; half score 1.0 and
    # are all spoof -> each bin's accuracy matches its confidence exactly.
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y_true, y_scores, n_bins=10) == pytest.approx(0.0)


def test_temperature_scaling_reduces_or_maintains_brier_on_overconfident_scores():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    # Deliberately overconfident: push scores toward 0/1 regardless of truth quality
    base_scores = np.clip(y_true * 0.9 + rng.normal(0, 0.05, size=200), 0.01, 0.99)
    logits = prob_to_logit(base_scores)
    temperature = fit_temperature_scaling(logits, y_true)
    calibrated = apply_temperature(logits, temperature)

    assert temperature > 0
    assert brier_score(y_true, calibrated) <= brier_score(y_true, base_scores) + 1e-9


def test_prob_to_logit_round_trip():
    p = np.array([0.1, 0.5, 0.9])
    logit = prob_to_logit(p)
    recovered = 1 / (1 + np.exp(-logit))
    assert np.allclose(recovered, p, atol=1e-6)
