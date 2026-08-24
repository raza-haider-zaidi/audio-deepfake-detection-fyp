"""Tests for offline window/clip score diagnostics and aggregation
strategies (evaluation/diagnostics.py). No model/network access needed."""

import pytest

from audio_deepfake_detector.evaluation.diagnostics import (
    aggregate_spoof_score,
    compute_clip_diagnostics,
    trimmed_mean,
)
from audio_deepfake_detector.utils.datatypes import WindowPrediction


def _windows(spoof_scores: list[float]) -> list[WindowPrediction]:
    return [
        WindowPrediction(
            window_index=i,
            start_sample=i * 32000,
            end_sample=i * 32000 + 64000,
            raw_label="spoof" if s >= 0.5 else "bonafide",
            probabilities={"bonafide": 1 - s, "spoof": s},
        )
        for i, s in enumerate(spoof_scores)
    ]


def test_compute_clip_diagnostics_basic_stats():
    diag = compute_clip_diagnostics(_windows([0.1, 0.9, 0.5, 0.3]))
    assert diag.n_windows == 4
    assert diag.mean_spoof == pytest.approx(0.45)
    assert diag.median_spoof == pytest.approx(0.4)
    assert diag.min_spoof == pytest.approx(0.1)
    assert diag.max_spoof == pytest.approx(0.9)
    assert diag.n_windows_spoof_vote == 2  # 0.9 and 0.5 (>= 0.5 threshold)
    assert diag.n_windows_bonafide_vote == 2
    assert diag.proportion_spoof_vote == pytest.approx(0.5)


def test_compute_clip_diagnostics_rejects_empty():
    with pytest.raises(ValueError):
        compute_clip_diagnostics([])


def test_trimmed_mean_matches_plain_mean_for_small_n():
    assert trimmed_mean(__import__("numpy").array([0.1, 0.9])) == pytest.approx(0.5)


def test_trimmed_mean_removes_outliers():
    import numpy as np

    scores = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
    # 10% cut of 10 elements = floor(1) => trims 1 from each tail
    result = trimmed_mean(scores, proportion_to_cut=0.1)
    assert result == pytest.approx(np.mean(scores[1:-1]))


def test_aggregate_spoof_score_mean_and_median():
    windows = _windows([0.2, 0.8])
    assert aggregate_spoof_score(windows, "mean") == pytest.approx(0.5)
    assert aggregate_spoof_score(windows, "median") == pytest.approx(0.5)


def test_aggregate_spoof_score_majority_vote():
    windows = _windows([0.9, 0.9, 0.1])  # 2 of 3 windows vote spoof (>= 0.5)
    assert aggregate_spoof_score(windows, "majority_vote") == pytest.approx(2 / 3)


def test_aggregate_spoof_score_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        aggregate_spoof_score(_windows([0.5]), "not_a_strategy")


def test_high_disagreement_clip_has_high_std():
    mixed = compute_clip_diagnostics(_windows([0.0, 1.0, 0.0, 1.0]))
    unanimous = compute_clip_diagnostics(_windows([1.0, 1.0, 1.0, 1.0]))
    assert mixed.std_spoof > unanimous.std_spoof
    assert unanimous.std_spoof == pytest.approx(0.0)
