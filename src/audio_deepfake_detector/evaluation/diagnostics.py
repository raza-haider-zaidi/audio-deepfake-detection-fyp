"""Window-level score diagnostics and clip-level aggregation strategies.

This module does NOT change production prediction logic (see
`models/candidate_b.py`, which still uses `aggregate_mean_probability`
unconditionally). It is an offline/diagnostic layer used by
`scripts/run_calibration.py` and tests to inspect *why* a clip received the
spoof score it did, and to compare candidate aggregation strategies against
labeled calibration data before any production change is made.

See docs/inference_calibration.md Step 3 / Step 11.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio_deepfake_detector.utils.datatypes import WindowPrediction


@dataclass
class ClipScoreDiagnostics:
    """Descriptive statistics over a clip's per-window spoof scores."""

    n_windows: int
    mean_spoof: float
    median_spoof: float
    trimmed_mean_spoof: float
    min_spoof: float
    max_spoof: float
    std_spoof: float
    n_windows_bonafide_vote: int
    n_windows_spoof_vote: int
    proportion_spoof_vote: float


def _spoof_scores(window_predictions: list[WindowPrediction], spoof_label: str = "spoof") -> np.ndarray:
    if not window_predictions:
        raise ValueError("Cannot compute diagnostics for an empty window_predictions list")
    return np.array([w.probabilities[spoof_label] for w in window_predictions], dtype=np.float64)


def trimmed_mean(scores: np.ndarray, proportion_to_cut: float = 0.1) -> float:
    """Mean after discarding the lowest/highest `proportion_to_cut` fraction
    of scores from each tail. Falls back to the plain mean when too few
    windows exist to trim without discarding everything (<= 2 windows, or a
    cut that would remove the whole array)."""
    n = scores.shape[0]
    if n <= 2:
        return float(np.mean(scores))
    k = int(np.floor(n * proportion_to_cut))
    if k == 0:
        return float(np.mean(scores))
    if 2 * k >= n:
        return float(np.mean(scores))
    sorted_scores = np.sort(scores)
    return float(np.mean(sorted_scores[k : n - k]))


def compute_clip_diagnostics(
    window_predictions: list[WindowPrediction],
    spoof_label: str = "spoof",
    trim_proportion: float = 0.1,
    vote_threshold: float = 0.5,
) -> ClipScoreDiagnostics:
    """Compute per-clip descriptive statistics from raw per-window scores.

    `vote_threshold` decides each window's binary "vote" for the
    bonafide/spoof-window counts (independent of whatever threshold is
    ultimately chosen for clip-level decisions — see
    docs/inference_calibration.md Step 9).
    """
    scores = _spoof_scores(window_predictions, spoof_label=spoof_label)
    n_spoof_vote = int(np.sum(scores >= vote_threshold))
    n_bonafide_vote = int(scores.shape[0] - n_spoof_vote)

    return ClipScoreDiagnostics(
        n_windows=int(scores.shape[0]),
        mean_spoof=float(np.mean(scores)),
        median_spoof=float(np.median(scores)),
        trimmed_mean_spoof=trimmed_mean(scores, trim_proportion),
        min_spoof=float(np.min(scores)),
        max_spoof=float(np.max(scores)),
        std_spoof=float(np.std(scores)),
        n_windows_bonafide_vote=n_bonafide_vote,
        n_windows_spoof_vote=n_spoof_vote,
        proportion_spoof_vote=float(n_spoof_vote / scores.shape[0]),
    )


def aggregate_spoof_score(
    window_predictions: list[WindowPrediction],
    strategy: str,
    spoof_label: str = "spoof",
    trim_proportion: float = 0.1,
) -> float:
    """Compute a single clip-level spoof score using the named aggregation
    strategy. Strategies are compared empirically in Step 11 — this
    function does not itself decide which one is "correct"."""
    scores = _spoof_scores(window_predictions, spoof_label=spoof_label)

    if strategy == "mean":
        return float(np.mean(scores))
    if strategy == "median":
        return float(np.median(scores))
    if strategy == "trimmed_mean":
        return trimmed_mean(scores, trim_proportion)
    if strategy == "majority_vote":
        # Fraction of windows voting spoof at the 0.5 per-window threshold;
        # this yields a value in [0, 1] comparable to the other strategies
        # (0.5 is a genuine tie), not merely a hard 0/1 label.
        return float(np.mean(scores >= 0.5))
    raise ValueError(
        f"Unknown aggregation strategy '{strategy}'. "
        "Expected one of: mean, median, trimmed_mean, majority_vote."
    )


AGGREGATION_STRATEGIES = ("mean", "median", "trimmed_mean", "majority_vote")
