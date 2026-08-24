"""Classification / calibration metrics for the Phase 4 calibration workflow.

All functions here take arrays of ground-truth labels and *spoof* scores in
[0, 1] (higher = more spoof-like), consistent with the `spoof` entry of
`PredictionResult.probabilities`. Nothing in this module is used by the
production Streamlit app — it exists purely to evaluate candidate deployment
strategies against labeled calibration data before any such strategy is
adopted (see docs/inference_calibration.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class ThresholdMetrics:
    threshold: float
    tpr: float  # recall on spoof (positive) class
    fpr: float  # bonafide clips incorrectly flagged as spoof
    fnr: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    accuracy: float

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "tpr": self.tpr,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
        }


def _labels_to_binary(labels: np.ndarray) -> np.ndarray:
    """labels: 1 = spoof (positive class), 0 = bonafide."""
    return np.asarray(labels, dtype=np.int64)


def compute_threshold_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> ThresholdMetrics:
    y_true = _labels_to_binary(y_true)
    y_pred = (np.asarray(y_scores) >= threshold).astype(np.int64)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = 1.0 - tpr
    tnr = 1.0 - fpr
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return ThresholdMetrics(
        threshold=float(threshold),
        tpr=tpr,
        fpr=fpr,
        fnr=fnr,
        precision=precision,
        recall=recall,
        f1=f1,
        balanced_accuracy=balanced_accuracy,
        accuracy=accuracy,
    )


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and the threshold at which it occurs.

    EER = the point where the false-positive rate (bonafide misclassified
    as spoof) equals the false-negative rate (spoof misclassified as
    bonafide). Returns (eer, threshold) with eer in [0, 1].
    """
    y_true = _labels_to_binary(y_true)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    # roc_curve's first threshold is +inf by construction; clip to [0, 1]
    threshold = float(np.clip(thresholds[idx], 0.0, 1.0))
    return eer, threshold


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(roc_auc_score(_labels_to_binary(y_true), y_scores))


def sweep_thresholds(y_true: np.ndarray, y_scores: np.ndarray, n_steps: int = 199) -> list[ThresholdMetrics]:
    """Evaluate ThresholdMetrics at `n_steps` evenly spaced thresholds in
    (0, 1), used to select best-balanced-accuracy / best-F1 / FPR<=target
    candidates without assuming 0.5 a priori."""
    thresholds = np.linspace(0.0, 1.0, n_steps + 2)[1:-1]
    return [compute_threshold_metrics(y_true, y_scores, t) for t in thresholds]


def best_balanced_accuracy_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> ThresholdMetrics:
    candidates = sweep_thresholds(y_true, y_scores)
    return max(candidates, key=lambda m: m.balanced_accuracy)


def best_f1_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> ThresholdMetrics:
    candidates = sweep_thresholds(y_true, y_scores)
    return max(candidates, key=lambda m: m.f1)


def max_fpr_threshold(y_true: np.ndarray, y_scores: np.ndarray, max_fpr: float = 0.05) -> ThresholdMetrics | None:
    """Highest-recall threshold whose FPR does not exceed `max_fpr` on the
    given (calibration) data. Returns None if no threshold in the swept
    range achieves this — callers must not silently substitute another
    threshold in that case (see docs/inference_calibration.md Step 9)."""
    candidates = [m for m in sweep_thresholds(y_true, y_scores) if m.fpr <= max_fpr]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.recall)


def brier_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true = _labels_to_binary(y_true).astype(np.float64)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    return float(np.mean((y_scores - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE: sum over bins of (bin weight) * |accuracy - confidence|,
    where "confidence" is the predicted spoof probability and "accuracy" is
    the empirical spoof rate within that score bin."""
    y_true = _labels_to_binary(y_true).astype(np.float64)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_scores)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_scores >= lo) & (y_scores <= hi)
        else:
            mask = (y_scores >= lo) & (y_scores < hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(y_scores[mask]))
        bin_acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def fit_temperature_scaling(logit_spoof: np.ndarray, y_true: np.ndarray, max_iter: int = 200) -> float:
    """Fit a single scalar temperature T minimizing NLL of
    sigmoid(logit_spoof / T) against binary labels, via simple grid+refine
    search (no torch/scipy.optimize dependency required). Returns T > 0;
    T == 1.0 means calibration made no change."""
    y_true = _labels_to_binary(y_true).astype(np.float64)
    logit_spoof = np.asarray(logit_spoof, dtype=np.float64)

    def nll(t: float) -> float:
        p = 1.0 / (1.0 + np.exp(-logit_spoof / t))
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    # Coarse-to-fine grid search over T in (0.05, 10].
    best_t, best_nll = 1.0, nll(1.0)
    lo, hi = 0.05, 10.0
    for _ in range(6):
        grid = np.linspace(lo, hi, 25)
        losses = [nll(t) for t in grid]
        idx = int(np.argmin(losses))
        if losses[idx] < best_nll:
            best_t, best_nll = float(grid[idx]), float(losses[idx])
        span = (hi - lo) / 25 * 3
        lo, hi = max(0.01, grid[idx] - span), grid[idx] + span
    return best_t


def apply_temperature(logit_spoof: np.ndarray, temperature: float) -> np.ndarray:
    logit_spoof = np.asarray(logit_spoof, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logit_spoof / temperature))


def prob_to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))
