"""Run the EXISTING production inference pipeline (sara_wav2vec2, unchanged
windowing/aggregation) over the labeled calibration set downloaded by
`scripts/download_calibration_data.py`, then evaluate candidate
calibration strategies (threshold, aggregation, VAD filtering, probability
calibration, inconclusive zone) against it.

This script does NOT change production prediction logic. It is a
research/evaluation tool for docs/inference_calibration.md.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_calibration.py

Writes:
    results/metrics/calibration_baseline.json
    results/metrics/calibration_thresholds.json
    results/metrics/aggregation_comparison.json
    results/metrics/vad_comparison.json
    results/figures/calibration_roc.png
    results/figures/score_distributions.png
    results/figures/calibration_confusion_matrix.png

These outputs are gitignored (results/metrics/*, results/figures/*) per
existing repository policy — see the "Reproducibility" section of
docs/inference_calibration.md for how to regenerate them.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audio_deepfake_detector.evaluation.diagnostics import (
    AGGREGATION_STRATEGIES,
    aggregate_spoof_score,
    compute_clip_diagnostics,
)
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
from audio_deepfake_detector.models.candidate_b import HOP_SAMPLES, WINDOW_SAMPLES
from audio_deepfake_detector.models.registry import create_detector
from audio_deepfake_detector.preprocessing.audio_loader import load_audio_file
from audio_deepfake_detector.preprocessing.vad import speech_ratio
from audio_deepfake_detector.preprocessing.windowing import make_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = REPO_ROOT / "data" / "raw" / "calibration"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
MODEL_ID = "sara_wav2vec2"
VAD_CANDIDATE_RATIOS = (0.0, 0.25, 0.5, 0.75)


def run_inference_over_manifest(manifest: list[dict]) -> list[dict]:
    """Runs the real sara_wav2vec2 model once (loaded once, reused) over
    every calibration file, recording raw per-window scores plus VAD
    speech_ratio per window computed on the SAME window boundaries the
    model used (WINDOW_SAMPLES/HOP_SAMPLES from models/candidate_b.py)."""
    detector = create_detector(MODEL_ID, device="cpu")
    detector.load()
    records: list[dict] = []
    t0 = time.time()
    try:
        for i, entry in enumerate(manifest):
            file_path = CALIBRATION_DIR / entry["path"]
            audio_sample = load_audio_file(file_path)
            result = detector.predict(audio_sample)

            windows = make_windows(audio_sample.waveform, WINDOW_SAMPLES, HOP_SAMPLES)
            window_speech_ratios = [float(speech_ratio(w)) for w in windows]

            diagnostics = compute_clip_diagnostics(result.window_predictions)
            record = {
                "path": entry["path"],
                "label": entry["label"],
                "domain": entry["domain"],
                "dataset": entry["dataset"],
                "duration_seconds": audio_sample.duration_seconds,
                "windows_analyzed": result.windows_analyzed,
                "mean_spoof_score": result.probabilities["spoof"],
                "aggregation_scores": {
                    strategy: aggregate_spoof_score(result.window_predictions, strategy)
                    for strategy in AGGREGATION_STRATEGIES
                },
                "diagnostics": asdict(diagnostics),
                "window_speech_ratios": window_speech_ratios,
                "clip_mean_speech_ratio": float(np.mean(window_speech_ratios)) if window_speech_ratios else 0.0,
                "window_spoof_scores": [w.probabilities["spoof"] for w in result.window_predictions],
            }
            records.append(record)
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{len(manifest)} clips processed ({elapsed:.1f}s elapsed)")
    finally:
        detector.unload()
    print(f"Inference over {len(records)} clips took {time.time() - t0:.1f}s total")
    return records


def y_true_from_records(records: list[dict]) -> np.ndarray:
    return np.array([1 if r["label"] == "spoof" else 0 for r in records], dtype=np.int64)


def vad_filtered_aggregate(record: dict, min_speech_ratio: float, strategy: str = "mean") -> float:
    """Recompute the aggregation score keeping only windows whose speech_ratio
    >= min_speech_ratio. Falls back to using ALL windows if filtering would
    discard every window in the clip (documented fallback, not a silent
    drop of the clip)."""
    ratios = record["window_speech_ratios"]
    scores = record["window_spoof_scores"]
    kept = [s for s, r in zip(scores, ratios) if r >= min_speech_ratio]
    if not kept:
        kept = scores
    kept_arr = np.array(kept, dtype=np.float64)
    if strategy == "mean":
        return float(np.mean(kept_arr))
    if strategy == "median":
        return float(np.median(kept_arr))
    raise ValueError(strategy)


def build_inconclusive_zone(y_true: np.ndarray, y_scores: np.ndarray, reliability_target: float = 0.95) -> dict:
    """Derive an evidence-based [low, high] inconclusive band from
    calibration data: the largest bonafide-side threshold `low` such that
    P(bonafide | score <= low) >= reliability_target, and the smallest
    spoof-side threshold `high` such that P(spoof | score >= high) >=
    reliability_target. Scores between them are "inconclusive"."""
    order = np.argsort(y_scores)
    sorted_scores = y_scores[order]
    sorted_true = y_true[order]  # 1 = spoof

    low = None
    for i in range(len(sorted_scores), 0, -1):
        cum_true = sorted_true[:i]
        if len(cum_true) == 0:
            continue
        bonafide_reliability = float(np.mean(cum_true == 0))
        if bonafide_reliability >= reliability_target:
            low = float(sorted_scores[i - 1])
            break

    high = None
    for i in range(len(sorted_scores)):
        cum_true = sorted_true[i:]
        if len(cum_true) == 0:
            continue
        spoof_reliability = float(np.mean(cum_true == 1))
        if spoof_reliability >= reliability_target:
            high = float(sorted_scores[i])
            break

    result = {"reliability_target": reliability_target, "low_threshold": low, "high_threshold": high}

    if low is not None and high is not None and high > low:
        accepted_mask = (y_scores <= low) | (y_scores >= high)
        n_inconclusive = int(np.sum(~accepted_mask))
        coverage = float(np.mean(accepted_mask))
        if np.any(accepted_mask):
            accepted_pred = (y_scores[accepted_mask] >= 0.5).astype(np.int64)
            accepted_true = y_true[accepted_mask]
            accepted_accuracy = float(np.mean(accepted_pred == accepted_true))
        else:
            accepted_accuracy = None
        result.update(
            {
                "coverage": coverage,
                "accuracy_on_accepted": accepted_accuracy,
                "proportion_inconclusive": float(n_inconclusive / len(y_scores)),
                "valid_band": True,
            }
        )
    else:
        result.update(
            {
                "coverage": None,
                "accuracy_on_accepted": None,
                "proportion_inconclusive": None,
                "valid_band": False,
                "note": "No valid non-overlapping band found at this reliability target on calibration data.",
            }
        )
    return result


def make_figures(records: list[dict], y_true: np.ndarray, y_scores: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, confusion_matrix

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_scores, ax=ax, name="sara_wav2vec2 (mean agg.)")
    ax.set_title("Calibration set ROC — combined in-domain + out-of-domain")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_roc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bonafide_scores = y_scores[y_true == 0]
    spoof_scores = y_scores[y_true == 1]
    bins = np.linspace(0, 1, 30)
    ax.hist(bonafide_scores, bins=bins, alpha=0.6, label="bonafide (ground truth)", color="#2a9d8f")
    ax.hist(spoof_scores, bins=bins, alpha=0.6, label="spoof (ground truth)", color="#e76f51")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="0.5 (current deployed threshold)")
    ax.set_xlabel("Mean spoof score (clip-level, mean aggregation)")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution by ground-truth label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_distributions.png", dpi=150)
    plt.close(fig)

    y_pred = (y_scores >= 0.5).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["bonafide", "spoof"])
    ax.set_yticklabels(["bonafide", "spoof"])
    ax.set_xlabel("Predicted (threshold=0.5)")
    ax.set_ylabel("Ground truth")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_title("Confusion matrix — baseline 0.5 threshold, mean agg.")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    manifest = json.loads((CALIBRATION_DIR / "manifest.json").read_text(encoding="utf-8"))
    print(f"Loaded manifest with {len(manifest)} calibration samples")

    records = run_inference_over_manifest(manifest)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "calibration_baseline.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    y_true = y_true_from_records(records)

    # --- Step 8/9: baseline (mean aggregation) + threshold calibration ---
    mean_scores = np.array([r["mean_spoof_score"] for r in records])
    baseline_0_5 = compute_threshold_metrics(y_true, mean_scores, 0.5)
    eer, eer_threshold = compute_eer(y_true, mean_scores)
    roc_auc = compute_roc_auc(y_true, mean_scores)
    balanced_acc_best = best_balanced_accuracy_threshold(y_true, mean_scores)
    f1_best = best_f1_threshold(y_true, mean_scores)
    fpr5 = max_fpr_threshold(y_true, mean_scores, max_fpr=0.05)

    def domain_fpr(domain: str) -> dict:
        idx = [i for i, r in enumerate(records) if r["domain"] == domain and r["label"] == "bonafide"]
        if not idx:
            return {"n": 0, "fpr_at_0_5": None}
        sub_scores = mean_scores[idx]
        fp = int(np.sum(sub_scores >= 0.5))
        return {"n": len(idx), "fpr_at_0_5": float(fp / len(idx))}

    thresholds_result = {
        "baseline_0_5_threshold": baseline_0_5.to_dict(),
        "baseline_fpr_by_domain": {
            "in_domain_bonafide": domain_fpr("in_domain"),
            "out_of_domain_bonafide": domain_fpr("out_of_domain"),
        },
        "eer": eer,
        "eer_threshold": eer_threshold,
        "roc_auc": roc_auc,
        "candidates": {
            "eer_threshold": compute_threshold_metrics(y_true, mean_scores, eer_threshold).to_dict(),
            "best_balanced_accuracy": balanced_acc_best.to_dict(),
            "best_f1": f1_best.to_dict(),
            "fpr_leq_5pct": fpr5.to_dict() if fpr5 is not None else None,
        },
    }
    (METRICS_DIR / "calibration_thresholds.json").write_text(json.dumps(thresholds_result, indent=2), encoding="utf-8")

    # --- Step 11: aggregation comparison ---
    aggregation_result = {}
    for strategy in AGGREGATION_STRATEGIES:
        scores = np.array([r["aggregation_scores"][strategy] for r in records])
        s_eer, s_eer_thr = compute_eer(y_true, scores)
        aggregation_result[strategy] = {
            "eer": s_eer,
            "eer_threshold": s_eer_thr,
            "roc_auc": compute_roc_auc(y_true, scores),
            "metrics_at_0_5": compute_threshold_metrics(y_true, scores, 0.5).to_dict(),
            "metrics_at_own_eer_threshold": compute_threshold_metrics(y_true, scores, s_eer_thr).to_dict(),
        }
    (METRICS_DIR / "aggregation_comparison.json").write_text(json.dumps(aggregation_result, indent=2), encoding="utf-8")

    # --- Step 12: VAD experiment ---
    vad_result = {}
    for ratio in VAD_CANDIDATE_RATIOS:
        scores = np.array([vad_filtered_aggregate(r, ratio, "mean") for r in records])
        v_eer, v_eer_thr = compute_eer(y_true, scores)
        vad_result[f"min_speech_ratio_{ratio}"] = {
            "eer": v_eer,
            "eer_threshold": v_eer_thr,
            "roc_auc": compute_roc_auc(y_true, scores),
            "metrics_at_0_5": compute_threshold_metrics(y_true, scores, 0.5).to_dict(),
            "metrics_at_own_eer_threshold": compute_threshold_metrics(y_true, scores, v_eer_thr).to_dict(),
        }
    (METRICS_DIR / "vad_comparison.json").write_text(json.dumps(vad_result, indent=2), encoding="utf-8")

    # --- Step 10: probability calibration (temperature scaling) ---
    logits = prob_to_logit(mean_scores)
    temperature = fit_temperature_scaling(logits, y_true)
    calibrated_scores = apply_temperature(logits, temperature)
    calibration_result = {
        "temperature": temperature,
        "brier_before": brier_score(y_true, mean_scores),
        "brier_after": brier_score(y_true, calibrated_scores),
        "ece_before": expected_calibration_error(y_true, mean_scores),
        "ece_after": expected_calibration_error(y_true, calibrated_scores),
    }

    # --- Step 14: inconclusive zone ---
    inconclusive_result = build_inconclusive_zone(y_true, mean_scores, reliability_target=0.95)

    combined_extra = {
        "probability_calibration": calibration_result,
        "inconclusive_zone": inconclusive_result,
        "n_samples": len(records),
        "n_bonafide": int(np.sum(y_true == 0)),
        "n_spoof": int(np.sum(y_true == 1)),
    }
    (METRICS_DIR / "calibration_extra.json").write_text(json.dumps(combined_extra, indent=2), encoding="utf-8")

    make_figures(records, y_true, mean_scores)

    print("\n=== SUMMARY ===")
    print(f"n_samples={len(records)}  n_bonafide={int(np.sum(y_true == 0))}  n_spoof={int(np.sum(y_true == 1))}")
    print(f"baseline@0.5 accuracy={baseline_0_5.accuracy:.3f} fpr={baseline_0_5.fpr:.3f} f1={baseline_0_5.f1:.3f}")
    print(f"EER={eer:.4f} at threshold={eer_threshold:.4f}  ROC-AUC={roc_auc:.4f}")
    print(f"temperature={temperature:.3f}  brier {calibration_result['brier_before']:.4f} -> {calibration_result['brier_after']:.4f}")
    print("Wrote results/metrics/*.json and results/figures/*.png")


if __name__ == "__main__":
    main()
