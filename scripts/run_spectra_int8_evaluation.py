"""INT8 dynamic-quantization parity + calibration + frozen evaluation for
spectra_aasist3_onnx (docs/spectra_production_optimization.md).

Uses the FP32 model (spectra-aasist3.onnx) and the INT8 dynamic-quantized
model (spectra-aasist3-int8-dynamic.onnx, produced separately via
onnxruntime.quantization.quantize_dynamic) directly via onnxruntime
sessions, reusing candidate_e.py's pure preprocessing functions
(author_compatible_preprocess, softmax_spoof_bonafide) without going
through the registry (the production adapter does not yet support a local
INT8 artifact path -- see Step 16).

Reuses the EXACT SAME calibration/evaluation utterance-ID split already
frozen in results/metrics/spectra_aasist3_split.json (seed=2024) -- no new
split is created, per the instruction to reuse existing calibration
material and evaluate on the untouched 200-clip evaluation set.

Writes (gitignored):
    results/metrics/spectra_int8_parity.json
    results/metrics/spectra_int8_calibration.json
    results/metrics/spectra_int8_evaluation.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.stats import pearsonr, spearmanr

from audio_deepfake_detector.evaluation.metrics import (
    best_balanced_accuracy_threshold,
    best_f1_threshold,
    compute_eer,
    compute_roc_auc,
    compute_threshold_metrics,
    max_fpr_threshold,
)
from audio_deepfake_detector.models.candidate_e import author_compatible_preprocess, softmax_spoof_bonafide
from audio_deepfake_detector.preprocessing.audio_loader import load_audio_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "spectra_inthewild"
METRICS_DIR = REPO_ROOT / "results" / "metrics"

FP32_PATH = REPO_ROOT / "models" / "cache" / "models--lab260--Spectra-AASIST3" / "snapshots" / "bc0ded888080ddad493177bb53aa6f5b95219d7c" / "spectra-aasist3.onnx"
INT8_PATH = REPO_ROOT / "models" / "cache" / "quantized" / "spectra-aasist3-int8-dynamic.onnx"

FP32_EER_THRESHOLD = 0.9299831390380859  # frozen in spectra_fp32_baseline.json


def load_sessions():
    fp32 = ort.InferenceSession(str(FP32_PATH), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(INT8_PATH), providers=["CPUExecutionProvider"])
    return fp32, int8


def score_clip(session: ort.InferenceSession, waveform: np.ndarray) -> tuple[float, float]:
    """Returns (bonafide_logit, spoof_prob)."""
    window = author_compatible_preprocess(waveform)
    (logits,) = session.run(["logits"], {"wav": window[None, :].astype(np.float32)})
    logits = logits[0]
    spoof_prob, bonafide_prob = softmax_spoof_bonafide(logits)
    return float(logits[1]), spoof_prob


def load_manifest_by_id() -> dict[str, dict]:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {m["utterance_id"]: m for m in manifest}


def load_split() -> dict:
    return json.loads((METRICS_DIR / "spectra_aasist3_split.json").read_text(encoding="utf-8"))


def y_true_for_ids(ids: list[str], by_id: dict) -> np.ndarray:
    return np.array([1 if by_id[i]["label"] == "spoof" else 0 for i in ids], dtype=np.int64)


def full_metrics_report(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    m = compute_threshold_metrics(y_true, scores, threshold)
    eer, eer_thr = compute_eer(y_true, scores)
    return {
        "threshold_used": threshold,
        "accuracy": m.accuracy,
        "balanced_accuracy": m.balanced_accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "roc_auc": compute_roc_auc(y_true, scores),
        "eer": eer,
        "eer_threshold_on_this_set": eer_thr,
        "bonafide_fpr": m.fpr,
        "spoof_fnr": m.fnr,
        "confusion_matrix": {
            "tp_spoof_correct": int(np.sum((scores >= threshold) & (y_true == 1))),
            "tn_bonafide_correct": int(np.sum((scores < threshold) & (y_true == 0))),
            "fp_bonafide_as_spoof": int(np.sum((scores >= threshold) & (y_true == 0))),
            "fn_spoof_as_bonafide": int(np.sum((scores < threshold) & (y_true == 1))),
        },
        "n_samples": int(len(y_true)),
    }


def main() -> None:
    fp32_sess, int8_sess = load_sessions()
    split = load_split()
    by_id = load_manifest_by_id()

    calibration_ids = split["calibration_ids"]
    evaluation_ids = split["evaluation_ids"]

    # --- Step 6: parity on 50 bonafide + 50 spoof from CALIBRATION only ---
    calib_bonafide = sorted([i for i in calibration_ids if by_id[i]["label"] == "bonafide"])[:50]
    calib_spoof = sorted([i for i in calibration_ids if by_id[i]["label"] == "spoof"])[:50]
    parity_ids = calib_bonafide + calib_spoof
    print(f"Parity subset: {len(calib_bonafide)} bonafide + {len(calib_spoof)} spoof = {len(parity_ids)}")

    fp32_logits, int8_logits = [], []
    fp32_spoof, int8_spoof = [], []
    t0 = time.time()
    for i, uid in enumerate(parity_ids):
        wav_path = DATA_DIR / by_id[uid]["path"]
        sample = load_audio_file(wav_path)
        fl, fs = score_clip(fp32_sess, sample.waveform)
        il, is_ = score_clip(int8_sess, sample.waveform)
        fp32_logits.append(fl)
        int8_logits.append(il)
        fp32_spoof.append(fs)
        int8_spoof.append(is_)
        if (i + 1) % 25 == 0:
            print(f"  parity {i+1}/{len(parity_ids)} ({time.time()-t0:.1f}s)")

    fp32_logits = np.array(fp32_logits)
    int8_logits = np.array(int8_logits)
    fp32_spoof = np.array(fp32_spoof)
    int8_spoof = np.array(int8_spoof)
    abs_diff = np.abs(fp32_logits - int8_logits)
    pearson_r, _ = pearsonr(fp32_logits, int8_logits)
    spearman_r, _ = spearmanr(fp32_logits, int8_logits)

    # Decision agreement at the FP32 calibration EER threshold (spoof-score convention)
    fp32_decision = fp32_spoof >= FP32_EER_THRESHOLD
    int8_decision_same_threshold = int8_spoof >= FP32_EER_THRESHOLD
    agreement = float(np.mean(fp32_decision == int8_decision_same_threshold))

    parity_result = {
        "n_parity_clips": len(parity_ids),
        "n_bonafide": len(calib_bonafide),
        "n_spoof": len(calib_spoof),
        "mean_absolute_logit_diff": float(np.mean(abs_diff)),
        "median_absolute_logit_diff": float(np.median(abs_diff)),
        "max_absolute_logit_diff": float(np.max(abs_diff)),
        "pearson_correlation": float(pearson_r),
        "spearman_rank_correlation": float(spearman_r),
        "decision_agreement_at_fp32_threshold": agreement,
        "fp32_threshold_used_for_agreement": FP32_EER_THRESHOLD,
    }
    (METRICS_DIR / "spectra_int8_parity.json").write_text(json.dumps(parity_result, indent=2), encoding="utf-8")
    print(f"Parity: MAE={parity_result['mean_absolute_logit_diff']:.4f} pearson={pearson_r:.4f} agreement={agreement:.3f}")

    # --- Step 7: INT8 calibration on the FULL 200-clip calibration set ---
    print("Scoring full calibration set with INT8...")
    calib_spoof_scores = []
    t0 = time.time()
    for i, uid in enumerate(calibration_ids):
        wav_path = DATA_DIR / by_id[uid]["path"]
        sample = load_audio_file(wav_path)
        _, spoof_prob = score_clip(int8_sess, sample.waveform)
        calib_spoof_scores.append(spoof_prob)
        if (i + 1) % 50 == 0:
            print(f"  calib {i+1}/{len(calibration_ids)} ({time.time()-t0:.1f}s)")
    calib_spoof_scores = np.array(calib_spoof_scores)
    y_calib = y_true_for_ids(calibration_ids, by_id)

    eer, eer_thr = compute_eer(y_calib, calib_spoof_scores)
    bal_acc = best_balanced_accuracy_threshold(y_calib, calib_spoof_scores)
    f1_best = best_f1_threshold(y_calib, calib_spoof_scores)
    low_fpr = max_fpr_threshold(y_calib, calib_spoof_scores, max_fpr=0.05)
    int8_calibration_result = {
        "n_calibration": len(calibration_ids),
        "eer": eer,
        "eer_threshold": eer_thr,
        "best_balanced_accuracy_threshold": bal_acc.to_dict(),
        "best_f1_threshold": f1_best.to_dict(),
        "low_fpr_5pct_threshold": low_fpr.to_dict() if low_fpr else None,
    }
    (METRICS_DIR / "spectra_int8_calibration.json").write_text(json.dumps(int8_calibration_result, indent=2), encoding="utf-8")
    print(f"INT8 calibration EER={eer:.4f} threshold={eer_thr:.4f}")

    # --- Step 9: frozen INT8 evaluation on the SAME 200 evaluation IDs as FP32 ---
    print("Scoring full evaluation set with INT8...")
    eval_spoof_scores = []
    t0 = time.time()
    for i, uid in enumerate(evaluation_ids):
        wav_path = DATA_DIR / by_id[uid]["path"]
        sample = load_audio_file(wav_path)
        _, spoof_prob = score_clip(int8_sess, sample.waveform)
        eval_spoof_scores.append(spoof_prob)
        if (i + 1) % 50 == 0:
            print(f"  eval {i+1}/{len(evaluation_ids)} ({time.time()-t0:.1f}s)")
    eval_spoof_scores = np.array(eval_spoof_scores)
    y_eval = y_true_for_ids(evaluation_ids, by_id)

    eval_report = full_metrics_report(y_eval, eval_spoof_scores, threshold=eer_thr)
    (METRICS_DIR / "spectra_int8_evaluation.json").write_text(
        json.dumps({"threshold_source": "int8_calibration_eer_threshold", **eval_report}, indent=2), encoding="utf-8"
    )
    print(f"INT8 evaluation @ frozen threshold: acc={eval_report['accuracy']:.3f} bonafide_fpr={eval_report['bonafide_fpr']:.3f}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
