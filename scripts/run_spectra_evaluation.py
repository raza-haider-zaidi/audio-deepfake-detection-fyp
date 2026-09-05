"""Formal evaluation of spectra_aasist3_onnx (lab260/Spectra-AASIST3) on a
balanced SpeechAntiSpoofingBenchmarks/InTheWild subset, and a same-clip
head-to-head comparison with sara_wav2vec2 (docs/spectra_aasist3_evaluation.md).

Requires data/raw/spectra_inthewild/manifest.json (produced by
scripts/download_spectra_inthewild_subset.py) -- 400 clips, 200 bonafide +
200 spoof, exact utterance IDs recorded in that manifest.

Writes (gitignored, results/metrics/* / results/figures/*):
    results/metrics/spectra_aasist3_split.json
    results/metrics/spectra_aasist3_calibration.json
    results/metrics/spectra_aasist3_evaluation.json
    results/metrics/spectra_aasist3_sara_comparison.json
    results/metrics/spectra_aasist3_full_clip_aggregation.json
    results/metrics/spectra_aasist3_robustness.json

RESEARCH-ONLY. No production inference logic is modified.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_deepfake_detector.evaluation.metrics import (
    best_balanced_accuracy_threshold,
    best_f1_threshold,
    compute_eer,
    compute_roc_auc,
    compute_threshold_metrics,
    max_fpr_threshold,
)
from audio_deepfake_detector.models.registry import create_detector
from audio_deepfake_detector.preprocessing.audio_loader import load_audio_file
from audio_deepfake_detector.utils.datatypes import AudioSample

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "spectra_inthewild"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
SPLIT_SEED = 2024


def load_manifest() -> list[dict]:
    return json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


def make_split(manifest: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(SPLIT_SEED)
    bonafide = sorted([m for m in manifest if m["label"] == "bonafide"], key=lambda m: m["utterance_id"])
    spoof = sorted([m for m in manifest if m["label"] == "spoof"], key=lambda m: m["utterance_id"])
    rng.shuffle(bonafide)
    rng.shuffle(spoof)
    calibration = bonafide[:100] + spoof[:100]
    evaluation = bonafide[100:200] + spoof[100:200]
    assert len(calibration) == 200 and len(evaluation) == 200
    assert not set(m["utterance_id"] for m in calibration) & set(m["utterance_id"] for m in evaluation)
    return calibration, evaluation


def y_true_from(entries: list[dict]) -> np.ndarray:
    return np.array([1 if e["label"] == "spoof" else 0 for e in entries], dtype=np.int64)


def score_spectra_author_compatible(entries: list[dict]) -> list[dict]:
    detector = create_detector("spectra_aasist3_onnx", device="cpu")
    detector.load()
    records = []
    t0 = time.time()
    try:
        for i, e in enumerate(entries):
            sample = load_audio_file(DATA_DIR / e["path"])
            result = detector.predict(sample)
            records.append(
                {
                    **e,
                    "spoof_score": result.probabilities["spoof"],
                    "bonafide_logit": result.window_predictions[0].probabilities.get("bonafide"),
                    "windows_analyzed": result.windows_analyzed,
                }
            )
            if (i + 1) % 50 == 0:
                print(f"  spectra author-compatible: {i+1}/{len(entries)} ({time.time()-t0:.1f}s)")
    finally:
        detector.unload()
    print(f"spectra author-compatible scoring of {len(records)} clips took {time.time()-t0:.1f}s")
    return records


def score_spectra_full_clip(entries: list[dict], aggregation: str) -> list[dict]:
    detector = create_detector("spectra_aasist3_onnx", device="cpu")
    detector.load()
    records = []
    t0 = time.time()
    try:
        for i, e in enumerate(entries):
            sample = load_audio_file(DATA_DIR / e["path"])
            full = detector.predict_full_clip(sample, aggregation=aggregation)
            records.append({**e, "spoof_score": full["aggregated_spoof_prob"], "n_windows": full["n_windows"]})
            if (i + 1) % 50 == 0:
                print(f"  spectra full-clip[{aggregation}]: {i+1}/{len(entries)} ({time.time()-t0:.1f}s)")
    finally:
        detector.unload()
    return records


def score_sara(entries: list[dict]) -> list[dict]:
    detector = create_detector("sara_wav2vec2", device="cpu")
    detector.load()
    records = []
    t0 = time.time()
    try:
        for i, e in enumerate(entries):
            sample = load_audio_file(DATA_DIR / e["path"])
            result = detector.predict(sample)
            records.append({**e, "spoof_score": result.probabilities["spoof"]})
            if (i + 1) % 50 == 0:
                print(f"  sara: {i+1}/{len(entries)} ({time.time()-t0:.1f}s)")
    finally:
        detector.unload()
    print(f"sara scoring of {len(records)} clips took {time.time()-t0:.1f}s")
    return records


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
        "bonafide_fpr": m.fpr,  # bonafide (label=0) misclassified as spoof
        "spoof_fnr": m.fnr,  # spoof (label=1) misclassified as bonafide
        "confusion_matrix": {
            "tp_spoof_correct": int(np.sum((scores >= threshold) & (y_true == 1))),
            "tn_bonafide_correct": int(np.sum((scores < threshold) & (y_true == 0))),
            "fp_bonafide_as_spoof": int(np.sum((scores >= threshold) & (y_true == 0))),
            "fn_spoof_as_bonafide": int(np.sum((scores < threshold) & (y_true == 1))),
        },
        "n_samples": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Robustness (Step 19)
# ---------------------------------------------------------------------------
def _to_mp3_and_back(waveform: np.ndarray, sr: int, bitrate_kbps: int, ffmpeg: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "in.wav"
        mp3_path = Path(td) / "out.mp3"
        wav_path2 = Path(td) / "back.wav"
        sf.write(str(wav_path), waveform, sr, subtype="PCM_16")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path), "-b:a", f"{bitrate_kbps}k", str(mp3_path)],
            check=True,
        )
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3_path), str(wav_path2)], check=True)
        out, _ = sf.read(str(wav_path2), dtype="float32")
        if out.ndim > 1:
            out = out.mean(axis=1)
        return out


def _telephone_filter(waveform: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, waveform).astype(np.float32)


def _add_noise_at_snr(waveform: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(waveform.astype(np.float64) ** 2)
    if signal_power <= 0:
        return waveform
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return (waveform + noise).astype(np.float32)


def build_degraded_dataset(entries: list[dict], ffmpeg: str) -> dict[str, list[AudioSample]]:
    rng = np.random.default_rng(9)
    variants: dict[str, list[AudioSample]] = {
        "original": [],
        "mp3_128k": [],
        "mp3_64k": [],
        "telephone_300_3400hz": [],
        "noise_20db": [],
        "noise_10db": [],
    }
    for e in entries:
        sample = load_audio_file(DATA_DIR / e["path"])
        wav, sr = sample.waveform, sample.sample_rate
        def _mk(waveform, tag):
            return AudioSample(waveform=waveform, sample_rate=sr, duration_seconds=len(waveform) / sr, source_name=f"{e['utterance_id']}_{tag}")

        variants["original"].append(sample)
        variants["mp3_128k"].append(_mk(_to_mp3_and_back(wav, sr, 128, ffmpeg), "mp3_128k"))
        variants["mp3_64k"].append(_mk(_to_mp3_and_back(wav, sr, 64, ffmpeg), "mp3_64k"))
        variants["telephone_300_3400hz"].append(_mk(_telephone_filter(wav, sr), "tel"))
        variants["noise_20db"].append(_mk(_add_noise_at_snr(wav, 20.0, rng), "n20"))
        variants["noise_10db"].append(_mk(_add_noise_at_snr(wav, 10.0, rng), "n10"))
    return variants


def score_samples_spectra(samples: list[AudioSample]) -> np.ndarray:
    detector = create_detector("spectra_aasist3_onnx", device="cpu")
    detector.load()
    try:
        return np.array([detector.predict(s).probabilities["spoof"] for s in samples])
    finally:
        detector.unload()


def main() -> None:
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH; install it and ensure it is available before running this script.")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    calibration, evaluation = make_split(manifest)
    split_record = {
        "seed": SPLIT_SEED,
        "calibration_ids": sorted(e["utterance_id"] for e in calibration),
        "evaluation_ids": sorted(e["utterance_id"] for e in evaluation),
        "n_calibration": len(calibration),
        "n_evaluation": len(evaluation),
    }
    (METRICS_DIR / "spectra_aasist3_split.json").write_text(json.dumps(split_record, indent=2), encoding="utf-8")
    print(f"Split: {len(calibration)} calibration, {len(evaluation)} evaluation, no overlap confirmed")

    # --- Step 13/14: author-compatible calibration ---
    calib_records = score_spectra_author_compatible(calibration)
    y_cal = y_true_from(calib_records)
    s_cal = np.array([r["spoof_score"] for r in calib_records])

    eer, eer_thr = compute_eer(y_cal, s_cal)
    bal_acc = best_balanced_accuracy_threshold(y_cal, s_cal)
    f1_best = best_f1_threshold(y_cal, s_cal)
    low_fpr = max_fpr_threshold(y_cal, s_cal, max_fpr=0.05)
    calibration_result = {
        "eer": eer,
        "eer_threshold": eer_thr,
        "best_balanced_accuracy_threshold": bal_acc.to_dict(),
        "best_f1_threshold": f1_best.to_dict(),
        "low_fpr_5pct_threshold": low_fpr.to_dict() if low_fpr else None,
        "n_calibration": len(calib_records),
    }
    (METRICS_DIR / "spectra_aasist3_calibration.json").write_text(json.dumps(calibration_result, indent=2), encoding="utf-8")
    print(f"Calibration EER={eer:.4f} threshold={eer_thr:.4f}")

    # --- Step 15: frozen evaluation ---
    eval_records = score_spectra_author_compatible(evaluation)
    y_eval = y_true_from(eval_records)
    s_eval = np.array([r["spoof_score"] for r in eval_records])
    eval_report = full_metrics_report(y_eval, s_eval, threshold=eer_thr)
    (METRICS_DIR / "spectra_aasist3_evaluation.json").write_text(
        json.dumps({"threshold_source": "calibration_eer_threshold", **eval_report, "raw_records": eval_records}, indent=2),
        encoding="utf-8",
    )
    print(f"Evaluation @ frozen threshold: acc={eval_report['accuracy']:.3f} bonafide_fpr={eval_report['bonafide_fpr']:.3f}")

    # --- Step 16/17: Sara same-clip comparison ---
    sara_eval_records = score_sara(evaluation)
    s_sara = np.array([r["spoof_score"] for r in sara_eval_records])
    sara_report = full_metrics_report(y_eval, s_sara, threshold=0.5)
    false_positive_speakers = [
        {"utterance_id": r["utterance_id"], "speaker": r.get("speaker"), "spoof_score": r["spoof_score"], "duration_seconds": r["duration_seconds"]}
        for r, s in zip(eval_records, s_eval)
        if r["label"] == "bonafide" and s >= eer_thr
    ]
    comparison_result = {
        "spectra_eval": eval_report,
        "sara_eval_at_0_5": sara_report,
        "n_shared_evaluation_clips": len(evaluation),
        "spectra_bonafide_false_positive_clips": false_positive_speakers,
        "bonafide_score_distribution": {
            "spectra": [float(s) for r, s in zip(eval_records, s_eval) if r["label"] == "bonafide"],
            "sara": [float(s) for r, s in zip(sara_eval_records, s_sara) if r["label"] == "bonafide"],
        },
    }
    (METRICS_DIR / "spectra_aasist3_sara_comparison.json").write_text(json.dumps(comparison_result, indent=2), encoding="utf-8")
    print(f"Sara @0.5: acc={sara_report['accuracy']:.3f} bonafide_fpr={sara_report['bonafide_fpr']:.3f}")

    # --- Step 18: full-clip aggregation extension (calibrate on calibration, freeze, evaluate) ---
    agg_results = {}
    for agg in ("first", "mean", "median", "majority"):
        recs = score_spectra_full_clip(calibration, agg)
        y = y_true_from(recs)
        s = np.array([r["spoof_score"] for r in recs])
        a_eer, a_thr = compute_eer(y, s)
        agg_results[agg] = {"calibration_eer": a_eer, "calibration_eer_threshold": a_thr}
    best_agg = min(agg_results, key=lambda k: agg_results[k]["calibration_eer"])
    eval_recs_best_agg = score_spectra_full_clip(evaluation, best_agg)
    y_eval_agg = y_true_from(eval_recs_best_agg)
    s_eval_agg = np.array([r["spoof_score"] for r in eval_recs_best_agg])
    best_agg_eval_report = full_metrics_report(y_eval_agg, s_eval_agg, threshold=agg_results[best_agg]["calibration_eer_threshold"])
    full_clip_result = {
        "calibration_by_aggregation": agg_results,
        "selected_aggregation": best_agg,
        "note": "OUR OWN application-level extension (multi-window over full clip), NOT attributed to the model authors. Aggregation strategy selected on calibration data only, then frozen and evaluated once on the evaluation set.",
        "frozen_evaluation_with_best_aggregation": best_agg_eval_report,
        "author_compatible_single_window_evaluation_for_reference": eval_report,
    }
    (METRICS_DIR / "spectra_aasist3_full_clip_aggregation.json").write_text(json.dumps(full_clip_result, indent=2), encoding="utf-8")
    print(f"Best full-clip aggregation: {best_agg} (calibration EER={agg_results[best_agg]['calibration_eer']:.4f})")

    # --- Step 19: robustness on a fixed subset of evaluation clips ---
    robustness_subset = [e for e in evaluation if e["label"] == "bonafide"][:20] + [e for e in evaluation if e["label"] == "spoof"][:20]
    y_robust = y_true_from(robustness_subset)
    variants = build_degraded_dataset(robustness_subset, ffmpeg)
    robustness_result = {}
    for variant_name, samples in variants.items():
        scores = score_samples_spectra(samples)
        eer_v, _ = compute_eer(y_robust, scores)
        m = compute_threshold_metrics(y_robust, scores, eer_thr)
        robustness_result[variant_name] = {
            "eer": eer_v,
            "roc_auc": compute_roc_auc(y_robust, scores),
            "f1_at_calibration_threshold": m.f1,
            "bonafide_fpr_at_calibration_threshold": m.fpr,
        }
        print(f"  robustness[{variant_name}]: EER={eer_v:.4f}")
    (METRICS_DIR / "spectra_aasist3_robustness.json").write_text(
        json.dumps({"n_subset": len(robustness_subset), "results": robustness_result}, indent=2), encoding="utf-8"
    )

    print("\n=== DONE === Wrote results/metrics/spectra_aasist3_*.json")


if __name__ == "__main__":
    main()
