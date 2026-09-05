"""User-triggered robustness analysis: re-scores the currently uploaded
clip under a fixed set of degraded conditions, using the SAME degradation
methodology already used for this project's frozen robustness experiment
(results/metrics/spectra_aasist3_robustness.json, scripts/run_spectra_evaluation.py)
-- reused here, not reinvented.

This is explicitly opt-in (never runs automatically), processes
conditions sequentially, reuses the already-loaded/cached detector, and
deletes every temporary file it creates before returning. No degraded
audio is ever written outside a per-call `tempfile.TemporaryDirectory()`,
and nothing here is stored permanently. This module NEVER changes the
Spectra classifier, its threshold, or its decision logic -- it only calls
the existing, unchanged `detector.predict()` once per condition.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_deepfake_detector.models.base import BaseDeepfakeDetector
from audio_deepfake_detector.utils.datatypes import AudioSample

CONDITIONS = ("original", "mp3_128k", "mp3_64k", "telephone_300_3400hz", "noise_20db", "noise_10db")
CONDITION_LABELS = {
    "original": "Original",
    "mp3_128k": "MP3 128 kbps",
    "mp3_64k": "MP3 64 kbps",
    "telephone_300_3400hz": "Telephone band (300–3400 Hz)",
    "noise_20db": "+20 dB SNR noise",
    "noise_10db": "+10 dB SNR noise",
}
NOISE_SEED = 9  # matches scripts/run_spectra_evaluation.py's fixed seed for reproducibility


def _to_mp3_and_back(waveform: np.ndarray, sr: int, bitrate_kbps: int, ffmpeg: str) -> np.ndarray:
    """Identical methodology to scripts/run_spectra_evaluation.py's
    `_to_mp3_and_back` (verified against that file before reuse)."""
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
    # TemporaryDirectory's __exit__ removes wav_path/mp3_path/wav_path2 --
    # no degraded audio persists after this function returns.


def _telephone_filter(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Identical methodology to scripts/run_spectra_evaluation.py's
    `_telephone_filter`."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, waveform).astype(np.float32)


def _add_noise_at_snr(waveform: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Identical methodology to scripts/run_spectra_evaluation.py's
    `_add_noise_at_snr`."""
    signal_power = np.mean(waveform.astype(np.float64) ** 2)
    if signal_power <= 0:
        return waveform
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return (waveform + noise).astype(np.float32)


def make_condition_waveform(condition: str, waveform: np.ndarray, sample_rate: int, ffmpeg: str) -> np.ndarray:
    if condition == "original":
        return waveform
    if condition == "mp3_128k":
        return _to_mp3_and_back(waveform, sample_rate, 128, ffmpeg)
    if condition == "mp3_64k":
        return _to_mp3_and_back(waveform, sample_rate, 64, ffmpeg)
    if condition == "telephone_300_3400hz":
        return _telephone_filter(waveform, sample_rate)
    if condition == "noise_20db":
        rng = np.random.default_rng(NOISE_SEED)
        return _add_noise_at_snr(waveform, 20.0, rng)
    if condition == "noise_10db":
        rng = np.random.default_rng(NOISE_SEED)
        return _add_noise_at_snr(waveform, 10.0, rng)
    raise ValueError(f"Unknown robustness condition: {condition}")


@dataclass
class RobustnessResult:
    condition: str
    label: str
    spoof_probability: float
    presentation_state: str
    delta_pp_from_original: float | None  # percentage points vs. the "original" row
    processing_time_ms: float = 0.0


def ffmpeg_available() -> str | None:
    """Returns the resolved ffmpeg path, or None if not found -- MP3
    conditions are skipped gracefully (not silently faked) if ffmpeg is
    unavailable in the deployment environment."""
    return shutil.which("ffmpeg")


def run_robustness_analysis(
    detector: BaseDeepfakeDetector,
    audio_sample: AudioSample,
    threshold: float,
    conditions: tuple[str, ...] = CONDITIONS,
    on_condition_start=None,
) -> list[RobustnessResult]:
    """Runs `conditions` SEQUENTIALLY against the already-loaded/cached
    `detector`. Each degraded waveform lives only in memory / a
    TemporaryDirectory for the duration of one condition's MP3 round-trip,
    and is discarded immediately after scoring. Presentation state uses
    the SAME frozen `presentation_state()` rule as production -- this
    function never alters the classifier or its threshold.

    `on_condition_start(condition_key, label)`, if given, is called before
    each condition begins -- used by the UI to show a real ("Testing MP3
    128 kbps...") status, never a fabricated progress percentage."""
    from audio_deepfake_detector.models.candidate_e import presentation_state as compute_presentation_state

    ffmpeg = ffmpeg_available()
    results: list[RobustnessResult] = []
    original_spoof_prob: float | None = None

    import time

    for condition in conditions:
        if condition in ("mp3_128k", "mp3_64k") and ffmpeg is None:
            continue  # skip gracefully rather than fabricate a result
        if on_condition_start is not None:
            on_condition_start(condition, CONDITION_LABELS[condition])
        t0 = time.perf_counter()
        degraded_waveform = make_condition_waveform(condition, audio_sample.waveform, audio_sample.sample_rate, ffmpeg or "")
        degraded_sample = AudioSample(
            waveform=degraded_waveform,
            sample_rate=audio_sample.sample_rate,
            duration_seconds=len(degraded_waveform) / audio_sample.sample_rate,
            source_name=f"{audio_sample.source_name}::{condition}",
        )
        prediction = detector.predict(degraded_sample)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        spoof_prob = prediction.probabilities["spoof"]
        if condition == "original":
            original_spoof_prob = spoof_prob
        delta = None if original_spoof_prob is None else (spoof_prob - original_spoof_prob) * 100
        results.append(
            RobustnessResult(
                condition=condition,
                label=CONDITION_LABELS[condition],
                spoof_probability=spoof_prob,
                presentation_state=compute_presentation_state(spoof_prob, threshold),
                delta_pp_from_original=delta,
                processing_time_ms=elapsed_ms,
            )
        )
        del degraded_waveform, degraded_sample  # release before the next condition

    return results


def stability_summary(results: list[RobustnessResult]) -> str:
    """'N of M degraded conditions retained the original presentation
    result' -- descriptive only."""
    if not results or results[0].condition != "original":
        return "Result stability could not be computed."
    original_state = results[0].presentation_state
    degraded = results[1:]
    if not degraded:
        return "No degraded conditions were run."
    retained = sum(1 for r in degraded if r.presentation_state == original_state)
    return f"{retained} of {len(degraded)} degraded conditions retained the original presentation result."
