"""Shared audio loading and normalization pipeline.

Decodes WAV/MP3/FLAC locally, converts to mono float32, and resamples to
16 kHz when necessary. Deliberately does NOT crop/pad audio to a fixed
length — that is a model-specific concern handled by adapters (see
src/audio_deepfake_detector/models/), since different models impose
different window/padding rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from audio_deepfake_detector.utils.datatypes import AudioSample

TARGET_SAMPLE_RATE = 16000
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac"}


class AudioLoadError(ValueError):
    """Raised when an audio file cannot be validly decoded and normalized."""


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    """Collapse a (samples, channels) or (channels, samples) array to mono."""
    if waveform.ndim == 1:
        return waveform
    # soundfile returns (frames, channels)
    return waveform.mean(axis=1)


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    import librosa

    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


def load_audio_bytes(data: bytes, source_name: str) -> AudioSample:
    """Load audio from in-memory bytes (e.g. an uploaded file) without
    permanently storing it to disk."""
    import io

    suffix = Path(source_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise AudioLoadError(
            f"Unsupported audio format '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if not data:
        raise AudioLoadError(f"Audio source '{source_name}' is empty (0 bytes).")

    try:
        waveform, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:  # soundfile raises various backend-specific errors
        raise AudioLoadError(
            f"Could not decode audio from '{source_name}': {exc}"
        ) from exc

    return _finalize(waveform, sr, source_name)


def load_audio_file(path: str | Path) -> AudioSample:
    """Load audio from a local file path."""
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"Audio file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise AudioLoadError(
            f"Unsupported audio format '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if path.stat().st_size == 0:
        raise AudioLoadError(f"Audio file is empty (0 bytes): {path}")

    try:
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioLoadError(f"Could not decode audio file '{path}': {exc}") from exc

    return _finalize(waveform, sr, path.name)


def _finalize(waveform: np.ndarray, sr: int, source_name: str) -> AudioSample:
    if waveform.size == 0:
        raise AudioLoadError(f"Decoded audio from '{source_name}' contains no samples.")

    waveform = _to_mono(np.asarray(waveform, dtype=np.float32))
    waveform = _resample(waveform, sr, TARGET_SAMPLE_RATE)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    if not np.all(np.isfinite(waveform)):
        raise AudioLoadError(f"Decoded audio from '{source_name}' contains non-finite samples.")

    duration_seconds = float(waveform.shape[0] / TARGET_SAMPLE_RATE)

    return AudioSample(
        waveform=waveform,
        sample_rate=TARGET_SAMPLE_RATE,
        duration_seconds=duration_seconds,
        source_name=source_name,
    )
