"""Tests for shared audio preprocessing (no ML models involved)."""

import numpy as np
import pytest
import soundfile as sf

from audio_deepfake_detector.preprocessing.audio_loader import (
    TARGET_SAMPLE_RATE,
    AudioLoadError,
    load_audio_bytes,
    load_audio_file,
)


def _make_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    import io

    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_load_mono_wav_at_target_rate():
    waveform = (0.1 * np.sin(np.linspace(0, 10, TARGET_SAMPLE_RATE))).astype(np.float32)
    data = _make_wav_bytes(waveform, TARGET_SAMPLE_RATE)

    sample = load_audio_bytes(data, "test.wav")

    assert sample.sample_rate == TARGET_SAMPLE_RATE
    assert sample.waveform.ndim == 1
    assert sample.waveform.dtype == np.float32
    assert abs(sample.duration_seconds - 1.0) < 0.01


def test_stereo_to_mono_conversion():
    left = (0.1 * np.sin(np.linspace(0, 10, TARGET_SAMPLE_RATE))).astype(np.float32)
    right = (0.1 * np.cos(np.linspace(0, 10, TARGET_SAMPLE_RATE))).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    data = _make_wav_bytes(stereo, TARGET_SAMPLE_RATE)

    sample = load_audio_bytes(data, "stereo.wav")

    assert sample.waveform.ndim == 1
    assert sample.waveform.shape[0] == TARGET_SAMPLE_RATE


def test_resampling_from_44100_to_16000():
    orig_sr = 44100
    waveform = (0.1 * np.sin(np.linspace(0, 10, orig_sr))).astype(np.float32)
    data = _make_wav_bytes(waveform, orig_sr)

    sample = load_audio_bytes(data, "highres.wav")

    assert sample.sample_rate == TARGET_SAMPLE_RATE
    assert abs(sample.duration_seconds - 1.0) < 0.05


def test_empty_bytes_rejected():
    with pytest.raises(AudioLoadError):
        load_audio_bytes(b"", "empty.wav")


def test_corrupt_audio_rejected():
    with pytest.raises(AudioLoadError):
        load_audio_bytes(b"not a real wav file", "corrupt.wav")


def test_unsupported_extension_rejected():
    with pytest.raises(AudioLoadError):
        load_audio_bytes(b"irrelevant", "clip.ogg")


def test_missing_file_rejected():
    with pytest.raises(AudioLoadError):
        load_audio_file("/nonexistent/path/does_not_exist.wav")
