"""Tests for app/validation.py — upload constraints, no model needed."""

import numpy as np
import pytest
import soundfile as sf

from app.errors import UserFacingError
from app.validation import MAX_DURATION_SECONDS, validate_and_load_upload


def _make_wav_bytes(duration_seconds: float, sample_rate: int = 16000) -> bytes:
    import io

    waveform = (0.1 * np.sin(np.linspace(0, 10, int(duration_seconds * sample_rate)))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_valid_short_clip_accepted():
    data = _make_wav_bytes(2.0)
    sample = validate_and_load_upload(data, "clip.wav")
    assert sample.duration_seconds == pytest.approx(2.0, abs=0.05)


def test_clip_over_max_duration_rejected():
    data = _make_wav_bytes(MAX_DURATION_SECONDS + 5)
    with pytest.raises(UserFacingError):
        validate_and_load_upload(data, "long.wav")


def test_clip_at_exactly_max_duration_accepted():
    data = _make_wav_bytes(MAX_DURATION_SECONDS)
    sample = validate_and_load_upload(data, "exact.wav")
    assert sample.duration_seconds <= MAX_DURATION_SECONDS + 0.01


def test_oversized_file_rejected():
    data = _make_wav_bytes(1.0)
    with pytest.raises(UserFacingError):
        validate_and_load_upload(data, "clip.wav", max_file_size_bytes=10)


def test_corrupt_file_rejected_with_friendly_message():
    with pytest.raises(UserFacingError) as exc_info:
        validate_and_load_upload(b"not audio data", "clip.wav")
    assert "couldn't be read" in exc_info.value.friendly_message


def test_empty_file_rejected():
    with pytest.raises(UserFacingError):
        validate_and_load_upload(b"", "clip.wav")


def test_unsupported_extension_rejected():
    data = _make_wav_bytes(1.0)
    with pytest.raises(UserFacingError):
        validate_and_load_upload(data, "clip.ogg")
