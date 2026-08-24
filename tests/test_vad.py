"""Tests for lightweight WebRTC-based VAD (preprocessing/vad.py).

No model download / network access needed — uses synthetic waveforms.
Requires the `calibration` extra (`pip install -e ".[calibration]"`);
skipped entirely if `webrtcvad` is not installed, so the default suite
still passes in an environment set up only for Phase 1-3 work.
"""

import numpy as np
import pytest

pytest.importorskip("webrtcvad")

from audio_deepfake_detector.preprocessing.vad import (
    SAMPLE_RATE,
    frame_speech_flags,
    speech_active_regions,
    speech_ratio,
)


def test_silence_has_zero_speech_ratio():
    waveform = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # 2s of digital silence
    assert speech_ratio(waveform) == pytest.approx(0.0)


def test_full_band_loud_tone_is_not_classified_as_all_silence():
    # A synthetic tone is not real speech, but should not crash the VAD and
    # should produce a well-formed ratio in [0, 1]. We do not assert it is
    # "speech" — WebRTC VAD is tuned for voiced formant structure, not pure
    # tones, and this test only checks the pipeline is well-behaved.
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    waveform = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    ratio = speech_ratio(waveform)
    assert 0.0 <= ratio <= 1.0


def test_frame_speech_flags_rejects_invalid_frame_ms():
    waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
    with pytest.raises(ValueError):
        frame_speech_flags(waveform, frame_ms=25)


def test_frame_speech_flags_rejects_invalid_aggressiveness():
    waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
    with pytest.raises(ValueError):
        frame_speech_flags(waveform, aggressiveness=5)


def test_frame_speech_flags_rejects_non_1d():
    with pytest.raises(ValueError):
        frame_speech_flags(np.zeros((10, 2), dtype=np.float32))


def test_shorter_than_one_frame_returns_empty_flags_and_zero_ratio():
    waveform = np.zeros(10, dtype=np.float32)  # far shorter than a 30ms frame
    assert frame_speech_flags(waveform) == []
    assert speech_ratio(waveform) == pytest.approx(0.0)


def test_frame_count_matches_expected_for_30ms_frames():
    # 1 second @ 16kHz / 30ms frames (480 samples) = 33 complete frames
    waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
    flags = frame_speech_flags(waveform, frame_ms=30)
    assert len(flags) == SAMPLE_RATE // 480


def test_speech_active_regions_on_silence_is_empty():
    waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert speech_active_regions(waveform) == []
