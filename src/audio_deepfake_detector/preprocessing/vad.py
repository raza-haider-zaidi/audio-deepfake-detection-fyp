"""Lightweight voice-activity detection (VAD) for 16 kHz mono audio.

This module answers one narrow question: "how much of this audio region
looks like speech?" It is built on WebRTC's VAD (via the `webrtcvad-wheels`
package — a maintained wheel-only fork of Google's `py-webrtcvad` that
ships prebuilt wheels for modern Python/OS combinations, avoiding a C
compiler requirement on Streamlit Community Cloud's Debian runtime and on
Windows dev machines).

IMPORTANT: WebRTC VAD is a signal-level speech/non-speech classifier. It is
NOT a deepfake/AI-voice detector and must never be presented to users as
evidence of authenticity. It is used here only to measure how much of a
window is speech-like, so that non-speech regions (silence, music,
transitions) can be identified and studied as a possible source of the
deepfake detector's false positives (see docs/inference_calibration.md).

`webrtcvad-wheels` is a research/calibration-only dependency (see the
`calibration` extra in pyproject.toml) — it is not part of the Streamlit
deployment's requirements.txt because the production app does not use VAD.
"""

from __future__ import annotations

import numpy as np
import webrtcvad

SAMPLE_RATE = 16000
VALID_FRAME_MS = (10, 20, 30)
DEFAULT_FRAME_MS = 30
DEFAULT_AGGRESSIVENESS = 2  # 0 (least aggressive) .. 3 (most aggressive)


def _frame_samples(frame_ms: int) -> int:
    return SAMPLE_RATE * frame_ms // 1000


def _float_to_pcm16_bytes(waveform: np.ndarray) -> bytes:
    """Convert a float32 [-1, 1] waveform to 16-bit PCM bytes for WebRTC VAD."""
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def frame_speech_flags(
    waveform: np.ndarray,
    frame_ms: int = DEFAULT_FRAME_MS,
    aggressiveness: int = DEFAULT_AGGRESSIVENESS,
) -> list[bool]:
    """Classify each fixed-size frame of `waveform` as speech or non-speech.

    `waveform` must be 16 kHz mono float32. Trailing samples that don't fill
    a complete frame are dropped (WebRTC VAD requires exact 10/20/30 ms
    frames). Returns one bool per complete frame; an all-silent or
    shorter-than-one-frame input returns an empty list.
    """
    if waveform.ndim != 1:
        raise ValueError(f"frame_speech_flags expects a 1-D waveform; got shape {waveform.shape}")
    if frame_ms not in VALID_FRAME_MS:
        raise ValueError(f"frame_ms must be one of {VALID_FRAME_MS}; got {frame_ms}")
    if not (0 <= aggressiveness <= 3):
        raise ValueError(f"aggressiveness must be in [0, 3]; got {aggressiveness}")

    frame_len = _frame_samples(frame_ms)
    n_frames = waveform.shape[0] // frame_len
    if n_frames == 0:
        return []

    vad = webrtcvad.Vad(aggressiveness)
    pcm_bytes = _float_to_pcm16_bytes(waveform[: n_frames * frame_len])
    frame_byte_len = frame_len * 2  # 16-bit samples

    flags: list[bool] = []
    for i in range(n_frames):
        start = i * frame_byte_len
        frame_bytes = pcm_bytes[start : start + frame_byte_len]
        flags.append(vad.is_speech(frame_bytes, SAMPLE_RATE))
    return flags


def speech_ratio(
    waveform: np.ndarray,
    frame_ms: int = DEFAULT_FRAME_MS,
    aggressiveness: int = DEFAULT_AGGRESSIVENESS,
) -> float:
    """Fraction of `waveform`'s complete frames classified as speech, in
    [0.0, 1.0]. Returns 0.0 for audio shorter than one frame (treated as
    "no measurable speech activity", not an error, since very short/empty
    windows are already handled by the caller)."""
    flags = frame_speech_flags(waveform, frame_ms=frame_ms, aggressiveness=aggressiveness)
    if not flags:
        return 0.0
    return sum(flags) / len(flags)


def speech_active_regions(
    waveform: np.ndarray,
    frame_ms: int = DEFAULT_FRAME_MS,
    aggressiveness: int = DEFAULT_AGGRESSIVENESS,
) -> list[tuple[float, float]]:
    """Return contiguous (start_seconds, end_seconds) regions classified as
    speech-active, merging adjacent speech frames."""
    flags = frame_speech_flags(waveform, frame_ms=frame_ms, aggressiveness=aggressiveness)
    frame_seconds = frame_ms / 1000.0

    regions: list[tuple[float, float]] = []
    region_start: int | None = None
    for i, is_speech in enumerate(flags):
        if is_speech and region_start is None:
            region_start = i
        elif not is_speech and region_start is not None:
            regions.append((region_start * frame_seconds, i * frame_seconds))
            region_start = None
    if region_start is not None:
        regions.append((region_start * frame_seconds, len(flags) * frame_seconds))
    return regions
