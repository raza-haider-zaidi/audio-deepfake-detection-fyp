"""Lightweight, no-ML-model audio-quality diagnostics.

Every value here is a plain signal-processing calculation on the decoded
waveform (peak amplitude, RMS, silence ratio, clipping ratio) or a simple
file-level fact (size, approximate bitrate). NONE of this is a model
prediction, and NONE of it is fed back into the Spectra-AASIST3 detector
-- see docs/analysis_platform.md, "Separation between supplementary
analysis and classifier output".

Deliberately NOT computed (per project policy -- do not invent
unmeasurable claims): microphone quality, room type, speaker identity/
authenticity, codec history, or number of prior compressions. Speech-
active proportion is likewise not computed here, since no lightweight,
already-available, reliable method for it exists in this project.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

NOT_AVAILABLE = "Not available"

SILENCE_AMPLITUDE_THRESHOLD = 0.01  # matches a common -40 dBFS-ish practical silence cutoff
CLIPPING_AMPLITUDE_THRESHOLD = 0.999  # samples this close to full-scale are considered clipped

SILENCE_RATIO_LIMITED_THRESHOLD = 0.6
CLIPPING_RATIO_LIMITED_THRESHOLD = 0.01
CLIPPING_RATIO_POOR_THRESHOLD = 0.05
SILENCE_RATIO_POOR_THRESHOLD = 0.85


@dataclass
class AudioQualityMetrics:
    peak_amplitude: float
    rms_level: float
    silence_ratio: float
    clipping_ratio: float
    approximate_bitrate_kbps: float | None


def compute_audio_quality(waveform: np.ndarray, sample_rate: int, file_size_bytes: int, duration_seconds: float) -> AudioQualityMetrics:
    """Pure signal-level diagnostics. `waveform` is assumed float32 in
    [-1, 1] (this project's existing decode convention)."""
    if waveform.size == 0:
        return AudioQualityMetrics(0.0, 0.0, 1.0, 0.0, None)

    abs_wave = np.abs(waveform)
    peak_amplitude = float(np.max(abs_wave))
    rms_level = float(np.sqrt(np.mean(np.square(waveform))))
    silence_ratio = float(np.mean(abs_wave < SILENCE_AMPLITUDE_THRESHOLD))
    clipping_ratio = float(np.mean(abs_wave >= CLIPPING_AMPLITUDE_THRESHOLD))

    approximate_bitrate_kbps = None
    if duration_seconds > 0:
        approximate_bitrate_kbps = (file_size_bytes * 8) / duration_seconds / 1000.0

    return AudioQualityMetrics(
        peak_amplitude=peak_amplitude,
        rms_level=rms_level,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        approximate_bitrate_kbps=approximate_bitrate_kbps,
    )


@dataclass
class SuitabilityAssessment:
    level: str  # "Good" | "Limited" | "Poor"
    reasons: list[str]


def assess_analysis_suitability(metrics: AudioQualityMetrics) -> SuitabilityAssessment:
    """Documented, deterministic rule -- NOT a model prediction, and this
    result NEVER feeds back into or alters the Spectra classification,
    calibrated threshold, or presentation state. It is advisory-only
    metadata shown alongside the result.

    Rule (checked in order, first match wins):
      - Poor: silence_ratio >= 0.85, OR clipping_ratio >= 0.05
      - Limited: silence_ratio >= 0.6, OR clipping_ratio >= 0.01
      - Good: otherwise
    """
    reasons: list[str] = []
    if metrics.silence_ratio >= SILENCE_RATIO_POOR_THRESHOLD:
        reasons.append(f"a high proportion of silence ({metrics.silence_ratio * 100:.0f}%)")
    if metrics.clipping_ratio >= CLIPPING_RATIO_POOR_THRESHOLD:
        reasons.append(f"heavy clipping ({metrics.clipping_ratio * 100:.1f}% of samples)")
    if reasons:
        return SuitabilityAssessment("Poor", reasons)

    if metrics.silence_ratio >= SILENCE_RATIO_LIMITED_THRESHOLD:
        reasons.append(f"a notable proportion of silence ({metrics.silence_ratio * 100:.0f}%)")
    if metrics.clipping_ratio >= CLIPPING_RATIO_LIMITED_THRESHOLD:
        reasons.append(f"some clipping ({metrics.clipping_ratio * 100:.1f}% of samples)")
    if reasons:
        return SuitabilityAssessment("Limited", reasons)

    return SuitabilityAssessment("Good", [])


@dataclass
class MediaMetadata:
    container: str
    codec: str
    bitrate_kbps: str
    duration_seconds: str
    sample_rate: str
    channels: str


def _ffprobe_field(value) -> str:
    return str(value) if value not in (None, "", "N/A", "unknown") else NOT_AVAILABLE


def probe_media_metadata(file_path: str) -> MediaMetadata:
    """Real container/codec/bitrate metadata via ffprobe, when available.
    NEVER guesses -- any field ffprobe cannot report, or if ffprobe itself
    is unavailable, is shown as "Not available" rather than fabricated."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return MediaMetadata(NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE)

    try:
        proc = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", file_path],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        info = json.loads(proc.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return MediaMetadata(NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE)

    fmt = info.get("format", {})
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    stream = audio_streams[0] if audio_streams else {}

    bitrate = fmt.get("bit_rate") or stream.get("bit_rate")
    bitrate_kbps = f"{int(bitrate) / 1000:.0f} kbps" if bitrate else NOT_AVAILABLE

    return MediaMetadata(
        container=_ffprobe_field(fmt.get("format_long_name") or fmt.get("format_name")),
        codec=_ffprobe_field(stream.get("codec_long_name") or stream.get("codec_name")),
        bitrate_kbps=bitrate_kbps,
        duration_seconds=_ffprobe_field(f"{float(fmt['duration']):.2f} s" if fmt.get("duration") else None),
        sample_rate=_ffprobe_field(f"{stream['sample_rate']} Hz" if stream.get("sample_rate") else None),
        channels=_ffprobe_field(stream.get("channels")),
    )
