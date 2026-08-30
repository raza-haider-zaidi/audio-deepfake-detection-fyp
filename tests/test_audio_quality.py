"""Tests for app/analysis/audio_quality.py -- pure signal calculations,
no model involved."""

import numpy as np

from app.analysis.audio_quality import (
    AudioQualityMetrics,
    assess_analysis_suitability,
    compute_audio_quality,
)


def test_compute_audio_quality_silent_clip():
    waveform = np.zeros(16000, dtype=np.float32)
    q = compute_audio_quality(waveform, 16000, 32000, 1.0)
    assert q.peak_amplitude == 0.0
    assert q.silence_ratio == 1.0
    assert q.clipping_ratio == 0.0


def test_compute_audio_quality_clipped_clip():
    waveform = np.ones(16000, dtype=np.float32)
    q = compute_audio_quality(waveform, 16000, 32000, 1.0)
    assert q.clipping_ratio == 1.0


def test_compute_audio_quality_empty_waveform_does_not_crash():
    q = compute_audio_quality(np.array([], dtype=np.float32), 16000, 0, 0.0)
    assert q.silence_ratio == 1.0


def test_compute_audio_quality_bitrate_uses_size_and_duration():
    waveform = (0.1 * np.sin(np.linspace(0, 10, 16000))).astype(np.float32)
    q = compute_audio_quality(waveform, 16000, 128000, 4.0)  # 128000 bytes / 4s
    assert q.approximate_bitrate_kbps == (128000 * 8) / 4.0 / 1000.0


def test_assess_suitability_good_for_normal_signal():
    waveform = (0.2 * np.sin(np.linspace(0, 50, 64000))).astype(np.float32)
    q = compute_audio_quality(waveform, 16000, 200000, 4.0)
    assessment = assess_analysis_suitability(q)
    assert assessment.level == "Good"
    assert assessment.reasons == []


def test_assess_suitability_poor_for_heavy_silence():
    q = AudioQualityMetrics(peak_amplitude=0.1, rms_level=0.01, silence_ratio=0.9, clipping_ratio=0.0, approximate_bitrate_kbps=128.0)
    assessment = assess_analysis_suitability(q)
    assert assessment.level == "Poor"
    assert assessment.reasons


def test_assess_suitability_poor_for_heavy_clipping():
    q = AudioQualityMetrics(peak_amplitude=1.0, rms_level=0.5, silence_ratio=0.0, clipping_ratio=0.1, approximate_bitrate_kbps=128.0)
    assessment = assess_analysis_suitability(q)
    assert assessment.level == "Poor"


def test_assess_suitability_limited_for_moderate_silence():
    q = AudioQualityMetrics(peak_amplitude=0.3, rms_level=0.05, silence_ratio=0.7, clipping_ratio=0.0, approximate_bitrate_kbps=128.0)
    assessment = assess_analysis_suitability(q)
    assert assessment.level == "Limited"


def test_suitability_never_touches_classifier_fields():
    """Sanity: the suitability dataclass has no field that could plausibly
    be confused with or feed into a model score/threshold/decision."""
    q = compute_audio_quality(np.zeros(16000, dtype=np.float32), 16000, 32000, 1.0)
    assessment = assess_analysis_suitability(q)
    forbidden_field_names = {"spoof_probability", "bonafide_probability", "threshold", "decision"}
    assert not forbidden_field_names & set(vars(assessment).keys())
    assert not forbidden_field_names & set(vars(q).keys())
