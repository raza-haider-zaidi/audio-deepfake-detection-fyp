"""Tests for app/analysis/robustness.py -- transform generation, temp-file
cleanup, and stability summary. Real ONNX-inference tests are marked
integration; pure-transform tests run in the default suite."""

import os
import tempfile

import numpy as np
import pytest

from app.analysis.robustness import (
    CONDITIONS,
    RobustnessResult,
    _add_noise_at_snr,
    _telephone_filter,
    ffmpeg_available,
    make_condition_waveform,
    stability_summary,
)


def test_telephone_filter_changes_waveform():
    waveform = (0.3 * np.sin(np.linspace(0, 50, 16000))).astype(np.float32)
    filtered = _telephone_filter(waveform, 16000)
    assert filtered.shape == waveform.shape
    assert not np.array_equal(filtered, waveform)


def test_add_noise_at_snr_is_deterministic_for_fixed_seed():
    waveform = np.ones(1000, dtype=np.float32) * 0.1
    rng1 = np.random.default_rng(9)
    rng2 = np.random.default_rng(9)
    out1 = _add_noise_at_snr(waveform, 10.0, rng1)
    out2 = _add_noise_at_snr(waveform, 10.0, rng2)
    np.testing.assert_array_equal(out1, out2)


def test_add_noise_at_snr_silent_input_returns_unchanged():
    waveform = np.zeros(1000, dtype=np.float32)
    rng = np.random.default_rng(1)
    out = _add_noise_at_snr(waveform, 10.0, rng)
    np.testing.assert_array_equal(out, waveform)


def test_make_condition_waveform_original_returns_same_array():
    waveform = np.ones(1000, dtype=np.float32)
    out = make_condition_waveform("original", waveform, 16000, "")
    assert out is waveform


def test_make_condition_waveform_unknown_condition_raises():
    with pytest.raises(ValueError):
        make_condition_waveform("not_a_real_condition", np.zeros(100, dtype=np.float32), 16000, "")


def test_conditions_tuple_matches_documented_set():
    assert CONDITIONS == ("original", "mp3_128k", "mp3_64k", "telephone_300_3400hz", "noise_20db", "noise_10db")


def test_stability_summary_counts_retained_results():
    results = [
        RobustnessResult("original", "Original", 0.9, "SPOOF", 0.0),
        RobustnessResult("mp3_128k", "MP3 128 kbps", 0.85, "SPOOF", -5.0),
        RobustnessResult("telephone_300_3400hz", "Telephone", 0.4, "BONAFIDE", -50.0),
    ]
    summary = stability_summary(results)
    assert summary == "1 of 2 degraded conditions retained the original presentation result."


def test_stability_summary_empty_results():
    assert stability_summary([]) == "Result stability could not be computed."


def test_ffmpeg_available_returns_string_or_none():
    result = ffmpeg_available()
    assert result is None or isinstance(result, str)


@pytest.mark.integration
@pytest.mark.slow
def test_run_robustness_analysis_cleans_up_temp_files_and_covers_all_conditions():
    from audio_deepfake_detector.models.registry import create_detector
    from audio_deepfake_detector.utils.datatypes import AudioSample

    from app.analysis.robustness import run_robustness_analysis

    detector = create_detector("spectra_aasist3_onnx_int8", device="cpu")
    try:
        detector.load(local_onnx_path="models/cache/quantized/spectra-aasist3-int8-dynamic.onnx")
        threshold = detector.model_info()["calibrated_threshold_spoof_probability"]

        rng = np.random.default_rng(3)
        waveform = (rng.standard_normal(4 * 16000) * 0.05).astype(np.float32)
        sample = AudioSample(waveform=waveform, sample_rate=16000, duration_seconds=4.0, source_name="test")

        before = set(os.listdir(tempfile.gettempdir()))
        results = run_robustness_analysis(detector, sample, threshold)
        after = set(os.listdir(tempfile.gettempdir()))

        assert after == before, "run_robustness_analysis must not leave temporary files behind"
        assert len(results) >= 4  # at least the non-ffmpeg-dependent conditions
        assert results[0].condition == "original"
        assert results[0].delta_pp_from_original == 0.0
        for r in results[1:]:
            assert r.delta_pp_from_original is not None
    finally:
        detector.unload()
