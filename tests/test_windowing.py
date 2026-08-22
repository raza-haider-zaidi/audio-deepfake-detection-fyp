"""Tests for shared windowing/padding/aggregation logic."""

import numpy as np
import pytest

from audio_deepfake_detector.preprocessing.windowing import aggregate_mean_probability, make_windows


def test_short_audio_is_zero_padded_to_one_window():
    waveform = np.ones(1000, dtype=np.float32)
    windows = make_windows(waveform, window_samples=64000, hop_samples=32000)

    assert len(windows) == 1
    assert windows[0].shape[0] == 64000
    assert np.all(windows[0][:1000] == 1.0)
    assert np.all(windows[0][1000:] == 0.0)


def test_exact_window_length_no_padding_needed():
    waveform = np.arange(64000, dtype=np.float32)
    windows = make_windows(waveform, window_samples=64000, hop_samples=32000)

    assert len(windows) == 1
    assert np.array_equal(windows[0], waveform)


def test_long_audio_produces_multiple_overlapping_windows():
    waveform = np.arange(150000, dtype=np.float32)
    windows = make_windows(waveform, window_samples=64000, hop_samples=32000)

    assert len(windows) > 1
    for w in windows:
        assert w.shape[0] == 64000
    # first window matches the start of the waveform exactly
    assert np.array_equal(windows[0], waveform[:64000])


def test_final_partial_window_is_zero_padded():
    waveform = np.ones(70000, dtype=np.float32)
    windows = make_windows(waveform, window_samples=64000, hop_samples=32000)

    last = windows[-1]
    assert last.shape[0] == 64000
    assert np.any(last == 0.0)  # padding present


def test_make_windows_rejects_non_1d():
    with pytest.raises(ValueError):
        make_windows(np.zeros((10, 2), dtype=np.float32), 64000, 32000)


def test_aggregate_mean_probability():
    window_probs = [
        {"bonafide": 0.8, "spoof": 0.2},
        {"bonafide": 0.6, "spoof": 0.4},
    ]
    aggregated = aggregate_mean_probability(window_probs)
    assert aggregated["bonafide"] == pytest.approx(0.7)
    assert aggregated["spoof"] == pytest.approx(0.3)


def test_aggregate_mean_probability_rejects_empty():
    with pytest.raises(ValueError):
        aggregate_mean_probability([])
