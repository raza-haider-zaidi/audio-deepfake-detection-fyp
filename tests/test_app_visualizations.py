"""Tests for app/visualizations.py — figure generation, no model needed."""

import matplotlib.figure
import numpy as np

from app.visualizations import _downsample_for_plot, plot_mel_spectrogram, plot_waveform


def test_downsample_for_plot_short_waveform_unchanged():
    waveform = np.zeros(1000, dtype=np.float32)
    result = _downsample_for_plot(waveform, max_points=4000)
    assert result.shape[0] == 1000


def test_downsample_for_plot_long_waveform_reduced():
    waveform = np.zeros(480000, dtype=np.float32)  # 30s @ 16kHz
    result = _downsample_for_plot(waveform, max_points=4000)
    assert result.shape[0] <= 4000
    assert result.shape[0] > 0


def test_plot_waveform_returns_figure():
    waveform = (0.1 * np.sin(np.linspace(0, 10, 16000))).astype(np.float32)
    fig = plot_waveform(waveform, sample_rate=16000)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_mel_spectrogram_returns_figure():
    waveform = (0.1 * np.sin(np.linspace(0, 10, 16000))).astype(np.float32)
    fig = plot_mel_spectrogram(waveform, sample_rate=16000)
    assert isinstance(fig, matplotlib.figure.Figure)
