"""Waveform and mel-spectrogram plotting helpers.

Pure functions that take a numpy waveform and return a matplotlib Figure.
No Streamlit import here — streamlit_app.py is responsible for rendering
(st.pyplot) and closing figures after rendering to avoid retaining memory
across analyses.

IMPORTANT: these visualizations are for user interpretability only. The
Wav2Vec2 detector operates on the raw waveform directly; the mel
spectrogram is not the model's input.
"""

from __future__ import annotations

import matplotlib.figure
import numpy as np

MEL_N_FFT = 400
MEL_WIN_LENGTH = 400
MEL_HOP_LENGTH = 160
MEL_N_MELS = 80

# Plot theme matching app/styles.py's dark surface tokens, so charts sit
# visually flush with the surrounding card rather than showing as a bright
# white rectangle in an otherwise dark interface.
PLOT_BG = "#FFFFFF"
PLOT_GRID = "#DEE3EE"
PLOT_TEXT = "#4A5578"
PLOT_LINE = "#4568F2"
PLOT_LINE_SECONDARY = "#0EA5B7"
PLOT_WARNING = "#C88A1C"

# Cap the number of points actually drawn for the waveform so a 30-second
# clip at 16 kHz (480,000 samples) doesn't create unnecessary UI/memory
# overhead. This only affects the plot, never the audio passed to the model.
MAX_WAVEFORM_PLOT_POINTS = 4000


def _downsample_for_plot(waveform: np.ndarray, max_points: int = MAX_WAVEFORM_PLOT_POINTS) -> np.ndarray:
    if waveform.shape[0] <= max_points:
        return waveform
    stride = int(np.ceil(waveform.shape[0] / max_points))
    return waveform[::stride]


def plot_waveform(waveform: np.ndarray, sample_rate: int) -> matplotlib.figure.Figure:
    plotted = _downsample_for_plot(waveform)
    stride = max(1, waveform.shape[0] // max(1, plotted.shape[0]))
    time_axis = np.arange(plotted.shape[0]) * stride / sample_rate

    fig = matplotlib.figure.Figure(figsize=(6.2, 2.6), facecolor=PLOT_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PLOT_BG)
    ax.plot(time_axis, plotted, linewidth=0.8, color=PLOT_LINE)
    ax.set_xlabel("Time (s)", color=PLOT_TEXT, fontsize=9)
    ax.set_ylabel("Amplitude", color=PLOT_TEXT, fontsize=9)
    ax.set_xlim(0, time_axis[-1] if time_axis.size else 1.0)
    ax.tick_params(colors=PLOT_TEXT, labelsize=8)
    ax.grid(True, color=PLOT_GRID, linewidth=0.6, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(PLOT_GRID)
    fig.tight_layout()
    return fig


def plot_segment_timeline(segment_spoof_probs: list[float], segment_duration_seconds: float, threshold: float) -> matplotlib.figure.Figure:
    """Segment-evidence timeline: spoof probability per non-overlapping
    segment, with the calibrated spoof threshold drawn as a reference
    line. Purely descriptive -- see app/analysis/evidence.py."""
    n = len(segment_spoof_probs)
    x = [(i + 0.5) * segment_duration_seconds for i in range(n)]

    fig = matplotlib.figure.Figure(figsize=(9.5, 2.6), facecolor=PLOT_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PLOT_BG)
    ax.plot(x, segment_spoof_probs, marker="o", markersize=4, linewidth=1.2, color=PLOT_LINE)
    ax.axhline(threshold, color=PLOT_WARNING, linewidth=1.2, linestyle="--", label="Calibrated spoof threshold")
    ax.axhline(0.5, color=PLOT_LINE_SECONDARY, linewidth=1.0, linestyle=":", label="0.5 softmax direction", alpha=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (s)", color=PLOT_TEXT, fontsize=9)
    ax.set_ylabel("Spoof probability", color=PLOT_TEXT, fontsize=9)
    ax.tick_params(colors=PLOT_TEXT, labelsize=8)
    ax.grid(True, color=PLOT_GRID, linewidth=0.6, alpha=0.6)
    legend = ax.legend(loc="upper right", fontsize=7, facecolor=PLOT_BG, edgecolor=PLOT_GRID)
    for text in legend.get_texts():
        text.set_color(PLOT_TEXT)
    for spine in ax.spines.values():
        spine.set_color(PLOT_GRID)
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(waveform: np.ndarray, sample_rate: int) -> matplotlib.figure.Figure:
    import librosa
    import librosa.display

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=MEL_N_FFT,
        win_length=MEL_WIN_LENGTH,
        hop_length=MEL_HOP_LENGTH,
        n_mels=MEL_N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = matplotlib.figure.Figure(figsize=(6.2, 2.6), facecolor=PLOT_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PLOT_BG)
    img = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=MEL_HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma",
    )
    ax.set_xlabel("Time (s)", color=PLOT_TEXT, fontsize=9)
    ax.set_ylabel("Frequency", color=PLOT_TEXT, fontsize=9)
    ax.tick_params(colors=PLOT_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PLOT_GRID)
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.tick_params(colors=PLOT_TEXT, labelsize=8)
    fig.tight_layout()
    return fig


def _style_axes(ax) -> None:
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=PLOT_TEXT, labelsize=8)
    ax.grid(True, color=PLOT_GRID, linewidth=0.6, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(PLOT_GRID)


def plot_threshold_curve(thresholds: list[float], fprs: list[float], fnrs: list[float], current_threshold: float) -> matplotlib.figure.Figure:
    """Research-only threshold explorer chart: FPR/FNR vs. a swept
    threshold, with the current PRODUCTION threshold marked -- never the
    other way around (production is never changed by this chart)."""
    fig = matplotlib.figure.Figure(figsize=(7.5, 3.0), facecolor=PLOT_BG)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    ax.plot(thresholds, fprs, color=PLOT_LINE, linewidth=1.6, label="Bonafide FPR")
    ax.plot(thresholds, fnrs, color=PLOT_LINE_SECONDARY, linewidth=1.6, label="Spoof FNR")
    ax.axvline(current_threshold, color=PLOT_WARNING, linewidth=1.2, linestyle="--", label="Production threshold")
    ax.set_xlabel("Hypothetical threshold (spoof probability)", color=PLOT_TEXT, fontsize=9)
    ax.set_ylabel("Rate", color=PLOT_TEXT, fontsize=9)
    legend = ax.legend(loc="upper center", fontsize=7, facecolor=PLOT_BG, edgecolor=PLOT_GRID)
    for text in legend.get_texts():
        text.set_color(PLOT_TEXT)
    fig.tight_layout()
    return fig


def plot_robustness_comparison(labels: list[str], spoof_probs: list[float], threshold: float) -> matplotlib.figure.Figure:
    """Bar chart of spoof probability per robustness condition, with the
    production threshold as a reference line."""
    fig = matplotlib.figure.Figure(figsize=(8.0, 3.0), facecolor=PLOT_BG)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    colors = [PLOT_WARNING if p >= threshold else PLOT_LINE_SECONDARY for p in spoof_probs]
    ax.bar(labels, [p * 100 for p in spoof_probs], color=colors)
    ax.axhline(threshold * 100, color=PLOT_LINE, linewidth=1.2, linestyle="--", label="Calibrated spoof threshold")
    ax.set_ylabel("Spoof probability (%)", color=PLOT_TEXT, fontsize=9)
    ax.tick_params(axis="x", labelrotation=20)
    legend = ax.legend(loc="upper right", fontsize=7, facecolor=PLOT_BG, edgecolor=PLOT_GRID)
    for text in legend.get_texts():
        text.set_color(PLOT_TEXT)
    fig.tight_layout()
    return fig
