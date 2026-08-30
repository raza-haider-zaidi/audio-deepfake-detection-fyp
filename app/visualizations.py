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
PLOT_BG = "#131922"
PLOT_GRID = "#232c3a"
PLOT_TEXT = "#8b96a8"
PLOT_LINE = "#4f8ff7"

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

    fig = matplotlib.figure.Figure(figsize=(6.2, 2.4), facecolor=PLOT_BG)
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

    fig = matplotlib.figure.Figure(figsize=(6.2, 2.8), facecolor=PLOT_BG)
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
