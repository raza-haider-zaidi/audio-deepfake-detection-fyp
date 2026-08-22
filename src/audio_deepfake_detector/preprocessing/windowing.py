"""Reusable fixed-window slicing for models with a fixed input length.

Model adapters call this with their own window/hop sizes rather than the
shared audio loader doing it automatically, because different candidate
models impose different framing rules (see docs/model_candidate_analysis.md).
"""

from __future__ import annotations

import numpy as np


def make_windows(
    waveform: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> list[np.ndarray]:
    """Split a 1-D waveform into fixed-length windows.

    - If the waveform is shorter than or equal to one window, it is
      zero-padded to exactly one window (short-audio case).
    - Otherwise it is split into overlapping windows of `window_samples`
      with a stride of `hop_samples`; the final partial window is
      zero-padded. This mirrors Candidate B's documented windowing
      strategy (docs/model_candidate_analysis.md) and is used as the
      general-purpose default for any model with a fixed window length.

    Returns a non-empty list of float32 arrays, each exactly
    `window_samples` long.
    """
    if waveform.ndim != 1:
        raise ValueError(f"make_windows expects a 1-D waveform; got shape {waveform.shape}")
    if window_samples <= 0:
        raise ValueError(f"window_samples must be positive; got {window_samples}")
    if hop_samples <= 0:
        raise ValueError(f"hop_samples must be positive; got {hop_samples}")

    n = waveform.shape[0]

    if n <= window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[:n] = waveform
        return [padded]

    windows: list[np.ndarray] = []
    start = 0
    while start < n:
        end = start + window_samples
        if end <= n:
            windows.append(waveform[start:end].astype(np.float32, copy=False))
        else:
            tail = waveform[start:]
            padded = np.zeros(window_samples, dtype=np.float32)
            padded[: tail.shape[0]] = tail
            windows.append(padded)
            break
        start += hop_samples
    return windows


def aggregate_mean_probability(window_probabilities: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-window class probabilities into a clip-level result
    using the mean across windows.

    This is a simple, defensible default — NOT claimed to be scientifically
    optimal. See docs/model_candidate_analysis.md / Phase 2 report.
    """
    if not window_probabilities:
        raise ValueError("Cannot aggregate an empty list of window probabilities")

    labels = window_probabilities[0].keys()
    return {
        label: float(np.mean([w[label] for w in window_probabilities]))
        for label in labels
    }
