"""Shared typed data structures for audio deepfake detection.

These types are used by preprocessing, model adapters, and the inference
service. They are plain dataclasses with no dependency on Streamlit or any
particular model implementation, so they can be reused by the CLI, the
benchmark harness, and (later) the Streamlit UI without coupling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioSample:
    """A decoded, normalized audio clip ready for model-specific framing."""

    waveform: np.ndarray  # 1-D float32 array, mono
    sample_rate: int
    duration_seconds: float
    source_name: str

    def __post_init__(self) -> None:
        if self.waveform.ndim != 1:
            raise ValueError(
                f"AudioSample.waveform must be 1-D mono; got shape {self.waveform.shape}"
            )
        if self.waveform.dtype != np.float32:
            raise ValueError(
                f"AudioSample.waveform must be float32; got {self.waveform.dtype}"
            )


@dataclass
class WindowPrediction:
    """Prediction for a single fixed-length window of audio."""

    window_index: int
    start_sample: int
    end_sample: int
    raw_label: str
    probabilities: dict[str, float]


@dataclass
class PredictionResult:
    """Clip-level prediction result from a model adapter."""

    raw_label: str
    normalized_label: str
    confidence: float
    probabilities: dict[str, float]
    model_id: str
    model_repository: str
    device: str
    inference_time_ms: float
    audio_duration_seconds: float
    windows_analyzed: int = 1
    window_predictions: list[WindowPrediction] = field(default_factory=list)


@dataclass
class ModelLoadMetadata:
    """Metadata recorded when a model adapter successfully loads."""

    model_id: str
    repository: str
    revision: str
    checkpoint_filename: str
    checkpoint_size_bytes: int | None
    base_architecture: str
    sample_rate: int
    window_seconds: float | None
    label_mapping: dict[int, str]
    torch_version: str
    transformers_version: str | None
    device: str
    loaded_at_unix: float = field(default_factory=time.time)


@dataclass
class BenchmarkStats:
    """Aggregated timing statistics over repeated warm-inference runs."""

    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    n_runs: int


@dataclass
class ModelBenchmarkResult:
    """Full CPU engineering benchmark for one model."""

    model_id: str
    repository: str
    revision: str
    device: str
    checkpoint_size_bytes: int | None
    ram_before_load_mb: float
    ram_after_load_mb: float
    load_memory_increase_mb: float
    cold_load_time_ms: float
    first_inference_time_ms: float
    warm_inference: BenchmarkStats | None
    load_succeeded: bool
    error: str | None = None
