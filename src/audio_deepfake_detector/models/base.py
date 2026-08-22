"""Model-independent adapter interface.

The generic inference service and CLI must not know anything about
Candidate A's or Candidate B's internal architecture — they only interact
through this interface. See docs/architecture.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from audio_deepfake_detector.utils.datatypes import AudioSample, ModelLoadMetadata, PredictionResult


class BaseDeepfakeDetector(ABC):
    """Common interface every model adapter must implement."""

    model_id: str
    repository: str
    device: str
    sample_rate: int
    window_seconds: float | None

    @abstractmethod
    def load(self) -> ModelLoadMetadata:
        """Load model weights into memory. Must be called before predict()."""

    @abstractmethod
    def predict(self, audio_sample: AudioSample) -> PredictionResult:
        """Run inference on a single decoded audio clip."""

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and any cached tensors from memory."""

    @abstractmethod
    def model_info(self) -> dict:
        """Return a JSON-serializable dict describing this model's config."""
