"""Generic inference service.

Knows nothing about Candidate A / Candidate B internals — only interacts
through BaseDeepfakeDetector, obtained via the model registry/factory.
"""

from __future__ import annotations

from audio_deepfake_detector.models.registry import create_detector
from audio_deepfake_detector.preprocessing.audio_loader import load_audio_file
from audio_deepfake_detector.utils.datatypes import PredictionResult
from audio_deepfake_detector.utils.device import resolve_device


def predict_file(file_path: str, model_id: str, device: str = "cpu") -> PredictionResult:
    """Load a model, run inference on a single audio file, and unload it.

    Convenience wrapper for the CLI. For repeated inference against the
    same model, prefer creating a detector once via create_detector()/load()
    and calling predict() multiple times.
    """
    resolved_device = resolve_device(device)
    audio_sample = load_audio_file(file_path)

    detector = create_detector(model_id, device=resolved_device)
    try:
        detector.load()
        return detector.predict(audio_sample)
    finally:
        detector.unload()
