"""Registry/factory that resolves configured model IDs into adapters.

This is the only place in the codebase that knows how a model_id string
maps to a concrete adapter class. The inference service, CLI, and benchmark
harness all go through this factory instead of importing adapter classes
directly.
"""

from __future__ import annotations

from audio_deepfake_detector.config.models_config import ModelsRegistryConfig, load_models_config
from audio_deepfake_detector.models.base import BaseDeepfakeDetector

_ADAPTER_FACTORIES: dict[str, str] = {
    "caa_wav2vec2": "audio_deepfake_detector.models.candidate_a:CandidateAWav2Vec2Detector",
    "sara_wav2vec2": "audio_deepfake_detector.models.candidate_b:CandidateBWav2Vec2Detector",
    "antideepfake_wav2vec2_small": "audio_deepfake_detector.models.candidate_c:CandidateAntiDeepfakeWav2Vec2Detector",
    "antideepfake_wav2vec2_onnx": "audio_deepfake_detector.models.candidate_d:CandidateAntiDeepfakeOnnxDetector",
}



def _import_adapter_class(dotted_path: str):
    module_path, class_name = dotted_path.split(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_detector(
    model_id: str,
    device: str = "cpu",
    config: ModelsRegistryConfig | None = None,
) -> BaseDeepfakeDetector:
    """Instantiate (but do not load) the adapter for a configured model_id."""
    if config is None:
        config = load_models_config()

    model_config = config.get(model_id)

    if model_id not in _ADAPTER_FACTORIES:
        raise KeyError(
            f"No adapter implementation registered for model id '{model_id}'. "
            f"Known adapters: {sorted(_ADAPTER_FACTORIES)}"
        )

    adapter_class = _import_adapter_class(_ADAPTER_FACTORIES[model_id])
    return adapter_class(model_config=model_config, device=device)


def list_available_model_ids(config: ModelsRegistryConfig | None = None) -> list[str]:
    if config is None:
        config = load_models_config()
    return sorted(set(config.enabled_models()) & set(_ADAPTER_FACTORIES))
