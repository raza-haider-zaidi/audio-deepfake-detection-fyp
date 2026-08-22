"""Real CPU model integration tests.

These download actual pretrained checkpoints (hundreds of MB) and run real
inference. They are excluded from the default suite:

    pytest -m "not integration and not slow"

Run explicitly with:

    pytest -v -m integration

Predicted labels on the deterministic smoke audio used here have NO
scientific or detection-quality meaning — these tests only verify the
load/inference/unload path works correctly on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from audio_deepfake_detector.config.models_config import load_models_config
from audio_deepfake_detector.models.registry import create_detector
from audio_deepfake_detector.preprocessing.audio_loader import load_audio_file
from scripts.generate_smoke_audio import generate_all

SMOKE_AUDIO_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture(scope="module")
def smoke_audio_sample():
    files = generate_all(output_dir=SMOKE_AUDIO_DIR)
    multitone = next(f for f in files if "multitone_4s" in f.name)
    return load_audio_file(multitone)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("model_id", ["sara_wav2vec2", "caa_wav2vec2"])
def test_real_model_cpu_inference(model_id, smoke_audio_sample):
    config = load_models_config()
    assert config.get(model_id).enabled

    detector = create_detector(model_id, device="cpu")
    try:
        metadata = detector.load()
        assert metadata.device == "cpu"

        result = detector.predict(smoke_audio_sample)

        # No CUDA tensors anywhere
        assert not torch.cuda.is_initialized()

        # Probabilities are finite and normalized
        probs = list(result.probabilities.values())
        assert all(np.isfinite(p) for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-3

        assert result.windows_analyzed >= 1
        assert result.inference_time_ms > 0

        # Repeated warm inference works
        for _ in range(3):
            repeat_result = detector.predict(smoke_audio_sample)
            assert repeat_result.windows_analyzed == result.windows_analyzed
    finally:
        detector.unload()
