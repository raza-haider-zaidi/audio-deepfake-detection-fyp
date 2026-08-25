"""Tests for the antideepfake_wav2vec2_small adapter (candidate_c.py).

This model is disabled (configs/models.yaml: enabled: false) and its load()
path is documented as BLOCKED in this environment because the official
inference code depends on `fairseq`, which does not install here (see
docs/replacement_model_evaluation.md). These tests therefore check:

  - the adapter module imports cleanly without fairseq being installed
    (fairseq is only imported lazily inside load())
  - registry wiring and config-derived label mapping are correct
  - CPU-only enforcement
  - load() fails with a clear, informative ImportError in this environment

The last point is itself a real regression test: if fairseq ever becomes
installable here, `test_load_raises_importerror_without_fairseq` will start
failing, which is the correct signal to revisit this model's status.
"""

from __future__ import annotations

import pytest

from audio_deepfake_detector.config.models_config import load_models_config
from audio_deepfake_detector.models.candidate_c import CandidateAntiDeepfakeWav2Vec2Detector
from audio_deepfake_detector.models.registry import create_detector


def test_model_config_is_disabled_and_not_in_available_ids():
    config = load_models_config()
    model_config = config.get("antideepfake_wav2vec2_small")
    assert model_config.enabled is False

    from audio_deepfake_detector.models.registry import list_available_model_ids

    assert "antideepfake_wav2vec2_small" not in list_available_model_ids()


def test_label_mapping_from_config_matches_official_script():
    # Official inference script: prob[0] = fake, prob[1] = real.
    config = load_models_config().get("antideepfake_wav2vec2_small")
    assert config.label_mapping[0] == "bonafide"
    assert config.label_mapping[1] == "spoof"


def test_create_detector_via_registry_resolves_correct_class():
    detector = create_detector("antideepfake_wav2vec2_small", device="cpu")
    assert isinstance(detector, CandidateAntiDeepfakeWav2Vec2Detector)
    assert detector.model_id == "antideepfake_wav2vec2_small"
    assert detector.window_seconds is None  # arbitrary-length model, no fixed window


def test_create_detector_rejects_non_cpu_device():
    config = load_models_config().get("antideepfake_wav2vec2_small")
    with pytest.raises(ValueError):
        CandidateAntiDeepfakeWav2Vec2Detector(model_config=config, device="cuda")


def test_predict_before_load_raises():
    detector = create_detector("antideepfake_wav2vec2_small", device="cpu")
    with pytest.raises(RuntimeError):
        detector.predict(audio_sample=None)  # type: ignore[arg-type]


@pytest.mark.integration
def test_load_raises_importerror_without_fairseq():
    detector = create_detector("antideepfake_wav2vec2_small", device="cpu")
    with pytest.raises(ImportError, match="fairseq"):
        detector.load()
