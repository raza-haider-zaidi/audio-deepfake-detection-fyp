"""Tests for the model registry/factory (no real model loading)."""

import pytest

from audio_deepfake_detector.models.candidate_b import CandidateBWav2Vec2Detector
from audio_deepfake_detector.models.registry import create_detector, list_available_model_ids


def test_list_available_model_ids_excludes_disabled_reference():
    ids = list_available_model_ids()
    assert "caa_wav2vec2" in ids
    assert "sara_wav2vec2" in ids
    assert "hf_reference" not in ids


def test_create_detector_resolves_correct_adapter_class():
    detector = create_detector("sara_wav2vec2", device="cpu")
    assert isinstance(detector, CandidateBWav2Vec2Detector)
    assert detector.model_id == "sara_wav2vec2"
    assert detector.device == "cpu"


def test_create_detector_unknown_model_id_raises():
    with pytest.raises(KeyError):
        create_detector("does_not_exist", device="cpu")


def test_create_detector_rejects_non_cpu_device():
    from audio_deepfake_detector.config.models_config import load_models_config

    config = load_models_config().get("sara_wav2vec2")
    with pytest.raises(ValueError):
        CandidateBWav2Vec2Detector(model_config=config, device="cuda")
