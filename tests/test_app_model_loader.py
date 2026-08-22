"""Tests for app/model_loader.py — configuration only, no real model load."""

from app.model_loader import DEPLOYMENT_MODEL_ID
from audio_deepfake_detector.config.models_config import load_models_config


def test_deployment_model_is_sara_wav2vec2():
    assert DEPLOYMENT_MODEL_ID == "sara_wav2vec2"


def test_deployment_model_is_enabled_and_candidate_a_is_not():
    config = load_models_config()
    assert config.get(DEPLOYMENT_MODEL_ID).enabled is True
    # Candidate A remains present for research purposes but must not be the
    # production/deployment model.
    assert config.get("caa_wav2vec2").deployment_role != "primary_deployment"


def test_deployment_model_revision_pinned():
    config = load_models_config()
    model_config = config.get(DEPLOYMENT_MODEL_ID)
    assert model_config.revision == "6c43629c953d6ff008501bf5f3eb983ac2321ad6"
