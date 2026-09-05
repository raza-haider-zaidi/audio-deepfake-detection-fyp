"""Tests for app/model_loader.py — configuration only, no real model load.

EXPERIMENTAL CANDIDATE BRANCH (feat/spectra-streamlit-candidate):
DEPLOYMENT_MODEL_ID is spectra_aasist3_onnx_int8 on this branch only.
main's deployment_loader still points at sara_wav2vec2 -- this branch does
not touch main.
"""

from app.model_loader import DEPLOYMENT_MODEL_ID
from audio_deepfake_detector.config.models_config import load_models_config


def test_deployment_model_is_spectra_int8_on_this_branch():
    assert DEPLOYMENT_MODEL_ID == "spectra_aasist3_onnx_int8"


def test_deployment_model_config_resolves():
    config = load_models_config()
    model_config = config.get(DEPLOYMENT_MODEL_ID)
    assert model_config.checkpoint_filename == "spectra-aasist3-int8-dynamic.onnx"
    assert model_config.expected_sha256 == "444f832d306a2be4f823119f84e698e8821db6a1aab248593d4b05b7a9a48108"


def test_sara_remains_present_but_not_the_deployment_model():
    # Sara must remain fully present as research evidence (Phase 4,
    # docs/inference_calibration.md, docs/spectra_aasist3_evaluation.md)
    # but must not be what THIS branch's Streamlit app loads by default.
    config = load_models_config()
    assert config.get("sara_wav2vec2") is not None
    assert DEPLOYMENT_MODEL_ID != "sara_wav2vec2"
