"""Tests for configs/models.yaml loading."""

from audio_deepfake_detector.config.models_config import load_models_config


def test_models_config_loads():
    config = load_models_config()
    assert "caa_wav2vec2" in config.models
    assert "sara_wav2vec2" in config.models
    assert "hf_reference" in config.models


def test_enabled_models_excludes_reference():
    config = load_models_config()
    enabled = config.enabled_models()
    assert "caa_wav2vec2" in enabled
    assert "sara_wav2vec2" in enabled
    assert "hf_reference" not in enabled  # disabled by default


def test_label_mappings_documented():
    config = load_models_config()
    caa = config.get("caa_wav2vec2")
    sara = config.get("sara_wav2vec2")
    assert caa.label_mapping == {0: "bonafide", 1: "spoof"}
    assert sara.label_mapping == {0: "bonafide", 1: "spoof"}


def test_unknown_model_id_raises():
    config = load_models_config()
    try:
        config.get("does_not_exist")
        assert False, "expected KeyError"
    except KeyError:
        pass
