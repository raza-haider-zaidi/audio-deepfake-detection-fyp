"""Tests for the antideepfake_wav2vec2_onnx adapter (candidate_d.py).

Unlike candidate_c.py (blocked at load(), because fairseq does not install),
this model's ONNX graph genuinely loads and runs on CPU. It is blocked one
step later, at predict(): the crop/pad-to-fixed-length preprocessing that
turns arbitrary-length audio into this model's required 64000-sample input
is not published anywhere (see docs/replacement_model_evaluation.md, "ONNX
Rescue Investigation"). predict() therefore deliberately raises rather than
guessing that preprocessing and fabricating a result.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_deepfake_detector.config.models_config import load_models_config
from audio_deepfake_detector.models.candidate_d import CandidateAntiDeepfakeOnnxDetector
from audio_deepfake_detector.models.registry import create_detector


def test_model_config_is_disabled_and_not_in_available_ids():
    config = load_models_config()
    model_config = config.get("antideepfake_wav2vec2_onnx")
    assert model_config.enabled is False

    from audio_deepfake_detector.models.registry import list_available_model_ids

    assert "antideepfake_wav2vec2_onnx" not in list_available_model_ids()


def test_config_window_matches_sara_window_exactly():
    # A fair CPU-latency comparison against sara_wav2vec2 depends on both
    # models sharing the same 4.0s / 64000-sample input window.
    config = load_models_config()
    onnx_config = config.get("antideepfake_wav2vec2_onnx")
    sara_config = config.get("sara_wav2vec2")
    assert onnx_config.window_samples == sara_config.window_samples == 64000
    assert onnx_config.window_seconds == sara_config.window_seconds == 4.0


def test_create_detector_via_registry_resolves_correct_class():
    detector = create_detector("antideepfake_wav2vec2_onnx", device="cpu")
    assert isinstance(detector, CandidateAntiDeepfakeOnnxDetector)
    assert detector.model_id == "antideepfake_wav2vec2_onnx"
    assert detector.window_seconds == 4.0


def test_create_detector_rejects_non_cpu_device():
    config = load_models_config().get("antideepfake_wav2vec2_onnx")
    with pytest.raises(ValueError):
        CandidateAntiDeepfakeOnnxDetector(model_config=config, device="cuda")


def test_predict_before_load_raises():
    detector = create_detector("antideepfake_wav2vec2_onnx", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        detector.predict(audio_sample=None)  # type: ignore[arg-type]


def test_run_raw_window_before_load_raises():
    detector = create_detector("antideepfake_wav2vec2_onnx", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        detector.run_raw_window(np.zeros(64000, dtype=np.float32))


def test_model_info_before_load_reports_not_loaded_and_blocked_predict():
    detector = create_detector("antideepfake_wav2vec2_onnx", device="cpu")
    info = detector.model_info()
    assert info["loaded"] is False
    assert info["predict_status"] == "blocked_unverified_preprocessing"
    assert info["label_mapping_verified_for_onnx_export"] is False


@pytest.mark.integration
@pytest.mark.slow
def test_load_succeeds_on_cpu_and_predict_raises_preprocessing_error():
    detector = create_detector("antideepfake_wav2vec2_onnx", device="cpu")
    try:
        metadata = detector.load()
        assert metadata.device == "cpu"

        # load() genuinely works -- this is the key difference from candidate_c.
        assert detector.model_info()["loaded"] is True

        with pytest.raises(RuntimeError, match="preprocessing"):
            detector.predict(audio_sample=None)  # type: ignore[arg-type]

        # The raw technical probe works and is deterministic (no accuracy claim).
        smoke = np.zeros(64000, dtype=np.float32)
        logits1, _ = detector.run_raw_window(smoke)
        logits2, _ = detector.run_raw_window(smoke)
        assert np.all(np.isfinite(logits1))
        assert np.array_equal(logits1, logits2)
    finally:
        detector.unload()

