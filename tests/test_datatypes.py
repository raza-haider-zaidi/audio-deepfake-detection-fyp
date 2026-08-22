"""Tests for shared typed data structures."""

import numpy as np
import pytest

from audio_deepfake_detector.utils.datatypes import (
    AudioSample,
    ModelLoadMetadata,
    PredictionResult,
    WindowPrediction,
)


def test_audio_sample_valid():
    sample = AudioSample(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
        source_name="test.wav",
    )
    assert sample.waveform.shape[0] == 16000


def test_audio_sample_rejects_non_1d():
    with pytest.raises(ValueError):
        AudioSample(
            waveform=np.zeros((16000, 2), dtype=np.float32),
            sample_rate=16000,
            duration_seconds=1.0,
            source_name="test.wav",
        )


def test_audio_sample_rejects_non_float32():
    with pytest.raises(ValueError):
        AudioSample(
            waveform=np.zeros(16000, dtype=np.float64),
            sample_rate=16000,
            duration_seconds=1.0,
            source_name="test.wav",
        )


def test_prediction_result_fields():
    result = PredictionResult(
        raw_label="bonafide",
        normalized_label="BONAFIDE",
        confidence=0.9,
        probabilities={"bonafide": 0.9, "spoof": 0.1},
        model_id="sara_wav2vec2",
        model_repository="Sara1708/deepfake-audio-wav2vec2",
        device="cpu",
        inference_time_ms=123.4,
        audio_duration_seconds=4.0,
        windows_analyzed=1,
    )
    assert result.normalized_label == "BONAFIDE"
    assert result.windows_analyzed == 1
    assert result.window_predictions == []


def test_window_prediction_fields():
    wp = WindowPrediction(
        window_index=0,
        start_sample=0,
        end_sample=64000,
        raw_label="spoof",
        probabilities={"bonafide": 0.1, "spoof": 0.9},
    )
    assert wp.end_sample - wp.start_sample == 64000


def test_model_load_metadata_fields():
    meta = ModelLoadMetadata(
        model_id="sara_wav2vec2",
        repository="Sara1708/deepfake-audio-wav2vec2",
        revision="6c43629c953d6ff008501bf5f3eb983ac2321ad6",
        checkpoint_filename="stage2_best.pt",
        checkpoint_size_bytes=491044441,
        base_architecture="wav2vec2-base + linear head",
        sample_rate=16000,
        window_seconds=4.0,
        label_mapping={0: "bonafide", 1: "spoof"},
        torch_version="2.13.0+cpu",
        transformers_version="5.15.1",
        device="cpu",
    )
    assert meta.device == "cpu"
    assert meta.label_mapping[1] == "spoof"
