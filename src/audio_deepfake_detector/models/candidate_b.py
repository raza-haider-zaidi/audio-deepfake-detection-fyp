"""Adapter for Sara1708/deepfake-audio-wav2vec2 ("sara_wav2vec2").

Architecture, windowing, padding, and label-index order are all reproduced
directly from the verified source code at
github.com/Saracasm/deepfake-audio-detection (confirmed reachable,
HTTP 200, on 2026-08-23) — not guessed. See
docs/model_candidate_analysis.md for the full verification trail.

Source class (src/models/wav2vec_classifier.py):

    class Wav2VecClassifier(nn.Module):
        def __init__(self, backbone_name="facebook/wav2vec2-base",
                     num_classes=2, freeze_backbone=True):
            self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)
            hidden_size = self.backbone.config.hidden_size
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, waveforms):
            outputs = self.backbone(waveforms)
            pooled = outputs.last_hidden_state.mean(dim=1)
            return self.classifier(pooled)

Source windowing (src/data/preprocessing.py): WINDOW_SAMPLES=64000,
HOP_SAMPLES=32000 (50% overlap), short clips zero-padded to one window.

Source label mapping (src/inference/predict.py):
    probs[:, 1] = spoof probability  ->  index 0 = bonafide, index 1 = spoof
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model as HFWav2Vec2Model

from audio_deepfake_detector.config.models_config import ModelConfig
from audio_deepfake_detector.models.base import BaseDeepfakeDetector
from audio_deepfake_detector.preprocessing.windowing import aggregate_mean_probability, make_windows
from audio_deepfake_detector.utils.datatypes import (
    AudioSample,
    ModelLoadMetadata,
    PredictionResult,
    WindowPrediction,
)

WINDOW_SAMPLES = 64000
HOP_SAMPLES = 32000


class _Wav2VecClassifier(nn.Module):
    """Faithful reproduction of the source repo's Wav2VecClassifier."""

    def __init__(self, backbone_name: str = "facebook/wav2vec2-base", num_classes: int = 2):
        super().__init__()
        self.backbone = HFWav2Vec2Model.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(waveforms)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


class CandidateBWav2Vec2Detector(BaseDeepfakeDetector):
    def __init__(self, model_config: ModelConfig, device: str = "cpu"):
        if device != "cpu":
            raise ValueError("This project only supports CPU inference in Phase 2.")
        self.model_config = model_config
        self.model_id = model_config.id
        self.repository = model_config.repository
        self.device = device
        self.sample_rate = model_config.sample_rate
        self.window_seconds = model_config.window_seconds
        self._model: _Wav2VecClassifier | None = None
        self._label_mapping = model_config.label_mapping or {0: "bonafide", 1: "spoof"}

    def load(self) -> ModelLoadMetadata:
        from huggingface_hub import hf_hub_download
        import transformers

        ckpt_path = hf_hub_download(
            repo_id=self.repository,
            filename=self.model_config.checkpoint_filename,
            revision=self.model_config.revision,
        )

        self._model = _Wav2VecClassifier(backbone_name=self.model_config.base_model, num_classes=2)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self._model.load_state_dict(state_dict)
        self._model.to("cpu")
        self._model.eval()

        return ModelLoadMetadata(
            model_id=self.model_id,
            repository=self.repository,
            revision=self.model_config.revision,
            checkpoint_filename=self.model_config.checkpoint_filename,
            checkpoint_size_bytes=self.model_config.checkpoint_size_bytes,
            base_architecture=self.model_config.architecture,
            sample_rate=self.sample_rate,
            window_seconds=self.window_seconds,
            label_mapping=self._label_mapping,
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            device=self.device,
        )

    def predict(self, audio_sample: AudioSample) -> PredictionResult:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        windows = make_windows(audio_sample.waveform, WINDOW_SAMPLES, HOP_SAMPLES)
        batch = torch.from_numpy(np.stack(windows, axis=0)).float()

        start = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        inference_time_ms = (time.perf_counter() - start) * 1000.0

        window_predictions: list[WindowPrediction] = []
        window_prob_dicts: list[dict[str, float]] = []
        for i, window_probs in enumerate(probs):
            prob_dict = {
                self._label_mapping[0]: float(window_probs[0]),
                self._label_mapping[1]: float(window_probs[1]),
            }
            window_prob_dicts.append(prob_dict)
            raw_label = self._label_mapping[int(window_probs.argmax())]
            window_predictions.append(
                WindowPrediction(
                    window_index=i,
                    start_sample=i * HOP_SAMPLES,
                    end_sample=i * HOP_SAMPLES + WINDOW_SAMPLES,
                    raw_label=raw_label,
                    probabilities=prob_dict,
                )
            )

        clip_probs = aggregate_mean_probability(window_prob_dicts)
        raw_label = max(clip_probs, key=clip_probs.get)
        confidence = clip_probs[raw_label]

        return PredictionResult(
            raw_label=raw_label,
            normalized_label="BONAFIDE" if raw_label == "bonafide" else "SPOOF",
            confidence=confidence,
            probabilities=clip_probs,
            model_id=self.model_id,
            model_repository=self.repository,
            device=self.device,
            inference_time_ms=inference_time_ms,
            audio_duration_seconds=audio_sample.duration_seconds,
            windows_analyzed=len(windows),
            window_predictions=window_predictions,
        )

    def unload(self) -> None:
        self._model = None
        import gc

        gc.collect()

    def model_info(self) -> dict:
        return {
            "model_id": self.model_id,
            "repository": self.repository,
            "revision": self.model_config.revision,
            "architecture": self.model_config.architecture,
            "sample_rate": self.sample_rate,
            "window_seconds": self.window_seconds,
            "label_mapping": self._label_mapping,
            "device": self.device,
            "loaded": self._model is not None,
            # UI-facing display metadata (docs/spectra_inconclusive_state.md
            # Step 7) -- generic technical-details rendering reads these
            # instead of hardcoding model-specific strings per adapter.
            "display_name": "Sara Wav2Vec2",
            "architecture_short": "Wav2Vec2 (facebook/wav2vec2-base backbone)",
            "runtime": "PyTorch (CPU)",
            "native_window_description": f"{WINDOW_SAMPLES} samples (4.0s), {HOP_SAMPLES}-sample hop (50% overlap sliding window)",
            "aggregation_description": (
                "Mean spoof probability across all sliding windows when a "
                "clip spans more than one 4-second window -- a project-level "
                "aggregation choice, not scientifically validated by the "
                "original model authors."
            ),
        }
