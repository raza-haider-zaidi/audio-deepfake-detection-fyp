"""Adapter for nii-yamagishilab/wav2vec-small-anti-deepfake ("antideepfake_wav2vec2_small").

STATUS: BLOCKED. This adapter faithfully reproduces the model's official
published inference script (the "Use this model" code block in the Hugging
Face model card README.md, verified 2026-08-26) so the implementation is
ready and not lost -- but `load()` cannot currently succeed in this
project's CPU/Windows/Python 3.12 environment, because the official code
path depends on `fairseq.models.wav2vec.Wav2Vec2Model`, and `fairseq` does
not install here. See configs/models.yaml (antideepfake_wav2vec2_small
entry) and docs/replacement_model_evaluation.md for the full, reproducible
verification trail of that failure. This model is disabled in
configs/models.yaml (enabled: false) and NOT wired into the Streamlit app.

Architecture (from the official script, not guessed):

    class SSLModel(torch.nn.Module):
        def __init__(self):
            cfg = Wav2Vec2Config(encoder_layers=12, encoder_embed_dim=768,
                                  quantize_targets=True, latent_dim=256,
                                  final_dim=256)
            self.model = Wav2Vec2Model(cfg)  # fairseq SSL model

        def extract_feat(self, input_data):
            return self.model(input_data, mask=False,
                               features_only=True)['x']

    class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
        def forward(self, wav):
            emb = self.m_ssl.extract_feat(wav)       # [B, T, D]
            emb = emb.transpose(1, 2)                 # [B, D, T]
            pooled = self.adap_pool1d(emb).squeeze(-1) # [B, D]
            return self.proj_fc(pooled)                # [B, 2]

Preprocessing (from the official script): mono-mix, resample to 16kHz,
per-utterance layer-norm over the raw waveform. No fixed-length windowing --
the model card documents arbitrary-length support (batched internally up to
100s per the HF card; this adapter runs one utterance per forward pass).

Label mapping (from the official script's own print statement:
`f"real prob = {prob[1]:.3f}, fake prob = {prob[0]:.3f}"`): index 0 = fake
(spoof), index 1 = real (bonafide).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from audio_deepfake_detector.config.models_config import ModelConfig
from audio_deepfake_detector.models.base import BaseDeepfakeDetector
from audio_deepfake_detector.utils.datatypes import (
    AudioSample,
    ModelLoadMetadata,
    PredictionResult,
    WindowPrediction,
)


class _SSLModel(nn.Module):
    """Fairseq wav2vec2-base SSL frontend, per the official inference script."""

    def __init__(self) -> None:
        super().__init__()
        from fairseq.models.wav2vec import Wav2Vec2Config, Wav2Vec2Model

        cfg = Wav2Vec2Config(
            encoder_layers=12,
            encoder_embed_dim=768,
            quantize_targets=True,
            latent_dim=256,
            final_dim=256,
        )
        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, input_data: torch.Tensor) -> torch.Tensor:
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        with torch.no_grad():
            features = self.model(input_data, mask=False, features_only=True)["x"]
        return features


class _DeepfakeDetector(nn.Module):
    """Faithful reproduction of the official DeepfakeDetector (SSL + pool + FC)."""

    def __init__(self) -> None:
        super().__init__()
        self.ssl_orig_output_dim = 768
        self.num_classes = 2
        self.m_ssl = _SSLModel()
        self.adap_pool1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = nn.Linear(in_features=self.ssl_orig_output_dim, out_features=self.num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        emb = self.m_ssl.extract_feat(wav)
        emb = emb.transpose(1, 2)
        pooled_emb = self.adap_pool1d(emb).squeeze(-1)
        return self.proj_fc(pooled_emb)


class CandidateAntiDeepfakeWav2Vec2Detector(BaseDeepfakeDetector):
    def __init__(self, model_config: ModelConfig, device: str = "cpu"):
        if device != "cpu":
            raise ValueError("This project only supports CPU inference.")
        self.model_config = model_config
        self.model_id = model_config.id
        self.repository = model_config.repository
        self.device = device
        self.sample_rate = model_config.sample_rate
        self.window_seconds = model_config.window_seconds
        self._model: _DeepfakeDetector | None = None
        self._label_mapping = model_config.label_mapping or {0: "bonafide", 1: "spoof"}

    def load(self) -> ModelLoadMetadata:
        try:
            import fairseq  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "antideepfake_wav2vec2_small requires the `fairseq` package, which "
                "does not currently install in this project's environment (verified "
                "pip failure -- see configs/models.yaml notes and "
                "docs/replacement_model_evaluation.md). This adapter cannot be "
                "loaded until that dependency is resolved."
            ) from exc

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        ckpt_path = hf_hub_download(
            repo_id=self.repository,
            filename=self.model_config.checkpoint_filename,
            revision=self.model_config.revision,
        )

        self._model = _DeepfakeDetector()
        state_dict = load_file(ckpt_path)
        self._model.load_state_dict(state_dict)
        self._model.to("cpu")
        self._model.eval()

        import transformers

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

    def _preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(waveform).float()
        with torch.no_grad():
            wav = torch.nn.functional.layer_norm(wav, wav.shape)
        return wav.unsqueeze(0)

    def predict(self, audio_sample: AudioSample) -> PredictionResult:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        wav = self._preprocess(audio_sample.waveform)

        start = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(wav)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        inference_time_ms = (time.perf_counter() - start) * 1000.0

        prob_dict = {
            self._label_mapping[0]: float(probs[0]),
            self._label_mapping[1]: float(probs[1]),
        }
        raw_label = self._label_mapping[int(probs.argmax())]

        window_predictions = [
            WindowPrediction(
                window_index=0,
                start_sample=0,
                end_sample=len(audio_sample.waveform),
                raw_label=raw_label,
                probabilities=prob_dict,
            )
        ]

        return PredictionResult(
            raw_label=raw_label,
            normalized_label="BONAFIDE" if raw_label == "bonafide" else "SPOOF",
            confidence=prob_dict[raw_label],
            probabilities=prob_dict,
            model_id=self.model_id,
            model_repository=self.repository,
            device=self.device,
            inference_time_ms=inference_time_ms,
            audio_duration_seconds=audio_sample.duration_seconds,
            windows_analyzed=1,
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
        }
