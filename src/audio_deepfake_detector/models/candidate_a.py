"""Adapter for caa-speech-detection-asvspoof2019/wav2vec2-v2-unfrozen ("caa_wav2vec2").

============================================================================
IMPORTANT — SOURCE VERIFICATION STATUS
============================================================================
The model card cites a source implementation at
github.com/sebastiaoteixeira/caa-ai-generated-speech-detector
(src.models.wav2vec2.model.Wav2Vec2Model). That repository was checked
directly on 2026-08-23 and returns HTTP 404 — it does not exist. The exact
forward-pass implementation cannot be independently verified from source.

What IS verified, directly from the downloaded checkpoint's own
`state_dict` keys (not the README prose):

    class_weights          torch.Size([2])
    classifier.0.weight    torch.Size([256, 768])   -> Linear(768, 256)
    classifier.0.bias      torch.Size([256])
    classifier.3.weight    torch.Size([2, 256])     -> Linear(256, 2)
    classifier.3.bias      torch.Size([2])
    encoder.* (212 keys)   -> a facebook/wav2vec2-base-shaped backbone,
                               attribute name "encoder" (not "backbone")

The Sequential indices 0 and 3 (with 1, 2 absent from the state_dict because
GELU/Dropout carry no parameters) are consistent with the model card's
documented head: Linear(768->256) -> GELU -> Dropout(0.1) -> Linear(256->2).
This matches the checkpoint's actual weight shapes, not just the README.

What is NOT verified: the exact pooling operation applied to the encoder's
per-frame hidden states before the classifier head. Pooling is typically
parameter-free (mean/max/first-token), so it leaves no trace in the
state_dict, and the missing source repo means it cannot be confirmed.

This adapter uses **mean pooling over the time dimension** as the forward
pass, because: (a) it is the standard choice for this architecture family,
(b) it is what the sibling candidate (Sara1708/deepfake-audio-wav2vec2,
verified from working source) actually does, and (c) the checkpoint loads
structurally under this assumption with no shape mismatches. This is
explicitly a documented assumption, not a verified fact. See
docs/model_candidate_analysis.md for the full trail.

Label mapping (index 0 = bonafide, index 1 = spoof) is stated directly in
the model card's usage example — treated as documented but not
code-verified (unlike Candidate B, where it was confirmed in source code).

LOAD VERIFICATION RESULT: loading the checkpoint into the reconstruction
above with strict=True fails on exactly one missing key,
"encoder.masked_spec_embed" (a parameter-free-at-inference SpecAugment
masking embedding), and zero unexpected keys. This is consistent with the
model card's statement that encoder masking is disabled at inference
(following Tak et al., 2022) and is treated as expected, not silently
patched over — see the explicit allow-list check in load() below. Every
other one of the 215 checkpoint keys matches this reconstruction exactly.
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
# No overlap is documented for this candidate ("padded/truncated to 64,000
# samples"); a hop equal to the window length means each window is disjoint.
HOP_SAMPLES = 64000


class _CandidateAModel(nn.Module):
    """Reconstructed from checkpoint state_dict keys + documented head
    description. Pooling method is an UNVERIFIED assumption (mean pooling)
    — see module docstring."""

    def __init__(self, backbone_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.encoder = HFWav2Vec2Model.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )
        self.register_buffer("class_weights", torch.zeros(2))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(waveforms)
        pooled = outputs.last_hidden_state.mean(dim=1)  # UNVERIFIED assumption
        return self.classifier(pooled)


class CandidateAWav2Vec2Detector(BaseDeepfakeDetector):
    def __init__(self, model_config: ModelConfig, device: str = "cpu"):
        if device != "cpu":
            raise ValueError("This project only supports CPU inference in Phase 2.")
        self.model_config = model_config
        self.model_id = model_config.id
        self.repository = model_config.repository
        self.device = device
        self.sample_rate = model_config.sample_rate
        self.window_seconds = model_config.window_seconds
        self._model: _CandidateAModel | None = None
        self._label_mapping = model_config.label_mapping or {0: "bonafide", 1: "spoof"}

    def load(self) -> ModelLoadMetadata:
        from huggingface_hub import hf_hub_download
        import transformers

        ckpt_path = hf_hub_download(
            repo_id=self.repository,
            filename=self.model_config.checkpoint_filename,
            revision=self.model_config.revision,
        )

        self._model = _CandidateAModel(backbone_name=self.model_config.base_model)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        # The checkpoint is missing exactly one key: encoder.masked_spec_embed
        # (SpecAugment-style masking embedding). The model card states
        # encoder masking is disabled at inference (following Tak et al.,
        # 2022), so this parameter is legitimately unused at inference time
        # -- its absence is consistent with the documented behavior, not an
        # architecture mismatch. We verify this explicitly rather than
        # silently swallowing any and all missing/unexpected keys.
        result = self._model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"encoder.masked_spec_embed"}
        unexpected_missing = set(result.missing_keys) - allowed_missing
        if unexpected_missing or result.unexpected_keys:
            raise RuntimeError(
                "Candidate A checkpoint did not match the checkpoint-derived "
                f"architecture as expected. Unexpected missing keys: "
                f"{unexpected_missing}. Unexpected extra keys: {result.unexpected_keys}. "
                "Refusing to silently force-load an unverified architecture."
            )
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
            "source_repository_status": self.model_config.source_repository_status,
        }
