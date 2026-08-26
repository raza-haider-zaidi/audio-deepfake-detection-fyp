"""Adapter for SpeechAntiSpoofingBenchmarks/Wav2Vec2-Small-AntiDeepfake ONNX
export ("antideepfake_wav2vec2_onnx").

STATUS: LOADS, BUT PREDICT() IS BLOCKED. This is a documented, community
("Speech Anti-Spoofing Arena") mirror -- not the original NII authors -- of
the same checkpoint as antideepfake_wav2vec2_small (candidate_c.py), with a
CPU-runnable ONNX export added on top so the fairseq dependency can be
avoided. Unlike candidate_c, the ONNX **session loads successfully on CPU**
(verified 2026-08-27, onnxruntime 1.29.0, CPUExecutionProvider only) and the
graph's input/output contract is read directly from the model, not guessed:

    input:  "wav"    float32  [batch, 64000]   (batch dynamic, time FIXED)
    output: "logits" float32  [batch, 2]

64000 samples @ 16 kHz = exactly 4.0 seconds -- this matches
sara_wav2vec2's own window_seconds/window_samples exactly (configs/models.yaml).

BUT: the ONNX graph only contains the neural network. The mirror's own
export script (trt_wav2vec2_small_antideepfake.py, commit 51b0d36, inspected
directly) states in its own docstring that "all preprocessing already lives
in the original `score_batch()`" -- and that function is NOT exported to
ONNX and NOT published anywhere this project could find:

  - trt_wav2vec2_small_antideepfake.py imports it from an external
    `wav2vec2_small_antideepfake` module / `AntiSpoofingModel` base class
    that is not a file in this HF model repo.
  - The Arena's own application code (HF Space
    SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena, `git`/API file
    listing inspected directly) contains only the leaderboard web app
    (app.py, badges.py, leaderboard.py, ...) -- no model-scoring framework
    code, no `score_batch`, no `wav2vec2_small_antideepfake.py`.
  - No GitHub repository for this "Arena" scoring framework could be found.

Consequently the exact crop/pad-to-64000-samples strategy (how score_batch
turns an arbitrary-length input into the model's fixed 4.0s window --
silence-pad vs. repeat/tile, left/right/center crop, multi-window
aggregation for longer clips, and whether the same per-utterance layer-norm
used by the original PyTorch model is still applied before or after that
crop/pad) is NOT verifiable from any published source. Per CLAUDE.md ("never
invent an architecture merely to make a checkpoint load" -- the same
principle applies to preprocessing) and the project owner's explicit
instruction ("do not proceed to scientific evaluation using guessed
preprocessing"), this adapter does NOT guess that logic.

`load()` therefore fully succeeds (downloading + running the ONNX graph is
real, verified, working CPU inference). `predict()` deliberately raises
instead of fabricating a detection result from unverified preprocessing.
Only a raw, clearly-non-scientific technical probe (`run_raw_window`) is
provided, for CPU benchmarking purposes only -- see docs/replacement_model_evaluation.md,
"ONNX Rescue Investigation".

Label/class mapping: the mirror's README states "Score = real logit" but
does not state which of the 2 output indices that is. The original NII
checkpoint (candidate_c.py, verified from its own official inference
script) uses index 0 = fake/spoof, index 1 = real/bonafide, and this ONNX
export is stated to be a bit-exact copy of that same checkpoint (identical
safetensors byte size: 380,210,632 bytes). This adapter therefore *assumes*
the same ordering for `model_info()` reporting purposes only, but this has
NOT been independently re-verified against the ONNX graph's own output
(doing so would require running real labeled bonafide/spoof audio through
correctly-preprocessed input, which is exactly the blocked step above) --
this assumption is reported as such, not as a verified fact.
"""

from __future__ import annotations

import time

import numpy as np

from audio_deepfake_detector.config.models_config import ModelConfig
from audio_deepfake_detector.models.base import BaseDeepfakeDetector
from audio_deepfake_detector.utils.datatypes import AudioSample, ModelLoadMetadata, PredictionResult

PREPROCESSING_UNVERIFIED_MESSAGE = (
    "antideepfake_wav2vec2_onnx: the ONNX graph loads and runs correctly on CPU, "
    "but the preprocessing that turns arbitrary-length audio into this model's "
    "required fixed 64000-sample (4.0s) input (crop/pad strategy, multi-window "
    "aggregation for longer clips) is not published anywhere this project could "
    "find -- the mirror's own export script depends on an external, unpublished "
    "`score_batch()` implementation. Per project policy this adapter refuses to "
    "guess that preprocessing and fabricate a detection result. See "
    "docs/replacement_model_evaluation.md, 'ONNX Rescue Investigation'."
)


class CandidateAntiDeepfakeOnnxDetector(BaseDeepfakeDetector):
    def __init__(self, model_config: ModelConfig, device: str = "cpu"):
        if device != "cpu":
            raise ValueError("This project only supports CPU inference.")
        self.model_config = model_config
        self.model_id = model_config.id
        self.repository = model_config.repository
        self.device = device
        self.sample_rate = model_config.sample_rate
        self.window_seconds = model_config.window_seconds
        self._session = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._input_length: int | None = None
        self._label_mapping = model_config.label_mapping or {0: "spoof", 1: "bonafide"}

    def load(self) -> ModelLoadMetadata:
        import onnxruntime as ort
        import torch
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id=self.repository,
            filename=self.model_config.checkpoint_filename,
            revision=self.model_config.revision,
        )

        session_options = ort.SessionOptions()
        self._session = ort.InferenceSession(
            ckpt_path, sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(
                "ONNX Runtime selected a non-CPU provider: "
                f"{self._session.get_providers()}. This project is CPU-only."
            )

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Unexpected ONNX graph shape: {len(inputs)} inputs, {len(outputs)} outputs "
                "(expected exactly 1 each)."
            )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        # inputs[0].shape is like ['batch', 64000]; the fixed (non-symbolic) dim is the input length.
        fixed_dims = [d for d in inputs[0].shape if isinstance(d, int)]
        if len(fixed_dims) != 1:
            raise RuntimeError(
                f"Expected exactly one fixed input dimension, got shape {inputs[0].shape}"
            )
        self._input_length = fixed_dims[0]

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
            transformers_version=None,
            device=self.device,
        )

    def predict(self, audio_sample: AudioSample) -> PredictionResult:
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")
        raise RuntimeError(PREPROCESSING_UNVERIFIED_MESSAGE)

    def run_raw_window(self, waveform_64000_samples: np.ndarray) -> tuple[np.ndarray, float]:
        """CPU benchmarking / smoke-test probe ONLY -- not a scientific prediction.

        Runs a raw [1, input_length] float32 array through the ONNX graph and
        returns (raw_logits, inference_time_ms). The caller is responsible for
        acknowledging this does not use verified preprocessing; results have
        no detection-accuracy meaning. See PREPROCESSING_UNVERIFIED_MESSAGE.
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() before run_raw_window().")
        if waveform_64000_samples.shape != (self._input_length,):
            raise ValueError(
                f"Expected shape ({self._input_length},), got {waveform_64000_samples.shape}"
            )
        x = waveform_64000_samples.astype(np.float32)[None, :]
        start = time.perf_counter()
        (logits,) = self._session.run([self._output_name], {self._input_name: x})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return logits[0], elapsed_ms

    def unload(self) -> None:
        self._session = None
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
            "input_length_samples": self._input_length,
            "label_mapping": self._label_mapping,
            "label_mapping_verified_for_onnx_export": False,
            "device": self.device,
            "loaded": self._session is not None,
            "predict_status": "blocked_unverified_preprocessing",
        }
