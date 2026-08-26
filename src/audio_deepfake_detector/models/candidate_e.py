"""Adapter for lab260/Spectra-AASIST3 ("spectra_aasist3_onnx").

Architecture: wav2vec 2.0 XLS-R-300m front-end (standard `transformers`
`Wav2Vec2Model`, NOT fairseq) -> MLP bridge -> KAN-enhanced AASIST back-end.
Verified directly from the model's own vendored, self-contained `model.py`
(no external dependency beyond `transformers`/`torch`/`huggingface_hub`).
Pre-release/unpublished model -- no peer-reviewed paper; appears in the
Speech Anti-Spoofing Arena's "Unpublished/Proprietary" tier (listed,
unranked). Author-reported metrics (README badges, .eval_results/*) are
maintainer-reported, not project-measured, and must never be presented as
this project's own results.

STATUS: preprocessing is DOCUMENTED (unlike antideepfake_wav2vec2_onnx,
candidate_d.py) but not shipped as runnable code in this repo. The model
card README (lab260/Spectra-AASIST3, verified directly, not via WebFetch
summarization) states in prose:

    "Preemphasis (0.97) is applied to the full waveform ... then a
    deterministic first-64,600-sample window (~4.04s; tile-repeat if
    shorter -- no random crop). No resampling in the wrapper (audio
    arrives at expected_sample_rate=16000). ... the bona-fide logit
    (index 1) is the score."

The README also links to `spectra_aasist3.py`/`spectra_aasist3_net.py` as
"the exact wrapper that produced the Arena scores" -- but neither file
exists in this repository (confirmed: both resolve HTTP 404). Only the
vendored network (`model.py`, containing `SpectraAASIST3`/`KANAASIST`) is
actually present. The exported ONNX graph (`spectra-aasist3.onnx`) confirms
this network requires a fixed [batch, 64600] float32 input -- consistent
with the README's documented preprocessing, verified directly from the
graph, not guessed.

Consequently, unlike candidate_d.py (antideepfake_wav2vec2_onnx), this
adapter DOES implement predict() -- because the preprocessing recipe is
adequately specified in prose from the model's own official card, not
merely implied. One convention is not explicitly spelled out and is
recorded here as an assumption, not a verified fact:

  - Pre-emphasis first-sample edge handling: this adapter uses the
    universal DSP convention y[0] = x[0], y[n] = x[n] - 0.97*x[n-1] for
    n >= 1 (no wrap-around). This is standard across speech-processing
    toolkits (Kaldi, ESPnet, etc.) for coefficient 0.97 and is not itself
    an unusual choice, but the README does not spell out this exact edge
    case, so it is flagged rather than silently assumed to be "obviously"
    correct.

Score direction (verified from model.py, not assumed): `classify()` reads
`self.forward(x)[:, 1]` (index 1) and returns `(x > threshold).float()`
with a baked-in `threshold=-1.0625009` default -- confirming index 1 is
the bona-fide logit and higher-is-more-bonafide, directly from source, not
just README prose. This adapter reports the raw bona-fide logit (index 1)
plus a softmax-derived spoof probability (index 0) for threshold work
reusing this project's existing `evaluation/metrics.py` (which expects a
[0, 1] "spoof score", higher = more spoof).
"""

from __future__ import annotations

import time

import numpy as np

from audio_deepfake_detector.config.models_config import ModelConfig
from audio_deepfake_detector.models.base import BaseDeepfakeDetector
from audio_deepfake_detector.utils.datatypes import (
    AudioSample,
    ModelLoadMetadata,
    PredictionResult,
    WindowPrediction,
)

PREEMPHASIS_COEFF = 0.97
REQUIRED_SAMPLES = 64600
AUTHOR_DEFAULT_THRESHOLD = -1.0625009  # from model.py SpectraAASIST3.classify(), NOT independently calibrated by this project

# Project-calibrated operating thresholds (spoof-probability convention, higher = spoof),
# frozen on the calibration subset in results/metrics/spectra_aasist3_split.json (seed=2024),
# NEVER selected on the evaluation set. See docs/spectra_production_optimization.md.
FP32_CALIBRATED_THRESHOLD = 0.9299831390380859
INT8_DYNAMIC_CALIBRATED_THRESHOLD = 0.939693808555603


def apply_preemphasis(waveform: np.ndarray, coeff: float = PREEMPHASIS_COEFF) -> np.ndarray:
    """y[0] = x[0], y[n] = x[n] - coeff*x[n-1] for n >= 1.

    Standard DSP convention (first-sample edge case not explicitly spelled
    out in the model card -- see module docstring).
    """
    x = np.asarray(waveform, dtype=np.float32)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coeff * x[:-1]
    return y


def window_to_required_length(waveform: np.ndarray, required: int = REQUIRED_SAMPLES) -> np.ndarray:
    """Deterministic first-`required`-sample crop for clips >= required
    samples; tile-repeat (np.tile, no random crop) then truncate for
    shorter clips. Both behaviors are stated explicitly in the model card."""
    x = np.asarray(waveform, dtype=np.float32)
    if x.size >= required:
        return x[:required]
    if x.size == 0:
        return np.zeros(required, dtype=np.float32)
    n_tiles = -(-required // x.size)  # ceil
    return np.tile(x, n_tiles)[:required]


def author_compatible_preprocess(waveform: np.ndarray) -> np.ndarray:
    """Full pipeline in the documented order: preemphasis on the FULL
    waveform first, then crop/pad to REQUIRED_SAMPLES -- per the README's
    own ordering ("Preemphasis ... is applied to the full waveform ...
    then a deterministic first-64,600-sample window")."""
    emphasized = apply_preemphasis(waveform)
    return window_to_required_length(emphasized)


def make_sequential_windows(waveform: np.ndarray, window: int = REQUIRED_SAMPLES) -> list[np.ndarray]:
    """OUR OWN application-level extension (Step 18) -- NOT attributed to
    the model authors. Non-overlapping sequential windows over the
    (already preemphasized) full clip, tile-repeat-padding the final
    partial window. Used only for the full-clip aggregation experiment,
    never for the author-compatible benchmark score."""
    emphasized = apply_preemphasis(waveform)
    if emphasized.size <= window:
        return [window_to_required_length(emphasized)]
    windows = []
    for start in range(0, emphasized.size, window):
        chunk = emphasized[start : start + window]
        if chunk.size < window:
            chunk = window_to_required_length(chunk)
        windows.append(chunk)
    return windows


def softmax_spoof_bonafide(logits: np.ndarray) -> tuple[float, float]:
    """logits: shape (2,), index 0 assumed spoof, index 1 = bonafide
    (verified from model.py's classify()). Returns (spoof_prob, bonafide_prob)."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    return float(probs[0]), float(probs[1])


class CandidateSpectraAasist3OnnxDetector(BaseDeepfakeDetector):
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
        # Calibrated decision threshold (spoof-probability convention). Defaults to the
        # FP32 calibration result; callers score with INT8_DYNAMIC_CALIBRATED_THRESHOLD
        # explicitly when using the quantized artifact (see production adapter notes in
        # docs/spectra_production_optimization.md -- there is currently no single
        # "quantization" field on ModelConfig, so the caller is responsible for passing
        # the threshold matching whichever artifact `local_onnx_path` points at).
        self.threshold = FP32_CALIBRATED_THRESHOLD

    def load(self, local_onnx_path: str | None = None) -> ModelLoadMetadata:
        """`local_onnx_path`: load a local ONNX file (e.g. a quantized
        variant produced by scripts/run_spectra_int8_evaluation.py) instead
        of downloading `checkpoint_filename` from the Hub. Used for the
        INT8 production-optimization experiment (docs/spectra_production_optimization.md);
        the INT8 artifact is not currently hosted anywhere and must not be
        committed to this repository (see Step 17 of that document) -- this
        parameter exists so the same adapter class can be pointed at it
        locally without inventing a second adapter class."""
        import onnxruntime as ort
        import torch

        if local_onnx_path is not None:
            ckpt_path = local_onnx_path
        else:
            from huggingface_hub import hf_hub_download

            ckpt_path = hf_hub_download(
                repo_id=self.repository,
                filename=self.model_config.checkpoint_filename,
                revision=self.model_config.revision,
            )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 2  # see docs/spectra_production_optimization.md Step 11
        self._session = ort.InferenceSession(
            ckpt_path, sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"ONNX Runtime selected a non-CPU provider: {self._session.get_providers()}. "
                "This project is CPU-only."
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
        fixed_dims = [d for d in inputs[0].shape if isinstance(d, int)]
        if len(fixed_dims) != 1:
            raise RuntimeError(f"Expected exactly one fixed input dimension, got shape {inputs[0].shape}")
        self._input_length = fixed_dims[0]
        if self._input_length != REQUIRED_SAMPLES:
            raise RuntimeError(
                f"ONNX graph input length ({self._input_length}) does not match the documented "
                f"required length ({REQUIRED_SAMPLES}) -- refusing to proceed with a mismatched "
                "preprocessing assumption."
            )

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

    def _raw_logits(self, window_64600: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")
        x = window_64600.astype(np.float32)[None, :]
        (logits,) = self._session.run([self._output_name], {self._input_name: x})
        return logits[0]

    def predict(self, audio_sample: AudioSample) -> PredictionResult:
        """Author-compatible scoring (Step 13): preemphasis on the full
        clip, then a single deterministic first-64,600-sample window
        (tile-repeat if shorter). This is the primary formal benchmark
        path -- NOT this project's own multi-window extension (see
        predict_full_clip below)."""
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        window = author_compatible_preprocess(audio_sample.waveform)

        start = time.perf_counter()
        logits = self._raw_logits(window)
        inference_time_ms = (time.perf_counter() - start) * 1000.0

        spoof_prob, bonafide_prob = softmax_spoof_bonafide(logits)
        prob_dict = {"spoof": spoof_prob, "bonafide": bonafide_prob}
        raw_label = "spoof" if spoof_prob >= bonafide_prob else "bonafide"

        window_predictions = [
            WindowPrediction(
                window_index=0,
                start_sample=0,
                end_sample=min(len(audio_sample.waveform), REQUIRED_SAMPLES),
                raw_label=raw_label,
                probabilities=prob_dict,
            )
        ]

        return PredictionResult(
            raw_label=raw_label,
            normalized_label="SPOOF" if raw_label == "spoof" else "BONAFIDE",
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

    def predict_full_clip(self, audio_sample: AudioSample, aggregation: str = "mean") -> dict:
        """OUR OWN application-level extension (Step 18) -- sequential
        non-overlapping 64,600-sample windows over the full (preemphasized)
        clip, aggregated by `aggregation` ('first', 'mean', 'median',
        'majority'). NOT attributed to the model authors; not used for the
        author-compatible benchmark (predict())."""
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() before predict_full_clip().")
        windows = make_sequential_windows(audio_sample.waveform)
        bonafide_logits = []
        spoof_probs = []
        for w in windows:
            logits = self._raw_logits(w)
            spoof_prob, _ = softmax_spoof_bonafide(logits)
            bonafide_logits.append(float(logits[1]))
            spoof_probs.append(spoof_prob)

        if aggregation == "first":
            agg_spoof_prob = spoof_probs[0]
        elif aggregation == "mean":
            agg_spoof_prob = float(np.mean(spoof_probs))
        elif aggregation == "median":
            agg_spoof_prob = float(np.median(spoof_probs))
        elif aggregation == "majority":
            agg_spoof_prob = float(np.mean([1.0 if p >= 0.5 else 0.0 for p in spoof_probs]))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return {
            "n_windows": len(windows),
            "window_spoof_probs": spoof_probs,
            "window_bonafide_logits": bonafide_logits,
            "aggregation": aggregation,
            "aggregated_spoof_prob": agg_spoof_prob,
        }

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
            "device": self.device,
            "loaded": self._session is not None,
            "author_default_threshold_on_bonafide_logit": AUTHOR_DEFAULT_THRESHOLD,
            "unpublished_model": True,
        }
