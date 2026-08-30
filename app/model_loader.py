"""Lazy, cached model loading for the Streamlit layer.

This module owns the ONLY st.cache_resource-decorated loader in the app,
so the model checkpoint is loaded at most once per running app process,
and only when analysis is first requested - never merely because someone
visits the page.

Delegates entirely to audio_deepfake_detector.models.registry; no model
logic is duplicated here.

EXPERIMENTAL CANDIDATE BRANCH (feat/spectra-streamlit-candidate):
DEPLOYMENT_MODEL_ID points at spectra_aasist3_onnx_int8 (ONNX Runtime CPU
only -- no torch/transformers import anywhere in this path, see
docs/spectra_streamlit_candidate.md) instead of the currently-live
sara_wav2vec2. sara_wav2vec2 remains fully present and importable via the
registry (research evidence, docs/inference_calibration.md,
docs/spectra_aasist3_evaluation.md) -- this only changes which model_id
THIS branch's Streamlit app loads by default; it is never imported unless
explicitly requested via create_detector("sara_wav2vec2", ...). main is
unaffected by this change.
"""

from __future__ import annotations

import streamlit as st

from audio_deepfake_detector.models.base import BaseDeepfakeDetector
from audio_deepfake_detector.models.registry import create_detector

DEPLOYMENT_MODEL_ID = "spectra_aasist3_onnx_int8"


@st.cache_resource(show_spinner=False)
def get_detector(model_id: str = DEPLOYMENT_MODEL_ID) -> BaseDeepfakeDetector:
    """Return a loaded detector, cached for the lifetime of the app process.

    Streamlit's cache_resource ensures this only runs once per (model_id)
    key across reruns/sessions in the same server process - repeated calls
    with the same model_id return the same in-memory object instead of
    reloading the checkpoint.
    """
    detector = create_detector(model_id, device="cpu")
    detector.load()
    return detector
