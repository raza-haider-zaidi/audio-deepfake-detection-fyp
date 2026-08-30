"""Methodology — the implemented analysis pipeline, with an explicit
distinction between what is model-native, what is this project's own
presentation/aggregation logic, and what is scientific evaluation logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.components import render_section_title, setup_page, step_flow_html  # noqa: E402

setup_page("Methodology")

st.markdown('<div class="adf-headline" style="font-size:1.7rem;">Methodology</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="adf-subtext">The implemented analysis pipeline, from upload to presented result.</div>',
    unsafe_allow_html=True,
)

PIPELINE_STEPS = [
    ("01", "Upload", "The uploaded file is validated (format, size, duration) and decoded."),
    ("02", "Normalize", "Audio is converted to mono and resampled to 16 kHz."),
    ("03", "Pre-emphasis", "A 0.97-coefficient pre-emphasis filter is applied, matching the detector's documented input contract."),
    ("04", "Segment", "The waveform is split into the detector's required fixed-length input segments."),
    ("05", "Inference", "Spectra-AASIST3 INT8 evaluates each segment via ONNX Runtime on CPU."),
    ("06", "Aggregate", "Segment-level evidence is combined (project-level logic, see below)."),
    ("07", "Decide", "The calibrated spoof-probability threshold is applied to produce a binary decision."),
    ("08", "Present", "The binary decision and score are mapped to Bonafide / Spoof / Inconclusive for display."),
]

st.markdown(step_flow_html(PIPELINE_STEPS), unsafe_allow_html=True)

render_section_title("What is model-native vs. project-implemented")
st.markdown(
    """
| Stage | Origin |
|---|---|
| Pre-emphasis (0.97), 64,600-sample native input | Documented by the model's own source/card |
| Decoding, resampling, validation | This project's presentation layer |
| Multi-segment evidence (Segment Evidence page) | **This project's own application-level extension** — not part of the model authors' methodology |
| Calibrated spoof-probability threshold | This project's own calibration experiment, run on a held-out calibration set |
| Bonafide / Spoof / Inconclusive presentation states | This project's own presentation logic, layered on top of a frozen binary decision |
| Audio quality diagnostics, suitability rating | This project's own descriptive signal-processing calculations — never fed back into the classifier |
"""
)

render_section_title("Scientific evaluation vs. presentation logic")
st.markdown(
    "The Evaluation page's metrics (EER, ROC-AUC, F1, bonafide FPR) were computed using the "
    "SAME binary decision rule described above, on a held-out evaluation set never used to "
    "select the threshold. The Inconclusive presentation state is a supplementary display "
    "concept and was **not** part of that scientific evaluation — see the Evaluation page for "
    "the exact binary-decision metrics."
)
