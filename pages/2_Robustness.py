"""Robustness Analysis — an OPTIONAL research tool, not part of normal
classification. Re-scores the clip already analyzed on the Analyze page
under a fixed set of degraded conditions (MP3 re-encoding, telephone-band
filtering, additive noise), reusing the exact methodology already used
for this project's frozen robustness experiment. Never runs
automatically; never alters the Spectra classifier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.analysis.robustness import ffmpeg_available, run_robustness_analysis, stability_summary  # noqa: E402
from app.components import render_section_title, setup_page  # noqa: E402
from app.model_loader import DEPLOYMENT_MODEL_ID, get_detector  # noqa: E402

setup_page("Robustness")

st.markdown('<div class="adf-headline" style="font-size:1.7rem;">Robustness Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="adf-subtext">Re-analyzes your most recently uploaded clip under common signal '
    "degradations (compression, telephone-band filtering, additive noise) to observe how "
    "stable the presentation result is. This is a research tool, not part of the standard "
    "analysis workflow.</div>",
    unsafe_allow_html=True,
)

audio_sample = st.session_state.get("last_audio_sample")
last_result = st.session_state.get("last_result")

if audio_sample is None or last_result is None:
    st.info("Analyze a clip on the **Analyze** page first, then return here to run robustness analysis on it.")
    st.stop()

st.caption(
    f"Using the most recently analyzed clip ({audio_sample.duration_seconds:.2f}s, "
    f"{audio_sample.sample_rate} Hz)."
)

if not ffmpeg_available():
    st.caption("Note: ffmpeg was not found — MP3 conditions will be skipped; other conditions still run.")

st.caption(
    "This may take longer than a standard analysis (each condition re-runs a full inference "
    "pass). Conditions are processed one at a time; no degraded audio is stored."
)

if st.button("Run robustness analysis", type="primary"):
    detector = get_detector(DEPLOYMENT_MODEL_ID)
    threshold = detector.model_info().get("calibrated_threshold_spoof_probability")
    with st.spinner("Running robustness analysis..."):
        results = run_robustness_analysis(detector, audio_sample, threshold)
    st.session_state["robustness_results"] = results

results = st.session_state.get("robustness_results")
if results:
    render_section_title("Results")
    rows = []
    for r in results:
        delta = "—" if r.delta_pp_from_original is None else f"{r.delta_pp_from_original:+.1f} pp"
        rows.append(
            {
                "Condition": r.label,
                "Spoof probability": f"{r.spoof_probability * 100:.1f}%",
                "Presentation result": r.presentation_state,
                "Difference from original": delta,
            }
        )
    st.dataframe(rows, width='stretch', hide_index=True)
    st.caption(f"Result stability: {stability_summary(results)}")
