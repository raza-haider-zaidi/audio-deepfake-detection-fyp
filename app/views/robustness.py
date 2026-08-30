"""Robustness — an OPTIONAL research tool, not part of normal
classification. Re-scores the clip already analyzed on the Analyze page
(or a freshly uploaded one) under a fixed set of degraded conditions,
reusing the exact methodology already used for this project's frozen
robustness experiment. Never runs automatically; never alters the
Spectra classifier.
"""

from __future__ import annotations

import streamlit as st

from app.analysis.robustness import ffmpeg_available, run_robustness_analysis, stability_summary
from app.components import render_empty_state, render_section_title
from app.errors import UserFacingError
from app.model_loader import DEPLOYMENT_MODEL_ID, get_detector
from app.validation import validate_and_load_upload
from app.visualizations import plot_robustness_comparison


def render() -> None:
    render_section_title(
        "Robustness analysis",
        "Evaluate whether a detection remains stable after controlled audio degradation.",
    )

    audio_sample = st.session_state.get("last_audio_sample")

    if audio_sample is None:
        render_empty_state(
            "Robustness analysis",
            "Analyze a recording on the Analyze page first, or upload one below, to test how its "
            "presentation result behaves under compression, telephone-band filtering, and additive noise.",
        )
        st.caption("Supported experiments: MP3 compression · Telephone bandwidth · Controlled noise")
        uploaded_file = st.file_uploader("Or upload an audio file directly", type=["wav", "mp3", "flac"])
        if uploaded_file is not None:
            try:
                audio_sample = validate_and_load_upload(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state["last_audio_sample"] = audio_sample
                st.rerun()
            except UserFacingError as exc:
                st.error(exc.friendly_message, icon=":material/error:")
        return

    st.caption(f"Using the currently loaded clip ({audio_sample.duration_seconds:.2f}s, {audio_sample.sample_rate} Hz).")

    if not ffmpeg_available():
        st.caption("Note: ffmpeg was not found — MP3 conditions will be skipped; other conditions still run.")

    st.caption(
        "This may take longer than a standard analysis (each condition re-runs a full inference "
        "pass). Conditions are processed one at a time; no degraded audio is stored."
    )

    if st.button("Run robustness analysis", type="primary"):
        detector = get_detector(DEPLOYMENT_MODEL_ID)
        threshold = detector.model_info().get("calibrated_threshold_spoof_probability")
        status_placeholder = st.empty()

        def on_start(_condition, label):
            status_placeholder.info(f"Testing {label}...", icon=":material/science:")

        results = run_robustness_analysis(detector, audio_sample, threshold, on_condition_start=on_start)
        status_placeholder.empty()
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
                    "Result": r.presentation_state,
                    "Spoof probability": f"{r.spoof_probability * 100:.1f}%",
                    "Difference from original": delta,
                    "Processing time": f"{r.processing_time_ms:.0f} ms",
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(f"Result stability: {stability_summary(results)}")

        threshold = get_detector(DEPLOYMENT_MODEL_ID).model_info().get("calibrated_threshold_spoof_probability")
        fig = plot_robustness_comparison([r.label for r in results], [r.spoof_probability for r in results], threshold)
        st.pyplot(fig, clear_figure=True)
