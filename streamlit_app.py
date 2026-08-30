"""AI Voice Deepfake Detector — Streamlit entrypoint.

Presentation layer only. All preprocessing/model/inference logic lives
under src/audio_deepfake_detector; this file (and the helpers under app/)
never implement model logic directly. Visual design lives in
app/styles.py (tokens + CSS) and app/components.py (reusable markup) --
see docs/ui_ux_design.md for the full design system and rationale.

Run locally with:
    .\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.components import (  # noqa: E402
    probability_comparison_html,
    render_disclaimer,
    render_evaluation_section,
    render_footer,
    render_hero,
    render_how_it_works,
    render_metrics_row,
    render_page_header,
    render_result_panel,
    render_section_title,
    render_why_model,
    threshold_visualization_html,
)
from app.errors import UserFacingError  # noqa: E402
from app.formatting import result_summary  # noqa: E402
from app.model_loader import DEPLOYMENT_MODEL_ID, get_detector  # noqa: E402
from app.styles import inject_global_styles  # noqa: E402
from app.validation import MAX_DURATION_SECONDS, MAX_FILE_SIZE_BYTES, validate_and_load_upload  # noqa: E402
from app.visualizations import plot_mel_spectrogram, plot_waveform  # noqa: E402
from audio_deepfake_detector.config.models_config import load_models_config  # noqa: E402

logger = logging.getLogger("audio_deepfake_detector.streamlit_app")

st.set_page_config(
    page_title="AI Voice Analysis",
    page_icon=":studio_microphone:",
    layout="wide",
)
st.markdown(inject_global_styles(), unsafe_allow_html=True)


def _render_technical_details(result, model_config, model_info: dict) -> None:
    """Built entirely from the ACTIVE detector's model_info() + model_config
    -- no model-specific string literals here, so this renders correctly
    for whichever model_info dict DEPLOYMENT_MODEL_ID actually resolves to."""
    display_name = model_info.get("display_name", model_info.get("model_id", "Unknown model"))
    architecture_short = model_info.get("architecture_short", model_config.architecture)
    runtime = model_info.get("runtime", "Unknown runtime")
    sample_rate = model_info.get("sample_rate", 16000)
    native_window = model_info.get("native_window_description", f"{model_info.get('window_seconds', '?')} seconds")
    threshold_desc = model_info.get("threshold_description")
    aggregation_desc = model_info.get("aggregation_description")
    preemphasis = model_info.get("preemphasis_coefficient")

    with st.expander("Technical details"):
        rows = [
            ("Model", display_name),
            ("Base architecture", architecture_short),
            ("Runtime", runtime),
            ("Model artifact", f"`{result.model_repository}`"),
            ("Revision", f"`{model_config.revision}`"),
            ("Input sample rate", f"{sample_rate} Hz"),
            ("Native segment length", native_window),
            ("Device", result.device.upper()),
            ("Segments analyzed", str(result.windows_analyzed)),
            ("Audio duration", f"{result.audio_duration_seconds:.2f} s"),
            ("Inference time", f"{result.inference_time_ms:.0f} ms"),
        ]
        if preemphasis is not None:
            rows.append(("Pre-emphasis", str(preemphasis)))
        if threshold_desc is not None:
            rows.append(("Decision threshold", threshold_desc))

        table = "\n".join(f"| {label} | {value} |" for label, value in rows)
        st.markdown(f"| | |\n|---|---|\n{table}")

        st.caption(
            "The detector operates on the audio waveform directly. The spectrogram shown "
            "above is provided for visual inspection and is not the model input."
        )
        if aggregation_desc:
            st.caption(f"Aggregation method: {aggregation_desc}")


def _render_segment_analysis(result) -> None:
    if result.windows_analyzed <= 1:
        return
    from app.formatting import window_table_rows

    rows = window_table_rows(result)
    n_spoof_leaning = sum(1 for r in rows if r["Prediction"].startswith("Likely AI"))
    n_bonafide_leaning = len(rows) - n_spoof_leaning

    render_section_title(
        "Segment analysis",
        f"{len(rows)} segments analyzed &middot; {n_spoof_leaning} spoof-leaning &middot; "
        f"{n_bonafide_leaning} bonafide-leaning",
    )
    with st.expander("View segment-level detail"):
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_upload_privacy_note() -> None:
    st.caption("Audio is analyzed for this session and is not permanently stored.")


def _render_error(message: str) -> None:
    st.error(message, icon=":material/error:")


def main() -> None:
    render_page_header()

    config = load_models_config()
    model_config = config.get(DEPLOYMENT_MODEL_ID)

    uploaded_file = st.file_uploader(
        f"Drop an audio sample here — WAV, MP3, or FLAC, up to "
        f"{MAX_DURATION_SECONDS:.0f} seconds and {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB",
        type=["wav", "mp3", "flac"],
    )

    if uploaded_file is None:
        render_hero()
        st.markdown("")
        _render_upload_privacy_note()
        st.divider()
        render_how_it_works()
        render_why_model()
        render_evaluation_section()
        render_disclaimer()
        render_footer()
        return

    file_bytes = uploaded_file.getvalue()
    file_key = hashlib.sha256(file_bytes).hexdigest()

    try:
        audio_sample = validate_and_load_upload(file_bytes, uploaded_file.name)
    except UserFacingError as exc:
        _render_error(exc.friendly_message)
        if exc.technical_detail:
            logger.warning("Upload validation failed: %s", exc.technical_detail)
        render_disclaimer()
        render_footer()
        return

    _render_upload_privacy_note()

    workspace_col, meta_col = st.columns([2, 1])
    with workspace_col:
        st.audio(file_bytes)
    with meta_col:
        st.markdown(
            f"""
| | |
|---|---|
| Filename | {uploaded_file.name} |
| Duration | {audio_sample.duration_seconds:.2f} s |
| Sample rate | {audio_sample.sample_rate} Hz |
| Format | {Path(uploaded_file.name).suffix.lstrip('.').upper()} |
| File size | {len(file_bytes) / 1024:.0f} KB |
            """
        )

    analyze_clicked = st.button("Analyze Audio", type="primary")

    if analyze_clicked:
        detector = None
        with st.status("Preparing detector...", expanded=False) as status:
            try:
                detector = get_detector(DEPLOYMENT_MODEL_ID)
            except Exception:  # noqa: BLE001 - convert to friendly UI message
                logger.exception("Model preparation failed")
                status.update(label="Detector unavailable", state="error")
                _render_error(
                    "**Detector unavailable.** The analysis model could not be prepared. "
                    "Please try again shortly."
                )
                render_disclaimer()
                render_footer()
                return

            status.update(label="Processing audio...")
            status.update(label="Running anti-spoof analysis...")
            try:
                result = detector.predict(audio_sample)
            except Exception:  # noqa: BLE001 - convert to friendly UI message
                logger.exception("Inference failed")
                status.update(label="Analysis failed", state="error")
                _render_error(
                    "Something went wrong while analyzing this audio. Please try again, "
                    "or try a different file."
                )
                render_disclaimer()
                render_footer()
                return

            status.update(label="Combining segment results...")
            status.update(label="Preparing visual analysis...")
            status.update(label="Analysis complete", state="complete")

        st.session_state["last_result"] = result
        st.session_state["last_result_file_key"] = file_key

    result = (
        st.session_state.get("last_result")
        if st.session_state.get("last_result_file_key") == file_key
        else None
    )
    if result is not None:
        detector_for_display = get_detector(DEPLOYMENT_MODEL_ID)
        model_info = detector_for_display.model_info()
        calibrated_threshold = model_info.get("calibrated_threshold_spoof_probability")
        summary = result_summary(result, calibrated_threshold=calibrated_threshold)
        state = summary["presentation_state"]

        extra_html = probability_comparison_html(result.probabilities["bonafide"], result.probabilities["spoof"])
        if summary["is_inconclusive"] and calibrated_threshold is not None:
            extra_html += threshold_visualization_html(result.probabilities["spoof"], calibrated_threshold)

        st.markdown("")
        render_result_panel(state, summary["prediction"], _explanation_for_state(state), extra_html)

        render_metrics_row(
            [
                (f"{result.audio_duration_seconds:.1f} sec", "Audio duration"),
                (str(result.windows_analyzed), "Segments analyzed"),
                (f"{result.inference_time_ms / 1000:.1f} sec", "Analysis time"),
                ("CPU / ONNX", "Runtime"),
            ]
        )

        _render_segment_analysis(result)

        st.divider()
        render_section_title(
            "Audio characteristics", "Visual representations of the submitted waveform and spectral content."
        )
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.caption("Waveform")
            waveform_fig = plot_waveform(audio_sample.waveform, audio_sample.sample_rate)
            st.pyplot(waveform_fig, clear_figure=True)
        with viz_col2:
            st.caption("Mel spectrogram")
            spectrogram_fig = plot_mel_spectrogram(audio_sample.waveform, audio_sample.sample_rate)
            st.pyplot(spectrogram_fig, clear_figure=True)
        st.caption(
            "The detector operates on waveform audio directly. This spectrogram is shown for "
            "visual inspection and is not the model input."
        )

        st.divider()
        render_how_it_works()
        render_why_model()
        render_evaluation_section()
        _render_technical_details(result, model_config, model_info)

    render_disclaimer()
    render_footer()


def _explanation_for_state(state: str) -> str:
    if state == "SPOOF":
        return (
            "The analyzed speech contains characteristics the detector associates with "
            "synthetic or spoofed audio."
        )
    if state == "INCONCLUSIVE":
        return (
            "The detector found elevated spoof indicators, but the score did not cross the "
            "calibrated spoof threshold."
        )
    return (
        "The analyzed speech is more consistent with genuine human speech under the model's "
        "calibrated operating threshold."
    )


if __name__ == "__main__":
    main()
