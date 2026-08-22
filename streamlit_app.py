"""AI Voice Deepfake Detector — Streamlit entrypoint.

Presentation layer only. All preprocessing/model/inference logic lives
under src/audio_deepfake_detector; this file (and the helpers under app/)
never implement model logic directly.

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

from app.errors import UserFacingError  # noqa: E402
from app.formatting import result_summary, window_table_rows  # noqa: E402
from app.model_loader import DEPLOYMENT_MODEL_ID, get_detector  # noqa: E402
from app.validation import MAX_DURATION_SECONDS, MAX_FILE_SIZE_BYTES, validate_and_load_upload  # noqa: E402
from app.visualizations import plot_mel_spectrogram, plot_waveform  # noqa: E402
from audio_deepfake_detector.config.models_config import load_models_config  # noqa: E402

logger = logging.getLogger("audio_deepfake_detector.streamlit_app")

st.set_page_config(
    page_title="AI Voice Deepfake Detector",
    page_icon=":studio_microphone:",
    layout="wide",
)


def _render_header() -> None:
    st.title("AI Voice Deepfake Detector")
    st.caption(
        "Detect potentially AI-generated or cloned speech using a pretrained "
        "Wav2Vec2 anti-spoofing model."
    )
    st.markdown(
        "Final-year Computer Science project — *Detecting AI-Generated and "
        "Cloned Voices: A Deep Learning System for Robust Audio Deepfake "
        "Detection*."
    )


def _render_disclaimer() -> None:
    st.info(
        "**Research prototype.** This system is a research/educational "
        "prototype and should not be treated as forensic evidence or used "
        "as the sole basis for legal, security, disciplinary, or identity "
        "decisions. Modern speech synthesis systems continue to evolve, and "
        "detection performance can vary across recording conditions, "
        "codecs, languages, and generation methods.",
        icon=":material/info:",
    )


def _render_why_wav2vec2() -> None:
    with st.expander("Why Wav2Vec2?"):
        st.markdown(
            """
Wav2Vec2 is a self-supervised speech model that learns representations
directly from raw waveform audio, rather than from hand-crafted spectral
features. This project uses a checkpoint that adapts a pretrained
Wav2Vec2 backbone for a bonafide-vs-spoof classification task.

The system uses an existing **pretrained** model rather than training a
speech foundation model from scratch — that would require substantially
more data and compute than is practical for this project. Later phases of
this project will independently evaluate how well this kind of detector
generalises to unseen synthesis methods and how robust it is to
real-world audio degradation.
                """
        )


def _render_technical_details(result, model_config) -> None:
    with st.expander("Technical details"):
        st.markdown(
            f"""
| | |
|---|---|
| Model | Wav2Vec2 Audio Deepfake Detector |
| Repository | `{result.model_repository}` |
| Base architecture | `facebook/wav2vec2-base` |
| Input sample rate | 16 kHz |
| Window | 4 seconds |
| Device | {result.device.upper()} |
| Windows analyzed | {result.windows_analyzed} |
| Audio duration | {result.audio_duration_seconds:.2f} s |
| Inference time | {result.inference_time_ms:.0f} ms |
| Model revision | `{model_config.revision}` |
            """
        )
        st.caption(
            "The Wav2Vec2 detector operates on the audio waveform directly. "
            "The spectrogram shown above is provided as a visual "
            "representation for the user, not as model input."
        )
        st.caption(
            "Aggregation method: when a clip spans more than one 4-second "
            "window, the clip-level result is the mean spoof probability "
            "across all windows. This is a project-level aggregation "
            "choice for this MVP, not a method scientifically validated by "
            "the original model authors."
        )


def _render_window_table(result) -> None:
    if result.windows_analyzed <= 1:
        return
    st.caption(f"Windows analyzed: {result.windows_analyzed}")
    with st.expander("Window-level analysis"):
        st.dataframe(window_table_rows(result), use_container_width=True, hide_index=True)


def main() -> None:
    _render_header()

    config = load_models_config()
    model_config = config.get(DEPLOYMENT_MODEL_ID)

    uploaded_file = st.file_uploader(
        "Upload an audio clip (WAV, MP3, or FLAC — up to "
        f"{MAX_DURATION_SECONDS:.0f} seconds, {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB max)",
        type=["wav", "mp3", "flac"],
    )

    if uploaded_file is None:
        _render_why_wav2vec2()
        _render_disclaimer()
        return

    file_bytes = uploaded_file.getvalue()
    file_key = hashlib.sha256(file_bytes).hexdigest()

    try:
        audio_sample = validate_and_load_upload(file_bytes, uploaded_file.name)
    except UserFacingError as exc:
        st.error(exc.friendly_message)
        if exc.technical_detail:
            logger.warning("Upload validation failed: %s", exc.technical_detail)
        _render_why_wav2vec2()
        _render_disclaimer()
        return

    st.subheader("Uploaded audio")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Filename", uploaded_file.name)
    info_col2.metric("Duration", f"{audio_sample.duration_seconds:.2f} s")
    info_col3.metric("Sample rate", f"{audio_sample.sample_rate} Hz")
    st.audio(file_bytes)

    analyze_clicked = st.button("Analyze Audio", type="primary")

    if analyze_clicked:
        try:
            with st.spinner(
                "Loading the detection model. The first analysis may take "
                "longer while the model is prepared."
            ):
                detector = get_detector(DEPLOYMENT_MODEL_ID)

            with st.spinner("Analyzing audio..."):
                result = detector.predict(audio_sample)
        except Exception as exc:  # noqa: BLE001 - convert to friendly UI message
            logger.exception("Inference failed")
            st.error(
                "Something went wrong while analyzing this audio. Please try "
                "again, or try a different file. If the problem persists, "
                "the detection model may be temporarily unavailable."
            )
            _render_disclaimer()
            return

        st.session_state["last_result"] = result
        st.session_state["last_result_file_key"] = file_key

    result = (
        st.session_state.get("last_result")
        if st.session_state.get("last_result_file_key") == file_key
        else None
    )
    if result is not None:
        summary = result_summary(result)
        st.subheader("Detection result")
        col1, col2 = st.columns([2, 1])
        with col1:
            if result.normalized_label == "SPOOF":
                st.warning(f"**Prediction:** {summary['prediction']}")
            else:
                st.success(f"**Prediction:** {summary['prediction']}")
            st.caption(
                "The model classifies this audio as "
                + (
                    "likely AI-generated/spoofed."
                    if result.normalized_label == "SPOOF"
                    else "likely bonafide."
                )
            )
        with col2:
            st.metric("Model confidence", summary["confidence"])

        prob_col1, prob_col2 = st.columns(2)
        prob_col1.metric("Bonafide probability", summary["bonafide_probability"])
        prob_col2.metric("Spoof probability", summary["spoof_probability"])

        _render_window_table(result)

        st.subheader("Waveform")
        waveform_fig = plot_waveform(audio_sample.waveform, audio_sample.sample_rate)
        st.pyplot(waveform_fig, clear_figure=True)

        st.subheader("Mel spectrogram")
        spectrogram_fig = plot_mel_spectrogram(audio_sample.waveform, audio_sample.sample_rate)
        st.pyplot(spectrogram_fig, clear_figure=True)

        _render_technical_details(result, model_config)

    _render_why_wav2vec2()
    _render_disclaimer()


if __name__ == "__main__":
    main()
