"""AI Voice Deepfake Detector — Streamlit entrypoint (Analyze page).

Presentation layer only. All preprocessing/model/inference logic lives
under src/audio_deepfake_detector; this file (and the helpers under app/)
never implement model logic directly. Visual design lives in
app/styles.py (tokens + CSS) and app/components.py (reusable markup).
Supplementary, descriptive analysis (audio quality, segment evidence,
evidence summary, robustness, reporting) lives under app/analysis/ and
app/reporting/ -- see docs/analysis_platform.md for the full
architecture and the explicit statement that none of it alters the
frozen Spectra-AASIST3 classification.

Run locally with:
    .\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.analysis.audio_quality import assess_analysis_suitability, compute_audio_quality  # noqa: E402
from app.analysis.evidence import (  # noqa: E402
    SEGMENT_DURATION_SECONDS,
    compute_segment_agreement,
    evidence_summary_sentences,
    segment_table,
)
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
from app.reporting.report import build_report_data, render_html_report, render_json_report  # noqa: E402
from app.styles import inject_global_styles  # noqa: E402
from app.validation import MAX_DURATION_SECONDS, MAX_FILE_SIZE_BYTES, validate_and_load_upload  # noqa: E402
from app.visualizations import plot_mel_spectrogram, plot_segment_timeline, plot_waveform  # noqa: E402
from audio_deepfake_detector.config.models_config import load_models_config  # noqa: E402

logger = logging.getLogger("audio_deepfake_detector.streamlit_app")

st.set_page_config(
    page_title="AI Voice Analysis",
    page_icon=":studio_microphone:",
    layout="wide",
)
st.markdown(inject_global_styles(), unsafe_allow_html=True)


def _render_technical_details(result, model_config, model_info: dict, n_segments: int) -> None:
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
            ("Segments analyzed", str(n_segments)),
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


def _render_recommended_input() -> None:
    with st.expander("Recommended input"):
        st.markdown(
            """
For more interpretable results:
- Use speech-dominant recordings.
- Prefer the original recording where available, rather than a re-shared copy.
- Avoid excessive background music.
- Avoid very short samples.
- Avoid heavily degraded or repeatedly re-compressed copies where possible.
- Keep audio within the current 30-second application limit.

Following these does not guarantee a particular result — it improves the conditions
under which the detector was evaluated.
"""
        )


def _render_upload_privacy_note() -> None:
    st.caption("Audio is analyzed for this session and is not permanently stored.")


def _render_error(message: str) -> None:
    st.error(message, icon=":material/error:")


def _render_audio_quality_section(audio_sample, file_size_bytes: int) -> tuple:
    quality = compute_audio_quality(audio_sample.waveform, audio_sample.sample_rate, file_size_bytes, audio_sample.duration_seconds)
    suitability = assess_analysis_suitability(quality)

    render_section_title("Analysis conditions")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Peak amplitude", f"{quality.peak_amplitude:.2f}")
    q2.metric("RMS level", f"{quality.rms_level:.3f}")
    q3.metric("Silence ratio", f"{quality.silence_ratio * 100:.0f}%")
    q4.metric("Clipping", f"{quality.clipping_ratio * 100:.1f}%")

    if suitability.level == "Good":
        st.caption("Analysis conditions: **Good**")
    else:
        reason_text = " and ".join(suitability.reasons)
        icon = ":material/warning:" if suitability.level == "Limited" else ":material/error:"
        st.warning(
            f"**{suitability.level} analysis conditions.** This recording contains {reason_text}. "
            "Detection results should therefore be interpreted cautiously.",
            icon=icon,
        )
    st.caption(
        "This assessment is a descriptive signal-quality check and does not influence the "
        "detector's classification, threshold, or presentation state."
    )
    return quality, suitability


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
        _render_recommended_input()
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
                segment_probs = None
                if audio_sample.duration_seconds > SEGMENT_DURATION_SECONDS:
                    status.update(label="Combining segment results...")
                    full_clip = detector.predict_full_clip(audio_sample, aggregation="mean")
                    segment_probs = full_clip["window_spoof_probs"]
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

            status.update(label="Preparing visual analysis...")
            status.update(label="Analysis complete", state="complete")

        st.session_state["last_result"] = result
        st.session_state["last_result_file_key"] = file_key
        st.session_state["last_segment_probs"] = segment_probs
        st.session_state["last_audio_sample"] = audio_sample
        st.session_state["last_filename"] = uploaded_file.name
        st.session_state["last_file_bytes_sha256"] = file_key
        st.session_state.pop("robustness_results", None)

    result = (
        st.session_state.get("last_result")
        if st.session_state.get("last_result_file_key") == file_key
        else None
    )
    if result is not None:
        segment_probs = st.session_state.get("last_segment_probs")
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

        n_segments = len(segment_probs) if segment_probs else 1
        render_metrics_row(
            [
                (f"{result.audio_duration_seconds:.1f} sec", "Audio duration"),
                (str(n_segments), "Segments analyzed"),
                (f"{result.inference_time_ms / 1000:.1f} sec", "Analysis time"),
                ("CPU / ONNX", "Runtime"),
            ]
        )

        agreement = compute_segment_agreement(segment_probs) if segment_probs else None
        with st.expander("Why this result?"):
            for sentence in evidence_summary_sentences(state, result.probabilities["spoof"], calibrated_threshold, agreement):
                st.markdown(f"- {sentence}")

        quality, suitability = _render_audio_quality_section(audio_sample, len(file_bytes))

        if segment_probs:
            st.divider()
            render_section_title(
                "Segment evidence",
                f"{len(segment_probs)} non-overlapping ~{SEGMENT_DURATION_SECONDS:.2f}s segments analyzed as a "
                "project-level extension (not used for the presented decision above).",
            )
            if agreement is not None:
                st.caption(
                    f"Segment agreement: **{agreement.level}** — {max(agreement.n_spoof_leaning, agreement.n_bonafide_leaning)} "
                    f"of {agreement.n_segments} analyzed segments leaned {agreement.dominant_direction} "
                    f"({agreement.agreement_percent:.0f}% agreement)."
                )
            timeline_fig = plot_segment_timeline(segment_probs, SEGMENT_DURATION_SECONDS, calibrated_threshold)
            st.pyplot(timeline_fig, clear_figure=True)
            st.caption(
                "Segments are sequential and non-overlapping. Each segment's interpretation applies the same "
                "calibrated threshold used for the overall decision, for descriptive purposes only — this is not "
                "an independently validated per-segment threshold."
            )
            with st.expander("View segment-level detail"):
                rows = segment_table(segment_probs, calibrated_threshold, audio_sample.duration_seconds)
                display_rows = [
                    {
                        "Segment": r["segment"],
                        "Time range": f"{r['start_seconds']:.2f}s – {r['end_seconds']:.2f}s",
                        "Bonafide probability": f"{r['bonafide_probability'] * 100:.1f}%",
                        "Spoof probability": f"{r['spoof_probability'] * 100:.1f}%",
                        "Interpretation": r["interpretation"],
                    }
                    for r in rows
                ]
                st.dataframe(display_rows, width='stretch', hide_index=True)
            st.page_link("pages/2_Robustness.py", label="Run robustness analysis on this clip →")

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
        _render_technical_details(result, model_config, model_info, n_segments)

        st.divider()
        render_section_title("Download analysis report")
        generated_at = datetime.now()
        report_rows = segment_table(segment_probs, calibrated_threshold, audio_sample.duration_seconds) if segment_probs else []
        report_data = build_report_data(
            generated_at=generated_at,
            filename=uploaded_file.name,
            file_sha256=file_key,
            duration_seconds=audio_sample.duration_seconds,
            sample_rate=audio_sample.sample_rate,
            audio_format=Path(uploaded_file.name).suffix.lstrip(".").upper(),
            channels=1,
            presentation_state=state,
            prediction_label=summary["prediction"],
            bonafide_probability=result.probabilities["bonafide"],
            spoof_probability=result.probabilities["spoof"],
            calibrated_threshold=calibrated_threshold,
            binary_model_decision=result.binary_model_decision or result.raw_label,
            n_segments=n_segments,
            segment_rows=report_rows,
            segment_agreement=agreement.__dict__ if agreement else None,
            audio_quality=quality.__dict__,
            suitability_level=suitability.level,
            model_info=model_info | {"repository": result.model_repository, "revision": model_config.revision},
            inference_time_ms=result.inference_time_ms,
        )
        report_col1, report_col2 = st.columns(2)
        report_col1.download_button(
            "Download HTML report",
            data=render_html_report(report_data),
            file_name=f"{report_data['report_id']}.html",
            mime="text/html",
        )
        report_col2.download_button(
            "Download JSON export",
            data=render_json_report(report_data),
            file_name=f"{report_data['report_id']}.json",
            mime="application/json",
        )
        st.caption(f"Report ID: {report_data['report_id']} — a local reproducibility identifier, not a database reference.")

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
