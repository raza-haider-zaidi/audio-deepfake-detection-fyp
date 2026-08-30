"""Analyze — the primary workflow: upload, metadata preview, analysis,
result, segment evidence, audio diagnostics, visual analysis, and report
download. See docs/analysis_platform_v2.md for the full architecture.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from app.analysis.audio_quality import assess_analysis_suitability, compute_audio_quality, probe_media_metadata
from app.analysis.evidence import (
    SEGMENT_DURATION_SECONDS,
    compute_segment_agreement,
    evidence_summary_sentences,
    segment_table,
)
from app.analysis.session import build_session_entry, get_session_entries, record_analysis
from app.components import (
    probability_comparison_html,
    render_capability_strip,
    render_disclaimer,
    render_empty_state,
    render_evaluation_section,
    render_feature_grid,
    render_footer,
    render_hero,
    render_how_it_works,
    render_metric_cards,
    render_result_panel,
    render_section_title,
    render_why_model,
    threshold_visualization_html,
)
from app.errors import UserFacingError
from app.formatting import result_summary
from app.model_loader import DEPLOYMENT_MODEL_ID, get_detector
from app.reporting.report import build_report_data, render_html_report, render_json_report
from app.validation import MAX_DURATION_SECONDS, MAX_FILE_SIZE_BYTES, validate_and_load_upload
from app.visualizations import plot_mel_spectrogram, plot_segment_timeline, plot_waveform
from audio_deepfake_detector.config.models_config import load_models_config

logger = logging.getLogger("audio_deepfake_detector.streamlit_app")


def _render_error(message: str) -> None:
    st.error(message, icon=":material/error:")


def _explanation_for_state(state: str) -> str:
    if state == "SPOOF":
        return "The analyzed speech contains characteristics the detector associates with synthetic or spoofed audio."
    if state == "INCONCLUSIVE":
        return "The detector found elevated spoof indicators, but the score did not cross the calibrated spoof threshold."
    return "The analyzed speech is more consistent with genuine human speech under the model's calibrated operating threshold."


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


def _render_analysis_session_panel() -> None:
    entries = get_session_entries(st.session_state)
    with st.expander(f"Analysis session ({len(entries)})"):
        st.caption("Session data is temporary and is not a persistent case database.")
        if not entries:
            st.caption("No analyses performed yet in this session.")
            return
        rows = [
            {
                "Time": e["timestamp"],
                "Filename": e["filename"],
                "Result": e["presentation_state"],
                "Spoof %": f"{e['spoof_probability'] * 100:.1f}%",
                "Segments": e["n_segments"],
            }
            for e in entries
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
        col1, col2 = st.columns(2)
        from app.analysis.session import export_session_summary

        col1.download_button(
            "Export session summary",
            data=export_session_summary(entries),
            file_name="analysis_session.json",
            mime="application/json",
        )
        if col2.button("Clear session"):
            from app.analysis.session import clear_session

            clear_session(st.session_state)
            st.rerun()


def _render_audio_metadata_inspector(audio_sample, file_bytes: bytes, filename: str, n_segments: int, model_info: dict) -> None:
    render_section_title("Audio diagnostics", "File, media, analysis, and signal-level information.")

    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        media = probe_media_metadata(tmp_path)
    finally:
        os.unlink(tmp_path)

    quality = compute_audio_quality(audio_sample.waveform, audio_sample.sample_rate, len(file_bytes), audio_sample.duration_seconds)
    suitability = assess_analysis_suitability(quality)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**File**")
        st.markdown(
            f"""
| | |
|---|---|
| Filename | {filename} |
| File size | {len(file_bytes) / 1024:.0f} KB |
| SHA-256 | `{hashlib.sha256(file_bytes).hexdigest()[:16]}…` |
            """
        )
    with col2:
        st.markdown("**Media**")
        st.markdown(
            f"""
| | |
|---|---|
| Container | {media.container} |
| Codec | {media.codec} |
| Bitrate | {media.bitrate_kbps} |
| Duration | {media.duration_seconds} |
            """
        )
    with col3:
        st.markdown("**Analysis**")
        st.markdown(
            f"""
| | |
|---|---|
| Analysis sample rate | {audio_sample.sample_rate} Hz |
| Segments | {n_segments} |
| Native segment length | {model_info.get('native_window_description', 'Not available')} |
            """
        )
    with col4:
        st.markdown("**Signal**")
        st.markdown(
            f"""
| | |
|---|---|
| Peak amplitude | {quality.peak_amplitude:.2f} |
| RMS level | {quality.rms_level:.3f} |
| Silence proportion | {quality.silence_ratio * 100:.0f}% |
| Clipping | {quality.clipping_ratio * 100:.1f}% |
            """
        )

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


def _render_segment_evidence(audio_sample, segment_probs, calibrated_threshold) -> None:
    st.divider()
    render_section_title(
        "Segment evidence",
        f"{len(segment_probs)} non-overlapping ~{SEGMENT_DURATION_SECONDS:.2f}s segments, analyzed as a "
        "project-level extension. This is window-level classification, not manipulation localization.",
    )

    agreement = compute_segment_agreement(segment_probs)
    if agreement is not None:
        st.markdown("**Segment consistency**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Segments", agreement.n_segments)
        c2.metric("Agreement", f"{agreement.agreement_percent:.0f}%")
        c3.metric("Mean spoof %", f"{agreement.score_mean * 100:.1f}%")
        c4.metric("Median spoof %", f"{agreement.score_median * 100:.1f}%")
        c5.metric("Std dev", f"{agreement.score_std:.3f}")
        dominant_count = max(agreement.n_spoof_leaning, agreement.n_bonafide_leaning)
        st.caption(
            f"Segment consistency: **{agreement.level}** — {dominant_count} of {agreement.n_segments} segments "
            f"produced {agreement.dominant_direction}-leaning probabilities."
        )

    timeline_fig = plot_segment_timeline(segment_probs, SEGMENT_DURATION_SECONDS, calibrated_threshold)
    st.pyplot(timeline_fig, clear_figure=True)
    st.caption(
        "Segments are sequential and non-overlapping. Each segment's interpretation applies the same "
        "calibrated threshold used for the overall decision, for descriptive purposes only — this is not "
        "an independently validated per-segment threshold."
    )

    rows = segment_table(segment_probs, calibrated_threshold, audio_sample.duration_seconds)
    with st.expander("View segment-level detail"):
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
        st.dataframe(display_rows, width="stretch", hide_index=True)

    with st.expander("Inspect a specific segment"):
        options = [f"Segment {r['segment']} ({r['start_seconds']:.1f}s–{r['end_seconds']:.1f}s)" for r in rows]
        choice = st.selectbox("Select a segment", options, key="segment_navigator_choice")
        idx = options.index(choice)
        chosen = rows[idx]
        start_sample = int(chosen["start_seconds"] * audio_sample.sample_rate)
        end_sample = int(chosen["end_seconds"] * audio_sample.sample_rate)
        excerpt = audio_sample.waveform[start_sample:end_sample]
        st.caption(
            f"Time range {chosen['start_seconds']:.2f}s–{chosen['end_seconds']:.2f}s · "
            f"Bonafide {chosen['bonafide_probability'] * 100:.1f}% · Spoof {chosen['spoof_probability'] * 100:.1f}% · "
            f"{chosen['interpretation']}"
        )
        if excerpt.size > 0:
            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                st.pyplot(plot_waveform(excerpt, audio_sample.sample_rate), clear_figure=True)
            with ex_col2:
                st.pyplot(plot_mel_spectrogram(excerpt, audio_sample.sample_rate), clear_figure=True)
    return rows, agreement


def render() -> None:
    config = load_models_config()
    model_config = config.get(DEPLOYMENT_MODEL_ID)

    uploaded_file = st.session_state.get("_analyze_uploaded_file")

    if not st.session_state.get("_has_uploaded_before"):
        render_hero()
        st.markdown("")
        render_capability_strip()
        st.divider()

    render_section_title("Analyze a recording", "Upload speech audio for anti-spoofing analysis.")
    uploaded_file = st.file_uploader(
        "Supported: WAV · MP3 · FLAC — up to "
        f"{MAX_DURATION_SECONDS:.0f} seconds and {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB",
        type=["wav", "mp3", "flac"],
    )
    st.caption("Processed for the current session and not intentionally retained.")

    if uploaded_file is None:
        _render_recommended_input()
        st.divider()
        render_feature_grid()
        st.divider()
        render_how_it_works()
        render_why_model()
        render_evaluation_section()
        render_disclaimer()
        render_footer()
        return

    st.session_state["_has_uploaded_before"] = True

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
            except Exception:  # noqa: BLE001
                logger.exception("Model preparation failed")
                status.update(label="Detector unavailable", state="error")
                _render_error("**Detector unavailable.** The analysis model could not be prepared. Please try again shortly.")
                render_disclaimer()
                render_footer()
                return

            status.update(label="Processing audio...")
            status.update(label="Running anti-spoof analysis...")
            try:
                result = detector.predict(audio_sample)
                segment_probs = None
                if audio_sample.duration_seconds > SEGMENT_DURATION_SECONDS:
                    n_expected = -(-int(audio_sample.duration_seconds * 16000) // 64600)
                    status.update(label=f"Analyzing {n_expected} segments...")
                    full_clip = detector.predict_full_clip(audio_sample, aggregation="mean")
                    segment_probs = full_clip["window_spoof_probs"]
                    status.update(label="Aggregating segment evidence...")
            except Exception:  # noqa: BLE001
                logger.exception("Inference failed")
                status.update(label="Analysis failed", state="error")
                _render_error("Something went wrong while analyzing this audio. Please try again, or try a different file.")
                render_disclaimer()
                render_footer()
                return

            status.update(label="Preparing results...")
            status.update(label="Analysis complete", state="complete")

        st.session_state["last_result"] = result
        st.session_state["last_result_file_key"] = file_key
        st.session_state["last_segment_probs"] = segment_probs
        st.session_state["last_audio_sample"] = audio_sample
        st.session_state["last_filename"] = uploaded_file.name
        st.session_state.pop("robustness_results", None)

        model_info_for_session = detector.model_info()
        summary_for_session = result_summary(result, calibrated_threshold=model_info_for_session.get("calibrated_threshold_spoof_probability"))
        agreement_for_session = compute_segment_agreement(segment_probs) if segment_probs else None
        record_analysis(
            st.session_state,
            build_session_entry(
                filename=uploaded_file.name,
                sha256=file_key,
                duration_seconds=audio_sample.duration_seconds,
                sample_rate=audio_sample.sample_rate,
                audio_format=Path(uploaded_file.name).suffix.lstrip(".").upper(),
                presentation_state=summary_for_session["presentation_state"],
                bonafide_probability=result.probabilities["bonafide"],
                spoof_probability=result.probabilities["spoof"],
                model_id=DEPLOYMENT_MODEL_ID,
                model_revision=model_config.revision,
                n_segments=len(segment_probs) if segment_probs else 1,
                segment_agreement_level=agreement_for_session.level if agreement_for_session else None,
            ),
        )

    result = st.session_state.get("last_result") if st.session_state.get("last_result_file_key") == file_key else None
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
        n_segments = len(segment_probs) if segment_probs else 1
        result_col, detail_col = st.columns([2, 1])
        with result_col:
            render_result_panel(state, summary["prediction"], _explanation_for_state(state), extra_html)
        with detail_col:
            st.markdown("**Analysis details**")
            st.markdown(
                f"""
| | |
|---|---|
| Duration | {result.audio_duration_seconds:.1f} sec |
| Segments | {n_segments} |
| Analysis time | {result.inference_time_ms / 1000:.1f} sec |
| Runtime | CPU / ONNX |
                """
            )

        agreement = compute_segment_agreement(segment_probs) if segment_probs else None
        with st.expander("Why this result?"):
            for sentence in evidence_summary_sentences(state, result.probabilities["spoof"], calibrated_threshold, agreement):
                st.markdown(f"- {sentence}")

        st.divider()
        quality, suitability = _render_audio_metadata_inspector(audio_sample, file_bytes, uploaded_file.name, n_segments, model_info)

        if segment_probs:
            rows, agreement = _render_segment_evidence(audio_sample, segment_probs, calibrated_threshold)
        else:
            rows = []

        try:
            from app.nav import robustness_page

            st.page_link(robustness_page, label="Run robustness analysis on this clip", icon=":material/science:")
        except Exception:  # noqa: BLE001 - nav module optional at import time in isolated tests
            pass

        st.divider()
        render_section_title("Waveform / spectrogram", "Visual representations of the submitted waveform and spectral content.")
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.caption("Waveform")
            st.pyplot(plot_waveform(audio_sample.waveform, audio_sample.sample_rate), clear_figure=True)
        with viz_col2:
            st.caption("Mel spectrogram")
            st.pyplot(plot_mel_spectrogram(audio_sample.waveform, audio_sample.sample_rate), clear_figure=True)
        st.caption(
            "The detector operates on waveform audio directly. This spectrogram is shown for "
            "visual inspection and is not the model input."
        )

        st.divider()
        render_section_title("Download analysis report")
        report_rows = segment_table(segment_probs, calibrated_threshold, audio_sample.duration_seconds) if segment_probs else []
        report_data = build_report_data(
            generated_at=datetime.now(),
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
            "Download HTML report", data=render_html_report(report_data), file_name=f"{report_data['report_id']}.html", mime="text/html"
        )
        report_col2.download_button(
            "Download JSON export", data=render_json_report(report_data), file_name=f"{report_data['report_id']}.json", mime="application/json"
        )
        st.caption(f"Report ID: {report_data['report_id']} — a local reproducibility identifier, not a database reference.")

        st.divider()
        _render_analysis_session_panel()

        st.divider()
        render_how_it_works()
        render_why_model()
        render_evaluation_section()

    render_disclaimer()
    render_footer()
