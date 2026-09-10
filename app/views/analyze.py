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
from app.analysis.input_sources import (
    SOURCE_MICROPHONE,
    SOURCE_VIDEO_URL,
    NormalizedAudioInput,
    from_helper_prepared_audio,
    from_microphone,
    from_uploaded_file,
    from_video,
    from_voice_note,
    probe_video,
)
from app.analysis.session import build_session_entry, get_session_entries, record_analysis
from app.analysis.url_helper_client import (
    HelperConnection,
    HelperConnectionError,
    HelperRequestError,
    HelperVideoMetadata,
    decode_connection_code,
    fetch_metadata_via_helper,
    prepare_via_helper,
    verify_connection,
)
from app.components import (
    probability_comparison_html,
    render_analysis_condition,
    render_capability_strip,
    render_disclaimer,
    render_empty_state,
    render_evaluation_section,
    render_feature_grid,
    render_footer,
    render_hero,
    render_how_it_works,
    render_metric_cards,
    render_media_identity,
    render_metadata_card,
    render_metadata_grid,
    render_professional_table,
    render_result_panel,
    render_section_gap,
    render_section_title,
    render_why_model,
    threshold_visualization_html,
)
from app.errors import UserFacingError
from app.formatting import result_summary
from app.model_loader import DEPLOYMENT_MODEL_ID, get_detector
from app.reporting.pdf_report import build_pdf_report
from app.reporting.report import build_report_data, render_html_report, render_json_report
from app.validation import (
    MAX_DURATION_SECONDS,
    MAX_FILE_SIZE_BYTES,
    MAX_VIDEO_ANALYSIS_WINDOW_SECONDS,
    MAX_VIDEO_FILE_SIZE_BYTES,
    SUPPORTED_VIDEO_EXTENSIONS,
    SUPPORTED_VOICE_NOTE_EXTENSIONS,
)
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
        from app.analysis.input_sources import SOURCE_LABELS

        rows = [
            {
                "Time": e["timestamp"],
                "Filename": e["filename"],
                "Source": SOURCE_LABELS.get(e.get("source_type", "audio_file"), "Audio File"),
                "Result": e["presentation_state"],
                "Spoof %": f"{e['spoof_probability'] * 100:.1f}%",
                "Segments": e["n_segments"],
            }
            for e in entries
        ]
        render_professional_table(
            ["Time", "Filename", "Source", "Result", "Spoof %", "Segments"],
            rows,
            numeric_columns={"Spoof %", "Segments"},
            status_columns={"Result"},
            stack_on_mobile=True,
        )
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


def _inspect_audio_metadata(audio_sample, file_bytes: bytes, filename: str):
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
    return media, quality, suitability


def _render_analyzed_media(audio_sample, file_bytes: bytes, filename: str, media) -> None:
    render_section_title("What was analyzed", "The submitted media identity and the normalized audio used for this result.")
    render_media_identity(
        filename,
        [
            ("Analysis sample rate", f"{audio_sample.sample_rate} Hz"),
            ("File size", f"{len(file_bytes) / 1024:.0f} KB"),
            ("Container", str(media.container)),
        ],
    )


def _render_technical_diagnostics(file_bytes: bytes, model_info: dict, quality, suitability, media) -> None:
    full_sha256 = hashlib.sha256(file_bytes).hexdigest()
    render_section_title(
        "Technical diagnostics",
        "Secondary media, signal, and reproducibility details for research inspection.",
    )
    render_analysis_condition(suitability.level, suitability.reasons)
    st.caption(
        "This assessment is a descriptive signal-quality check and does not influence the "
        "detector's classification, threshold, or presentation state."
    )
    with st.expander("Recording and runtime details"):
        render_metadata_grid(
            [
                (
                    "Media",
                    [
                        ("Codec", str(media.codec)),
                        ("Bitrate", str(media.bitrate_kbps)),
                    ],
                ),
                (
                    "Analysis",
                    [
                        ("Native segment length", str(model_info.get("native_window_description", "Not available"))),
                    ],
                ),
                (
                    "Signal",
                    [
                        ("Peak amplitude", f"{quality.peak_amplitude:.2f}"),
                        ("RMS level", f"{quality.rms_level:.3f}"),
                        ("Silence proportion", f"{quality.silence_ratio * 100:.0f}%"),
                        ("Clipping", f"{quality.clipping_ratio * 100:.1f}%"),
                    ],
                ),
                (
                    "Reproducibility",
                    [("SHA-256", f"{full_sha256[:16]}…", full_sha256)],
                ),
            ]
        )


def _render_segment_evidence(audio_sample, segment_probs, calibrated_threshold) -> None:
    render_section_gap()
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
        render_professional_table(
            ["Segment", "Time range", "Bonafide probability", "Spoof probability", "Interpretation"],
            display_rows,
            numeric_columns={"Segment", "Bonafide probability", "Spoof probability"},
            status_columns={"Interpretation"},
        )

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


def _handle_input_error(exc: UserFacingError) -> None:
    _render_error(exc.friendly_message)
    if exc.technical_detail:
        logger.warning("Input validation failed: %s", exc.technical_detail)


def _source_audio_file() -> tuple[NormalizedAudioInput | None, bytes | None]:
    uploaded = st.file_uploader(
        "Supported: WAV · MP3 · FLAC — up to "
        f"{MAX_DURATION_SECONDS:.0f} seconds and {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB",
        type=["wav", "mp3", "flac"],
        key="_src_audio_file",
    )
    if uploaded is None:
        return None, None
    data = uploaded.getvalue()
    try:
        return from_uploaded_file(data, uploaded.name), data
    except UserFacingError as exc:
        _handle_input_error(exc)
        return None, None


def _source_microphone() -> tuple[NormalizedAudioInput | None, bytes | None]:
    render_section_title(
        "Record audio",
        "Record speech directly through your device microphone, or play suspicious audio from another "
        "device and capture a short sample.",
    )
    st.caption(
        "Microphone capture introduces room, speaker and microphone effects that may affect detection "
        "performance."
    )
    recording = st.audio_input("Acoustic capture", sample_rate=16000, key="_src_microphone")
    if recording is None:
        return None, None
    data = recording.getvalue()
    try:
        return from_microphone(data), data
    except UserFacingError as exc:
        _handle_input_error(exc)
        return None, None


def _source_voice_note() -> tuple[NormalizedAudioInput | None, bytes | None]:
    render_section_title(
        "Analyze a voice note",
        "Upload a saved voice message or mobile recording. The audio will be decoded and normalized "
        "before being analyzed by the same detector.",
    )
    st.caption("Supported: " + " · ".join(sorted(ext.lstrip(".").upper() for ext in SUPPORTED_VOICE_NOTE_EXTENSIONS)))
    uploaded = st.file_uploader(
        "Upload a voice note",
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_VOICE_NOTE_EXTENSIONS),
        key="_src_voice_note",
    )
    if uploaded is None:
        return None, None
    data = uploaded.getvalue()
    try:
        with st.spinner("Decoding voice note..."):
            normalized_input = from_voice_note(data, uploaded.name)
        return normalized_input, data
    except UserFacingError as exc:
        _handle_input_error(exc)
        return None, None


def _source_video() -> tuple[NormalizedAudioInput | None, bytes | None]:
    render_section_title(
        "Video audio analysis",
        "Extract and analyze the speech track from a video using the audio deepfake detector. Visual "
        "frames are not analyzed.",
    )
    st.caption(
        f"Supported: MP4 · MOV · MKV · WEBM — up to {MAX_VIDEO_FILE_SIZE_BYTES // (1024 * 1024)} MB. "
        f"Maximum analyzed audio duration: {MAX_VIDEO_ANALYSIS_WINDOW_SECONDS:.0f} seconds."
    )
    uploaded = st.file_uploader(
        "Upload a video",
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS),
        key="_src_video",
    )
    if uploaded is None:
        return None, None
    data = uploaded.getvalue()
    try:
        probed = probe_video(data, uploaded.name)
    except UserFacingError as exc:
        _handle_input_error(exc)
        return None, None

    st.video(data)
    render_metadata_card(
        "Video file",
        [
            ("Filename", uploaded.name),
            ("Container", probed.container),
            ("Video codec", probed.video_codec or "Not available"),
            ("Audio codec", probed.audio_codec),
            ("Video duration", f"{probed.duration_seconds:.1f} s" if probed.duration_seconds else "Not available"),
            ("File size", f"{len(data) / (1024 * 1024):.1f} MB"),
        ],
    )

    total_duration = probed.duration_seconds or 0.0
    window_seconds = min(MAX_VIDEO_ANALYSIS_WINDOW_SECONDS, total_duration) if total_duration else MAX_VIDEO_ANALYSIS_WINDOW_SECONDS
    if total_duration > MAX_VIDEO_ANALYSIS_WINDOW_SECONDS:
        start_seconds = st.slider(
            "Analyze from",
            min_value=0.0,
            max_value=float(total_duration - MAX_VIDEO_ANALYSIS_WINDOW_SECONDS),
            value=0.0,
            step=1.0,
            format="%.0f s",
            key="_src_video_start",
        )
    else:
        start_seconds = 0.0
    st.caption(f"Analysis window: {start_seconds:.0f}s – {start_seconds + window_seconds:.0f}s")

    try:
        with st.spinner("Extracting selected audio and normalizing..."):
            normalized_input = from_video(data, uploaded.name, start_seconds=start_seconds, window_seconds=window_seconds)
    except UserFacingError as exc:
        _handle_input_error(exc)
        return None, None
    return normalized_input, data


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


_HELPER_CONNECTION_KEY = "_url_helper_connection"


def _render_helper_connection_card() -> HelperConnection | None:
    """Video URL analysis is served by a local URL ingestion helper the
    user runs on their own machine (see docs/local_url_helper.md) --
    Streamlit itself never retrieves YouTube media directly. Returns the
    verified connection for the current session, or None (having already
    rendered the connect card) if not yet connected.

    The endpoint and token live ONLY in st.session_state -- never written
    to Streamlit secrets, a file, or a log line (see
    app/analysis/url_helper_client.py)."""
    connection: HelperConnection | None = st.session_state.get(_HELPER_CONNECTION_KEY)
    if connection is not None:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.caption(f"Local URL helper connected · {connection.endpoint}")
        with col2:
            if st.button("Disconnect", key="_url_helper_disconnect"):
                st.session_state.pop(_HELPER_CONNECTION_KEY, None)
                st.rerun()
        return connection

    with st.container(border=True):
        st.markdown("**Local URL Helper**")
        st.caption("Video URL analysis uses a local secure helper to retrieve public media.")
        code = st.text_input("Connection code", key="_url_helper_code", type="password")
        if st.button("Connect", key="_url_helper_connect"):
            try:
                candidate = decode_connection_code(code)
                with st.spinner("Verifying helper connection..."):
                    verify_connection(candidate)
            except (HelperConnectionError, HelperRequestError) as exc:
                _render_error(str(exc))
            else:
                st.session_state[_HELPER_CONNECTION_KEY] = candidate
                st.rerun()
    return None


def _source_video_url() -> tuple[NormalizedAudioInput | None, bytes | None]:
    render_section_title(
        "Analyze Online Video Audio",
        "Paste a public video link to inspect a selected portion of its audio track.",
    )
    st.caption(
        "Supported initially: YouTube · YouTube Shorts. Audio track analysis only. "
        "No visual deepfake detection."
    )

    connection = _render_helper_connection_card()
    if connection is None:
        st.info(
            "Video URL helper is not connected. Start the Audio Deepfake URL Helper on the "
            "analysis computer, then connect it here."
        )
        st.caption(
            "Alternatively, download/export the recording and analyze it using Audio File or Video File."
        )
        return None, None

    url = st.text_input(
        "Video URL", key="_src_video_url", placeholder="https://www.youtube.com/watch?v=..."
    )
    load_clicked = st.button("Load Video", key="_src_video_url_load")

    if load_clicked:
        st.session_state.pop("_video_url_metadata", None)
        st.session_state.pop("_video_url_normalized", None)
        if not url:
            _render_error("Please paste a video URL.")
        else:
            try:
                with st.spinner("Retrieving video information..."):
                    metadata = fetch_metadata_via_helper(connection, url)
                st.session_state["_video_url_metadata"] = metadata
                st.session_state["_video_url_metadata_url"] = url
            except (HelperConnectionError, HelperRequestError) as exc:
                _render_error(str(exc))

    metadata: HelperVideoMetadata | None = st.session_state.get("_video_url_metadata")
    metadata_url = st.session_state.get("_video_url_metadata_url")
    if metadata is None or metadata_url != url:
        return None, None

    if metadata.thumbnail_url:
        st.image(metadata.thumbnail_url, width=320)
    render_metadata_card(
        "Online video",
        [
            ("Platform", metadata.platform),
            ("Video title", metadata.title),
            ("Duration", _format_hms(metadata.duration_seconds) if metadata.duration_seconds else "Not available"),
            ("Channel / uploader", metadata.uploader or "Not available"),
            ("Audio availability", "Available"),
        ],
    )

    total_duration = metadata.duration_seconds or 0.0
    window_seconds = min(MAX_VIDEO_ANALYSIS_WINDOW_SECONDS, total_duration) if total_duration else MAX_VIDEO_ANALYSIS_WINDOW_SECONDS
    if total_duration > MAX_VIDEO_ANALYSIS_WINDOW_SECONDS:
        start_seconds = st.slider(
            "Analysis start time",
            min_value=0.0,
            max_value=float(total_duration - MAX_VIDEO_ANALYSIS_WINDOW_SECONDS),
            value=0.0,
            step=1.0,
            format="%.0f s",
            key="_src_video_url_start",
        )
    else:
        start_seconds = 0.0
    st.caption(
        f"Video duration: {_format_hms(total_duration)} · "
        f"Analyzed interval: {_format_hms(start_seconds)} – {_format_hms(start_seconds + window_seconds)}"
    )

    extract_clicked = st.button("Prepare Selected Audio", key="_src_video_url_extract")
    cache_key = f"{metadata_url}:{start_seconds:.0f}"

    if extract_clicked:
        try:
            with st.spinner("Retrieving and extracting the selected audio interval..."):
                prepared = prepare_via_helper(connection, metadata_url, start_seconds, window_seconds)
                normalized_input = from_helper_prepared_audio(
                    prepared.wav_bytes,
                    display_filename=prepared.metadata.title,
                    start_seconds=start_seconds,
                    window_seconds=window_seconds,
                    platform=prepared.metadata.platform,
                    source_title=prepared.metadata.title,
                    source_url=prepared.metadata.webpage_url,
                    source_uploader=prepared.metadata.uploader,
                    video_duration_seconds=prepared.metadata.duration_seconds,
                    source_format=prepared.source_extension,
                )
            st.session_state["_video_url_normalized"] = normalized_input
            st.session_state["_video_url_normalized_key"] = cache_key
        except (HelperConnectionError, HelperRequestError) as exc:
            st.session_state.pop("_video_url_normalized", None)
            _render_error(str(exc))
            return None, None

    normalized_input = st.session_state.get("_video_url_normalized")
    if normalized_input is not None and st.session_state.get("_video_url_normalized_key") == cache_key:
        return normalized_input, None
    return None, None


def render() -> None:
    config = load_models_config()
    model_config = config.get(DEPLOYMENT_MODEL_ID)

    if not st.session_state.get("_has_uploaded_before"):
        render_hero()
        st.markdown("")
        render_capability_strip()
        st.divider()

    render_section_title("Analyze a recording", "Choose an input source for anti-spoofing analysis.")
    source_choice = st.segmented_control(
        "Input source",
        options=["Audio File", "Microphone", "Voice Note", "Video", "Video URL"],
        default="Audio File",
        label_visibility="collapsed",
        key="_analyze_source",
    )
    if not source_choice:
        source_choice = "Audio File"

    source_fns = {
        "Audio File": _source_audio_file,
        "Microphone": _source_microphone,
        "Voice Note": _source_voice_note,
        "Video": _source_video,
        "Video URL": _source_video_url,
    }
    normalized_input, preview_bytes = source_fns[source_choice]()
    is_video_source = source_choice == "Video"
    is_video_url_source = source_choice == "Video URL"

    st.caption("Processed for the current session and not intentionally retained.")

    if normalized_input is None:
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

    audio_sample = normalized_input.audio_sample
    filename = normalized_input.display_filename
    file_bytes = preview_bytes
    file_key = f"{normalized_input.source_type}:{normalized_input.sha256}"

    if is_video_url_source:
        render_metadata_card(
            "Prepared audio",
            [
                ("Platform", normalized_input.source_metadata.get("platform", "Not available")),
                ("Video", normalized_input.source_metadata.get("source_title", filename)),
                ("Extracted audio duration", f"{audio_sample.duration_seconds:.2f} s"),
                ("Analyzed interval", normalized_input.source_metadata["selected_interval"]),
                ("Source", normalized_input.source_label),
            ],
        )
    elif is_video_source:
        render_metadata_card(
            "Prepared audio",
            [
                ("Extracted audio duration", f"{audio_sample.duration_seconds:.2f} s"),
                ("Analyzed interval", normalized_input.source_metadata["selected_interval"]),
                ("Source", normalized_input.source_label),
            ],
        )
    else:
        workspace_col, meta_col = st.columns([2, 1])
        with workspace_col:
            st.audio(file_bytes)
        with meta_col:
            render_metadata_card(
                "Audio file",
                [
                    ("Filename", filename),
                    ("Duration", f"{audio_sample.duration_seconds:.2f} s"),
                    ("Sample rate", f"{audio_sample.sample_rate} Hz"),
                    ("Source", normalized_input.source_label),
                    ("File size", f"{len(file_bytes) / 1024:.0f} KB"),
                ],
            )

    analyze_label = "Analyze Recording" if normalized_input.source_type == SOURCE_MICROPHONE else "Analyze Audio"
    analyze_clicked = st.button(analyze_label, type="primary")

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
        st.session_state["last_filename"] = filename
        st.session_state["last_source_type"] = normalized_input.source_type
        st.session_state["last_source_metadata"] = normalized_input.source_metadata
        st.session_state.pop("robustness_results", None)

        model_info_for_session = detector.model_info()
        summary_for_session = result_summary(result, calibrated_threshold=model_info_for_session.get("calibrated_threshold_spoof_probability"))
        agreement_for_session = compute_segment_agreement(segment_probs) if segment_probs else None
        record_analysis(
            st.session_state,
            build_session_entry(
                filename=filename,
                sha256=normalized_input.sha256,
                duration_seconds=audio_sample.duration_seconds,
                sample_rate=audio_sample.sample_rate,
                audio_format=normalized_input.source_metadata.get("format", Path(filename).suffix.lstrip(".").upper()),
                presentation_state=summary_for_session["presentation_state"],
                bonafide_probability=result.probabilities["bonafide"],
                spoof_probability=result.probabilities["spoof"],
                model_id=DEPLOYMENT_MODEL_ID,
                model_revision=model_config.revision,
                n_segments=len(segment_probs) if segment_probs else 1,
                segment_agreement_level=agreement_for_session.level if agreement_for_session else None,
                source_type=normalized_input.source_type,
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

        n_segments = len(segment_probs) if segment_probs else 1
        acoustic_advisory = None
        if normalized_input.source_type == SOURCE_MICROPHONE:
            acoustic_advisory = (
                "Acoustic capture advisory: audio recorded through a loudspeaker and microphone may "
                "differ substantially from the original source. Room acoustics, playback equipment, "
                "microphone processing and background noise can affect detector output."
            )
        render_result_panel(
            state,
            summary["prediction"],
            _explanation_for_state(state),
            extra_html,
            source_label=normalized_input.source_label,
            metadata=[
                ("Duration", f"{result.audio_duration_seconds:.1f} sec"),
                ("Segments", str(n_segments)),
                ("Analysis time", f"{result.inference_time_ms / 1000:.1f} sec"),
                ("Runtime", "CPU / ONNX"),
            ],
            advisory=acoustic_advisory,
        )

        agreement = compute_segment_agreement(segment_probs) if segment_probs else None
        with st.container(key="result_explanation"):
            with st.expander("Why this result?"):
                for sentence in evidence_summary_sentences(state, result.probabilities["spoof"], calibrated_threshold, agreement):
                    st.markdown(f"- {sentence}")

        if segment_probs:
            _, agreement = _render_segment_evidence(audio_sample, segment_probs, calibrated_threshold)

        render_section_gap()
        metadata_probe_bytes = file_bytes if file_bytes is not None else audio_sample.waveform.tobytes()
        metadata_probe_filename = filename if file_bytes is not None else "clip.wav"
        media, quality, suitability = _inspect_audio_metadata(
            audio_sample,
            metadata_probe_bytes,
            metadata_probe_filename,
        )
        _render_analyzed_media(audio_sample, metadata_probe_bytes, filename, media)

        render_section_gap()
        render_section_title("Waveform / spectrogram", "Visual representations of the submitted waveform and spectral content.")
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            with st.container(border=True, key="waveform_plot"):
                st.caption("Waveform")
                st.pyplot(plot_waveform(audio_sample.waveform, audio_sample.sample_rate), clear_figure=True)
        with viz_col2:
            with st.container(border=True, key="spectrogram_plot"):
                st.caption("Mel spectrogram")
                st.pyplot(plot_mel_spectrogram(audio_sample.waveform, audio_sample.sample_rate), clear_figure=True)
        st.caption(
            "The detector operates on waveform audio directly. This spectrogram is shown for "
            "visual inspection and is not the model input."
        )

        render_section_gap()
        _render_technical_diagnostics(metadata_probe_bytes, model_info, quality, suitability, media)
        _render_analysis_session_panel()

        render_section_gap()
        report_rows = segment_table(segment_probs, calibrated_threshold, audio_sample.duration_seconds) if segment_probs else []
        report_data = build_report_data(
            generated_at=datetime.now(),
            filename=filename,
            file_sha256=normalized_input.sha256,
            duration_seconds=audio_sample.duration_seconds,
            sample_rate=audio_sample.sample_rate,
            audio_format=normalized_input.source_metadata.get("format", Path(filename).suffix.lstrip(".").upper()),
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
            source_type=normalized_input.source_type,
            source_metadata=normalized_input.source_metadata,
        )
        with st.spinner("Generating report..."):
            pdf_bytes = build_pdf_report(report_data)
        try:
            from app.views.robustness import render as render_robustness

            robustness_page = st.Page(
                render_robustness,
                title="Robustness",
                url_path="robustness",
                icon=":material/science:",
            )
        except Exception:  # noqa: BLE001 - page-link target is optional in isolated view tests
            robustness_page = None

        with st.container(key="next_actions"):
            render_section_title("Next steps", "Stress-test this result or export a reproducible analysis record.")
            robustness_col, report_col = st.columns([1, 2])
            with robustness_col:
                with st.container(key="robustness_action"):
                    st.markdown(
                        '<div class="adf-action-kicker">Robustness</div>'
                        '<div class="adf-action-title">Test this clip under degraded conditions</div>'
                        '<div class="adf-action-body">Compare the result across controlled audio transformations.</div>',
                        unsafe_allow_html=True,
                    )
                    if robustness_page is not None:
                        st.page_link(
                            robustness_page,
                            label="Run robustness analysis on this clip",
                            icon=":material/science:",
                            width="stretch",
                        )
            with report_col:
                with st.container(key="report_downloads"):
                    st.markdown(
                        '<div class="adf-action-kicker">Report</div>'
                        '<div class="adf-action-title">Download analysis report</div>'
                        '<div class="adf-action-body">Choose the format that fits your review or research workflow.</div>',
                        unsafe_allow_html=True,
                    )
                    report_col1, report_col2, report_col3 = st.columns(3)
                    report_col1.download_button(
                        "Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{report_data['report_id']}.pdf",
                        mime="application/pdf",
                        type="primary",
                        width="stretch",
                    )
                    report_col2.download_button(
                        "Download HTML report",
                        data=render_html_report(report_data),
                        file_name=f"{report_data['report_id']}.html",
                        mime="text/html",
                        width="stretch",
                    )
                    report_col3.download_button(
                        "Download JSON export",
                        data=render_json_report(report_data),
                        file_name=f"{report_data['report_id']}.json",
                        mime="application/json",
                        width="stretch",
                    )
                    st.markdown(
                        f'<div class="adf-report-id"><span>Report ID</span><strong>{report_data["report_id"]}</strong>'
                        "<span>Local reproducibility identifier; not a database reference.</span></div>",
                        unsafe_allow_html=True,
                    )

        render_section_gap()
        render_how_it_works()
        render_why_model()
        render_evaluation_section()

    render_disclaimer()
    render_footer()
