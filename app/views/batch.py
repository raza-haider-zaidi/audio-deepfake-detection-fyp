"""Batch Analysis — sequentially analyzes multiple uploaded files with
the SAME frozen detector. Never runs concurrently (protects Streamlit
Cloud memory); processes one file at a time, reusing the cached model.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from app.analysis.audio_quality import assess_analysis_suitability, compute_audio_quality
from app.components import render_empty_state, render_professional_table, render_section_title
from app.errors import UserFacingError
from app.formatting import result_summary
from app.model_loader import DEPLOYMENT_MODEL_ID, get_detector
from app.validation import validate_and_load_upload

logger = logging.getLogger("audio_deepfake_detector.batch")

MAX_BATCH_FILES = 5  # conservative: each analysis costs ~0.6-2s CPU inference
# plus a full preprocessing pass; 5 sequential files keeps a batch run
# under ~30s end-to-end on the measured INT8 per-window latency
# (results/metrics/spectra_resource_fp32_vs_int8.json), avoiding a
# Streamlit Cloud request-timeout risk from an unbounded queue.


def render() -> None:
    render_section_title("Batch analysis", "Analyze multiple recordings sequentially with the same frozen detector.")

    uploaded_files = st.file_uploader(
        f"Select up to {MAX_BATCH_FILES} audio files (WAV, MP3, FLAC)",
        type=["wav", "mp3", "flac"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        render_empty_state(
            "Batch analysis",
            "Select multiple recordings above to analyze them sequentially with the deployed detector. "
            f"Up to {MAX_BATCH_FILES} files per batch, processed one at a time to protect deployment memory.",
        )
        st.caption("No files are analyzed concurrently, and no batch is stored beyond this session.")
        return

    if len(uploaded_files) > MAX_BATCH_FILES:
        st.warning(f"Only the first {MAX_BATCH_FILES} files will be analyzed in this batch.")
        uploaded_files = uploaded_files[:MAX_BATCH_FILES]

    from app.components import professional_table_html

    def _render_queue(placeholder, rows: list[dict]) -> None:
        placeholder.markdown(
            professional_table_html(
                ["Filename", "Size (KB)", "Status"], rows, numeric_columns={"Size (KB)"}, status_columns={"Status"}
            ),
            unsafe_allow_html=True,
        )

    queue_rows = [{"Filename": f.name, "Size (KB)": f"{len(f.getvalue()) / 1024:.0f}", "Status": "Queued"} for f in uploaded_files]
    queue_placeholder = st.empty()
    _render_queue(queue_placeholder, queue_rows)

    if not st.button("Run batch analysis", type="primary"):
        return

    detector = None
    try:
        with st.spinner("Preparing detector..."):
            detector = get_detector(DEPLOYMENT_MODEL_ID)
    except Exception:  # noqa: BLE001
        logger.exception("Model preparation failed for batch analysis")
        st.error("**Detector unavailable.** The analysis model could not be prepared. Please try again shortly.")
        return

    model_info = detector.model_info()
    threshold = model_info.get("calibrated_threshold_spoof_probability")
    results = []
    progress = st.progress(0.0, text="Starting batch analysis...")

    for i, uploaded_file in enumerate(uploaded_files):
        queue_rows[i]["Status"] = "Processing"
        _render_queue(queue_placeholder, queue_rows)
        progress.progress(i / len(uploaded_files), text=f"Analyzing {i + 1} of {len(uploaded_files)}: {uploaded_file.name}")

        file_bytes = uploaded_file.getvalue()
        try:
            audio_sample = validate_and_load_upload(file_bytes, uploaded_file.name)
            prediction = detector.predict(audio_sample)
            summary = result_summary(prediction, calibrated_threshold=threshold)
            quality = compute_audio_quality(audio_sample.waveform, audio_sample.sample_rate, len(file_bytes), audio_sample.duration_seconds)
            suitability = assess_analysis_suitability(quality)
            results.append(
                {
                    "File": uploaded_file.name,
                    "Duration (s)": round(audio_sample.duration_seconds, 2),
                    "Presentation result": summary["prediction"],
                    "Bonafide probability": round(prediction.probabilities["bonafide"], 4),
                    "Spoof probability": round(prediction.probabilities["spoof"], 4),
                    "Analysis conditions": suitability.level,
                    "Inference time (ms)": round(prediction.inference_time_ms, 1),
                    "SHA-256": hashlib.sha256(file_bytes).hexdigest(),
                }
            )
            queue_rows[i]["Status"] = "Complete"
        except UserFacingError as exc:
            results.append(
                {
                    "File": uploaded_file.name,
                    "Duration (s)": None,
                    "Presentation result": f"Error: {exc.friendly_message}",
                    "Bonafide probability": None,
                    "Spoof probability": None,
                    "Analysis conditions": None,
                    "Inference time (ms)": None,
                    "SHA-256": hashlib.sha256(file_bytes).hexdigest(),
                }
            )
            queue_rows[i]["Status"] = "Failed"
        except Exception:  # noqa: BLE001
            logger.exception("Batch inference failed for %s", uploaded_file.name)
            results.append(
                {
                    "File": uploaded_file.name,
                    "Duration (s)": None,
                    "Presentation result": "Error: analysis failed",
                    "Bonafide probability": None,
                    "Spoof probability": None,
                    "Analysis conditions": None,
                    "Inference time (ms)": None,
                    "SHA-256": hashlib.sha256(file_bytes).hexdigest(),
                }
            )
            queue_rows[i]["Status"] = "Failed"
        _render_queue(queue_placeholder, queue_rows)

    progress.progress(1.0, text="Batch analysis complete.")
    st.session_state["last_batch_results"] = results

    render_section_title("Batch results")
    render_professional_table(
        ["File", "Duration (s)", "Presentation result", "Bonafide probability", "Spoof probability", "Analysis conditions", "Inference time (ms)", "SHA-256"],
        results,
        numeric_columns={"Duration (s)", "Bonafide probability", "Spoof probability", "Inference time (ms)"},
        status_columns={"Presentation result", "Analysis conditions"},
        stack_on_mobile=True,
    )

    csv_buffer = io.StringIO()
    if results:
        writer = csv.DictWriter(csv_buffer, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    col1, col2 = st.columns(2)
    col1.download_button("Download CSV", data=csv_buffer.getvalue(), file_name="batch_results.csv", mime="text/csv")

    batch_report = _render_batch_report_html(results, model_info)
    col2.download_button("Download batch report", data=batch_report, file_name="batch_report.html", mime="text/html")


def _render_batch_report_html(results: list[dict], model_info: dict) -> str:
    import html

    rows_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v)) if v is not None else '—'}</td>" for v in r.values()) + "</tr>" for r in results
    )
    headers_html = "".join(f"<th>{html.escape(k)}</th>" for k in (results[0].keys() if results else []))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Batch Analysis Report</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; background:#F5F7FB; color:#1B2440; padding:2rem; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #DEE3EE; padding: 0.4rem 0.6rem; font-size: 0.85rem; text-align: left; }}
th {{ background: #EEF1F8; }}
.disclaimer {{ margin-top: 1.5rem; font-size: 0.8rem; color: #4A5578; }}
</style></head>
<body>
<h1>Batch Analysis Report</h1>
<p>Generated {datetime.now().isoformat(timespec='seconds')} — Model: {html.escape(model_info.get('display_name', ''))}</p>
<table><tr>{headers_html}</tr>{rows_html}</table>
<div class="disclaimer">Research prototype. Not for forensic, legal, security, disciplinary, or identity decisions.</div>
</body></html>
"""
