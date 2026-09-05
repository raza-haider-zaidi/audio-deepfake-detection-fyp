"""Builds a session-local analysis report (HTML + JSON) from the current
analysis result. No PDF dependency is introduced (per project policy,
prefer a lightweight, reliable HTML report over adding a PDF library);
JSON is offered as a secondary technical export.

Nothing here implies a persistent database or forensic validation. The
report ID is derived deterministically from the input file's SHA-256
hash plus the analysis date, purely as a local reproducibility
identifier -- not a claim that this analysis was recorded anywhere.
"""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from app.branding import display_model_artifact

DISCLAIMER_TEXT = (
    "Research prototype. This detector should not be used as the sole basis for "
    "forensic, legal, security, disciplinary, or identity decisions. Detection "
    "performance can vary with synthesis methods, recording conditions, "
    "compression, language, and background noise."
)

FORBIDDEN_PHRASES = ("verified authentic", "confirmed deepfake", "forensically validated")

SOURCE_DISPLAY_LABELS = {
    "audio_file": "Audio File",
    "microphone": "Microphone Capture",
    "voice_note": "Voice Note",
    "video_audio": "Video Audio Track",
    "video_url": "Online Video",
}


def compute_file_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def generate_report_id(sha256_hex: str, generated_at: datetime) -> str:
    """AD-YYYYMMDD-XXXX, where XXXX is derived from the file hash. This is
    a LOCAL, session-scoped identifier -- it does not reference or imply
    any persistent database."""
    date_part = generated_at.strftime("%Y%m%d")
    short_hash = sha256_hex[:4].upper()
    return f"AD-{date_part}-{short_hash}"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def build_report_data(
    *,
    generated_at: datetime,
    filename: str,
    file_sha256: str,
    duration_seconds: float,
    sample_rate: int,
    audio_format: str,
    channels: int,
    presentation_state: str,
    prediction_label: str,
    bonafide_probability: float,
    spoof_probability: float,
    calibrated_threshold: float,
    binary_model_decision: str,
    n_segments: int,
    segment_rows: list[dict],
    segment_agreement: dict | None,
    audio_quality: dict | None,
    suitability_level: str | None,
    model_info: dict,
    inference_time_ms: float,
    source_type: str = "audio_file",
    source_metadata: dict | None = None,
) -> dict[str, Any]:
    """Assembles every reportable field into a plain, JSON-serializable
    dict. No field here is computed independently of the values already
    produced elsewhere in the app -- this function only reshapes them."""
    report_id = generate_report_id(file_sha256, generated_at)
    data = {
        "report_id": report_id,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "analysis_information": {
            "filename": filename,
            "sha256": file_sha256,
            "duration_seconds": round(duration_seconds, 3),
            "sample_rate_hz": sample_rate,
            "format": audio_format,
            "channels": channels,
            "source_type": source_type,
            "source_metadata": source_metadata or {},
        },
        "result": {
            "presentation_result": prediction_label,
            "presentation_state": presentation_state,
            "bonafide_class_probability": bonafide_probability,
            "spoof_class_probability": spoof_probability,
            "calibrated_threshold": calibrated_threshold,
            "binary_model_decision": binary_model_decision,
        },
        "segment_analysis": {
            "n_segments": n_segments,
            "segments": segment_rows,
            "segment_agreement": segment_agreement,
        },
        "audio_conditions": {
            "quality_metrics": audio_quality,
            "analysis_suitability": suitability_level,
        },
        "model": {
            "display_name": model_info.get("display_name"),
            "architecture": model_info.get("architecture_short"),
            "runtime": model_info.get("runtime"),
            "repository": display_model_artifact(model_info.get("repository")),
            "revision": model_info.get("revision"),
            "threshold_description": model_info.get("threshold_description"),
        },
        "performance": {
            "inference_time_ms": inference_time_ms,
        },
        "disclaimer": DISCLAIMER_TEXT,
    }
    return _jsonable(data)


def render_json_report(report_data: dict) -> str:
    return json.dumps(report_data, indent=2)


def _row(label: str, value: Any) -> str:
    return f"<tr><td class='label'>{html.escape(str(label))}</td><td class='value'>{html.escape(str(value))}</td></tr>"


def render_html_report(report_data: dict) -> str:
    """A single, self-contained HTML document -- inline CSS only, no
    external assets, matching the app's dark/professional visual
    direction so a downloaded report looks consistent with the product."""
    ai = report_data["analysis_information"]
    result = report_data["result"]
    seg = report_data["segment_analysis"]
    model = report_data["model"]
    perf = report_data["performance"]

    segment_rows_html = ""
    for row in seg["segments"][:200]:  # guard against pathological sizes
        segment_rows_html += (
            "<tr>"
            f"<td>{row['segment']}</td>"
            f"<td>{row['start_seconds']:.2f}s – {row['end_seconds']:.2f}s</td>"
            f"<td>{row['bonafide_probability'] * 100:.1f}%</td>"
            f"<td>{row['spoof_probability'] * 100:.1f}%</td>"
            f"<td>{html.escape(row['interpretation'])}</td>"
            "</tr>"
        )

    agreement_html = ""
    if seg.get("segment_agreement"):
        a = seg["segment_agreement"]
        agreement_html = (
            f"<p>Segment agreement: <strong>{html.escape(a['level'])}</strong> "
            f"({a['agreement_percent']:.0f}% of {a['n_segments']} segments leaned {html.escape(a['dominant_direction'])}).</p>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Audio Deepfake Analysis Report — {html.escape(report_data['report_id'])}</title>
<style>
  body {{ background:#0b0f14; color:#e8ecf1; font-family: -apple-system, 'Segoe UI', Arial, sans-serif; margin:0; padding:2rem; }}
  .sheet {{ max-width:760px; margin:0 auto; }}
  h1 {{ font-size:1.4rem; margin-bottom:0.2rem; }}
  h2 {{ font-size:1rem; text-transform:uppercase; letter-spacing:0.06em; color:#8b96a8; margin-top:2rem; border-bottom:1px solid #232c3a; padding-bottom:0.4rem; }}
  .meta {{ color:#8b96a8; font-size:0.85rem; margin-bottom:1.5rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
  td {{ padding:0.35rem 0.5rem; border-bottom:1px solid #232c3a; }}
  td.label {{ color:#8b96a8; width:40%; }}
  td.value {{ color:#e8ecf1; font-family: ui-monospace, monospace; word-break: break-all; }}
  .disclaimer {{ margin-top:2rem; font-size:0.8rem; color:#8b96a8; border-top:1px solid #232c3a; padding-top:1rem; }}
</style>
</head>
<body>
<div class="sheet">
  <h1>Audio Deepfake Analysis Report</h1>
  <div class="meta">Report ID {html.escape(report_data['report_id'])} &middot; generated {html.escape(report_data['generated_at'])}</div>

  <h2>Analysis Information</h2>
  <table>
    {_row("Input source", SOURCE_DISPLAY_LABELS.get(ai.get("source_type"), ai.get("source_type", "Audio File")))}
    {_row("Filename", ai["filename"])}
    {_row("SHA-256", ai["sha256"])}
    {_row("Duration", f"{ai['duration_seconds']:.2f} s")}
    {_row("Sample rate", f"{ai['sample_rate_hz']} Hz")}
    {_row("Format", ai["format"])}
    {_row("Channels", ai["channels"])}
    {"".join(_row(k.replace('_', ' ').title(), v) for k, v in (ai.get("source_metadata") or {}).items() if k != "format")}
  </table>

  <h2>Result</h2>
  <table>
    {_row("Presentation result", result["presentation_result"])}
    {_row("Bonafide class probability", f"{result['bonafide_class_probability'] * 100:.1f}%")}
    {_row("Spoof class probability", f"{result['spoof_class_probability'] * 100:.1f}%")}
    {_row("Calibrated threshold", f"{result['calibrated_threshold'] * 100:.2f}%")}
    {_row("Binary model decision", result["binary_model_decision"])}
  </table>

  <h2>Segment Analysis</h2>
  <p>{seg["n_segments"]} segment(s) analyzed.</p>
  {agreement_html}
  <table>
    <tr><td class="label">Segment</td><td class="label">Time range</td><td class="label">Bonafide</td><td class="label">Spoof</td><td class="label">Interpretation</td></tr>
    {segment_rows_html}
  </table>

  <h2>Model</h2>
  <table>
    {_row("Model", model.get("display_name") or "")}
    {_row("Architecture", model.get("architecture") or "")}
    {_row("Runtime", model.get("runtime") or "")}
    {_row("Model artifact", model.get("repository") or "")}
    {_row("Revision", model.get("revision") or "")}
    {_row("Decision threshold", model.get("threshold_description") or "")}
  </table>

  <h2>Performance</h2>
  <table>
    {_row("Inference time", f"{perf['inference_time_ms']:.0f} ms")}
  </table>

  <div class="disclaimer">{html.escape(report_data["disclaimer"])}</div>
</div>
</body>
</html>
"""
