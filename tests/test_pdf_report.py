"""Tests for app/reporting/pdf_report.py -- the premium downloadable PDF
report. Verifies valid PDF bytes are produced (never written to disk),
that required content (brand name, SHA-256, prediction, model revision,
disclaimer) appears in the rendered text, that all three presentation
states render without error, and that source-specific metadata (voice
note / video / microphone) is included."""

from __future__ import annotations

import io
from datetime import datetime

import pytest
from pypdf import PdfReader

from app.reporting.pdf_report import build_pdf_report
from app.reporting.report import build_report_data

BASE_KWARGS = dict(
    generated_at=datetime(2026, 9, 3, 12, 0, 0),
    filename="clip.wav",
    file_sha256="deadbeef" * 8,
    duration_seconds=10.0,
    sample_rate=16000,
    audio_format="WAV",
    channels=1,
    bonafide_probability=0.2,
    spoof_probability=0.8,
    calibrated_threshold=0.9397,
    n_segments=2,
    segment_rows=[
        {"segment": 1, "start_seconds": 0.0, "end_seconds": 5.0, "bonafide_probability": 0.25, "spoof_probability": 0.75, "interpretation": "SPOOF"},
        {"segment": 2, "start_seconds": 5.0, "end_seconds": 10.0, "bonafide_probability": 0.15, "spoof_probability": 0.85, "interpretation": "SPOOF"},
    ],
    segment_agreement={
        "n_segments": 2, "n_spoof_leaning": 2, "n_bonafide_leaning": 0, "dominant_direction": "spoof",
        "agreement_percent": 100.0, "score_mean": 0.8, "score_median": 0.8, "score_std": 0.05, "score_range": 0.1, "level": "High",
    },
    audio_quality={"peak_amplitude": 0.9, "rms_level": 0.1, "silence_ratio": 0.02, "clipping_ratio": 0.0, "approximate_bitrate_kbps": 256.0},
    suitability_level="Good",
    model_info={
        "display_name": "Spectra-AASIST3 INT8", "architecture_short": "XLS-R-300M + KAN-AASIST",
        "runtime": "ONNX Runtime CPU", "repository": "org/spectra-aasist3", "revision": "abcdef1234",
        "threshold_description": "Calibrated 93.97% spoof probability",
    },
    inference_time_ms=250.0,
)


def _report_data(**overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    return build_report_data(**kwargs)


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_build_pdf_report_returns_valid_pdf_bytes():
    pdf_bytes = build_pdf_report(_report_data(presentation_state="SPOOF", prediction_label="Likely AI-Generated / Spoofed", binary_model_decision="SPOOF"))
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 1000


def test_pdf_has_multiple_pages():
    pdf_bytes = build_pdf_report(_report_data(presentation_state="SPOOF", prediction_label="Likely AI-Generated / Spoofed", binary_model_decision="SPOOF"))
    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 4


def test_pdf_contains_brand_name():
    pdf_bytes = build_pdf_report(_report_data(presentation_state="BONAFIDE", prediction_label="Likely Real / Bonafide", binary_model_decision="BONAFIDE"))
    text = _pdf_text(pdf_bytes)
    assert "AUDIO DEEPFAKE ANALYSIS" in text


def test_pdf_contains_sha256():
    data = _report_data(presentation_state="BONAFIDE", prediction_label="Likely Real / Bonafide", binary_model_decision="BONAFIDE")
    pdf_bytes = build_pdf_report(data)
    text = _pdf_text(pdf_bytes)
    assert data["analysis_information"]["sha256"] in text


def test_pdf_contains_prediction_label():
    data = _report_data(presentation_state="SPOOF", prediction_label="Likely AI-Generated / Spoofed", binary_model_decision="SPOOF")
    pdf_bytes = build_pdf_report(data)
    text = _pdf_text(pdf_bytes)
    assert "Likely AI-Generated" in text


def test_pdf_contains_model_revision():
    data = _report_data(presentation_state="BONAFIDE", prediction_label="Likely Real / Bonafide", binary_model_decision="BONAFIDE")
    pdf_bytes = build_pdf_report(data)
    text = _pdf_text(pdf_bytes)
    assert "abcdef1234" in text


def test_pdf_contains_disclaimer():
    data = _report_data(presentation_state="BONAFIDE", prediction_label="Likely Real / Bonafide", binary_model_decision="BONAFIDE")
    pdf_bytes = build_pdf_report(data)
    text = _pdf_text(pdf_bytes)
    assert "RESEARCH PROTOTYPE" in text
    assert "forensic" in text.lower()


def test_pdf_never_uses_forbidden_confidence_or_certainty_phrases():
    data = _report_data(presentation_state="BONAFIDE", prediction_label="Likely Real / Bonafide", binary_model_decision="BONAFIDE")
    pdf_bytes = build_pdf_report(data)
    text = _pdf_text(pdf_bytes).lower()
    for phrase in ("verified authentic", "confirmed deepfake", "forensically validated"):
        assert phrase not in text


@pytest.mark.parametrize("state,label,decision", [
    ("BONAFIDE", "Likely Real / Bonafide", "BONAFIDE"),
    ("SPOOF", "Likely AI-Generated / Spoofed", "SPOOF"),
    ("INCONCLUSIVE", "Inconclusive / Mixed Evidence", "SPOOF"),
])
def test_pdf_renders_for_every_presentation_state(state, label, decision):
    data = _report_data(presentation_state=state, prediction_label=label, binary_model_decision=decision)
    pdf_bytes = build_pdf_report(data)
    assert pdf_bytes[:5] == b"%PDF-"


def test_pdf_includes_source_specific_metadata_for_video():
    data = _report_data(
        presentation_state="BONAFIDE", prediction_label="Likely Real / Bonafide", binary_model_decision="BONAFIDE",
        source_type="video_audio",
        source_metadata={"video_filename": "clip.mp4", "container": "MP4", "video_codec": "H.264", "audio_codec": "AAC", "selected_interval": "5s–35s"},
    )
    pdf_bytes = build_pdf_report(data)
    text = _pdf_text(pdf_bytes)
    assert "Video Audio Track" in text
    assert "clip.mp4" in text


def test_pdf_includes_source_specific_metadata_for_voice_note():
    data = _report_data(
        presentation_state="BONAFIDE", prediction_label="Likely Real / Bonafide", binary_model_decision="BONAFIDE",
        source_type="voice_note", source_metadata={"format": "OGG", "normalized_to": "Mono · 16 kHz"},
    )
    pdf_bytes = build_pdf_report(data)
    text = _pdf_text(pdf_bytes)
    assert "Voice Note" in text


def test_pdf_includes_source_specific_metadata_for_microphone():
    data = _report_data(
        presentation_state="BONAFIDE", prediction_label="Likely Real / Bonafide", binary_model_decision="BONAFIDE",
        source_type="microphone", source_metadata={"capture_type": "Live microphone recording"},
    )
    pdf_bytes = build_pdf_report(data)
    text = _pdf_text(pdf_bytes)
    assert "Microphone Capture" in text


def test_pdf_handles_missing_segment_data_gracefully():
    data = _report_data(
        presentation_state="BONAFIDE", prediction_label="Likely Real / Bonafide", binary_model_decision="BONAFIDE",
        n_segments=1, segment_rows=[], segment_agreement=None,
    )
    pdf_bytes = build_pdf_report(data)
    assert pdf_bytes[:5] == b"%PDF-"


def test_build_pdf_report_does_not_write_to_disk(tmp_path, monkeypatch):
    """The report is generated entirely in memory -- no disk write
    happens anywhere in build_pdf_report."""
    monkeypatch.chdir(tmp_path)
    data = _report_data(presentation_state="BONAFIDE", prediction_label="Likely Real / Bonafide", binary_model_decision="BONAFIDE")
    build_pdf_report(data)
    assert list(tmp_path.iterdir()) == []
