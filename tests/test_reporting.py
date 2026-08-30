"""Tests for app/reporting/report.py -- SHA-256 handling, report ID
generation, and HTML/JSON report content."""

import json
from datetime import datetime

from app.reporting.report import (
    DISCLAIMER_TEXT,
    FORBIDDEN_PHRASES,
    build_report_data,
    compute_file_sha256,
    generate_report_id,
    render_html_report,
    render_json_report,
)


def test_compute_file_sha256_matches_hashlib():
    import hashlib

    data = b"some audio bytes"
    assert compute_file_sha256(data) == hashlib.sha256(data).hexdigest()


def test_generate_report_id_format():
    sha = "abcdef1234567890"
    report_id = generate_report_id(sha, datetime(2026, 8, 31, 10, 0, 0))
    assert report_id == "AD-20260831-ABCD"


def _sample_report_data(**overrides):
    defaults = dict(
        generated_at=datetime(2026, 8, 31, 10, 0, 0),
        filename="clip.wav",
        file_sha256="abc123def456",
        duration_seconds=4.04,
        sample_rate=16000,
        audio_format="WAV",
        channels=1,
        presentation_state="BONAFIDE",
        prediction_label="Likely Real / Bonafide",
        bonafide_probability=0.9,
        spoof_probability=0.1,
        calibrated_threshold=0.9397,
        binary_model_decision="bonafide",
        n_segments=1,
        segment_rows=[{"segment": 1, "start_seconds": 0.0, "end_seconds": 4.04, "bonafide_probability": 0.9, "spoof_probability": 0.1, "interpretation": "BONAFIDE"}],
        segment_agreement=None,
        audio_quality={"peak_amplitude": 0.3, "rms_level": 0.1, "silence_ratio": 0.05, "clipping_ratio": 0.0, "approximate_bitrate_kbps": 256.0},
        suitability_level="Good",
        model_info={"display_name": "Spectra-AASIST3 INT8", "architecture_short": "XLS-R-300M + KAN-enhanced AASIST", "runtime": "ONNX Runtime (CPU)", "repository": "Limitless-8/spectra-aasist3-int8-audio-deepfake", "revision": "abc123", "threshold_description": "0.9397"},
        inference_time_ms=650.0,
    )
    defaults.update(overrides)
    return build_report_data(**defaults)


def test_build_report_data_includes_all_required_sections():
    data = _sample_report_data()
    assert set(data.keys()) >= {
        "report_id",
        "generated_at",
        "analysis_information",
        "result",
        "segment_analysis",
        "audio_conditions",
        "model",
        "performance",
        "disclaimer",
    }


def test_build_report_data_is_json_serializable():
    data = _sample_report_data()
    json.dumps(data)  # must not raise


def test_report_disclaimer_present_and_no_forbidden_claims():
    data = _sample_report_data()
    assert data["disclaimer"] == DISCLAIMER_TEXT
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in data["disclaimer"].lower()


def test_render_html_report_contains_key_fields():
    data = _sample_report_data()
    html_report = render_html_report(data)
    assert data["report_id"] in html_report
    assert "clip.wav" in html_report
    assert "abc123def456" in html_report
    assert "Likely Real / Bonafide" in html_report
    assert "Spectra-AASIST3 INT8" in html_report
    assert DISCLAIMER_TEXT in html_report


def test_render_html_report_contains_no_forbidden_authenticity_claims():
    data = _sample_report_data()
    html_report = render_html_report(data).lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in html_report


def test_render_json_report_round_trips():
    data = _sample_report_data()
    parsed = json.loads(render_json_report(data))
    assert parsed["report_id"] == data["report_id"]
    assert parsed["result"]["presentation_result"] == "Likely Real / Bonafide"


def test_report_never_claims_confidence_terminology():
    data = _sample_report_data()
    full_text = render_html_report(data).lower() + render_json_report(data).lower()
    assert "model confidence" not in full_text
