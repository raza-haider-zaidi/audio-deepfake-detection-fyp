"""Tests for app/views/batch.py -- the sequential-only Batch Analysis
page. Only tests the pure logic (file-count cap, HTML report builder);
the Streamlit widget flow is covered by manual/visual QA since it
requires real file uploads."""

from app.views.batch import MAX_BATCH_FILES, _render_batch_report_html


def test_max_batch_files_is_conservative():
    assert 1 <= MAX_BATCH_FILES <= 10


def test_render_batch_report_html_includes_disclaimer_and_rows():
    results = [
        {
            "File": "a.wav",
            "Duration (s)": 3.0,
            "Presentation result": "Bonafide",
            "Bonafide probability": 0.9,
            "Spoof probability": 0.1,
            "Analysis conditions": "Good",
            "Inference time (ms)": 120.0,
            "SHA-256": "abc123",
        }
    ]
    html = _render_batch_report_html(results, {"display_name": "Spectra-AASIST3 INT8"})
    assert "a.wav" in html
    assert "Research prototype" in html
    assert "forensic" in html.lower()


def test_render_batch_report_html_handles_empty_results():
    html = _render_batch_report_html([], {"display_name": "Spectra-AASIST3 INT8"})
    assert "<table>" in html


def test_render_batch_report_html_escapes_none_as_dash():
    results = [
        {
            "File": "b.wav",
            "Duration (s)": None,
            "Presentation result": "Error: analysis failed",
            "Bonafide probability": None,
            "Spoof probability": None,
            "Analysis conditions": None,
            "Inference time (ms)": None,
            "SHA-256": "def456",
        }
    ]
    html = _render_batch_report_html(results, {"display_name": "Spectra-AASIST3 INT8"})
    assert "—" in html
