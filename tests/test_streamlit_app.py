"""Streamlit AppTest-based smoke tests for streamlit_app.py.

These verify the app renders correctly and does NOT load the model just
from an initial page render — that only happens lazily when "Analyze
Audio" is clicked (covered by the integration test, since it requires a
real model download). Also verifies the redesigned copy/structure from
the UI/UX design phase (docs/ui_ux_design.md) replaced the outdated
Wav2Vec2-era text.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    return at


def _all_markdown_text(app) -> str:
    return " ".join(m.value for m in app.markdown)


def test_app_renders_without_error(app):
    assert not app.exception


def test_app_headline_present(app):
    text = _all_markdown_text(app)
    assert "Detect AI-generated" in text


def test_app_has_file_uploader(app):
    assert len(app.get("file_uploader")) == 1


def test_app_initial_render_has_no_result(app):
    text = _all_markdown_text(app)
    assert "ANALYSIS RESULT" not in text


def test_app_initial_render_does_not_load_model(app):
    # get_detector() is only reachable from inside the Analyze Audio button
    # handler; on initial render (no upload, no click) it must never be
    # invoked. We check indirectly: no technical-details expander exists yet.
    expander_labels = [e.label for e in app.expander]
    assert "Technical details" not in expander_labels


def test_app_shows_disclaimer(app):
    text = _all_markdown_text(app).lower()
    assert "research prototype" in text


def test_app_shows_research_prototype_badge(app):
    text = _all_markdown_text(app)
    assert "Research Prototype" in text


def test_app_how_it_works_expander_present(app):
    expander_labels = [e.label for e in app.expander]
    assert any("Spectra-AASIST3" in label for label in expander_labels)


def test_app_no_outdated_wav2vec2_copy(app):
    text = _all_markdown_text(app)
    expander_labels = [e.label for e in app.expander]
    assert "Why Wav2Vec2?" not in expander_labels
    assert "Wav2Vec2 Audio Deepfake Detector" not in text


def test_app_evaluation_expander_present(app):
    expander_labels = [e.label for e in app.expander]
    assert "Evaluation" in expander_labels


def test_app_footer_mentions_spectra(app):
    text = _all_markdown_text(app)
    assert "Spectra-AASIST3 INT8" in text
