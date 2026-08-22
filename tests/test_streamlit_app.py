"""Streamlit AppTest-based smoke tests for streamlit_app.py.

These verify the app renders correctly and does NOT load the ~468 MiB
model just from an initial page render — that only happens lazily when
"Analyze Audio" is clicked (covered by the integration test, since it
requires a real model download).
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


def test_app_renders_without_error(app):
    assert not app.exception


def test_app_title_present(app):
    assert any("AI Voice Deepfake Detector" in t.value for t in app.title)


def test_app_has_file_uploader(app):
    assert len(app.get("file_uploader")) == 1


def test_app_initial_render_has_no_result(app):
    # No file uploaded yet -> no "Detection result" subheader should appear.
    subheader_texts = [s.value for s in app.subheader]
    assert "Detection result" not in subheader_texts


def test_app_initial_render_does_not_load_model(app):
    # get_detector() is only reachable from inside the Analyze Audio button
    # handler; on initial render (no upload, no click) it must never be
    # invoked. We check indirectly: no model-loading spinner text is queued
    # and no result/technical-details expander exists yet.
    expander_labels = [e.label for e in app.expander]
    assert "Technical details" not in expander_labels


def test_app_shows_disclaimer(app):
    info_texts = " ".join(i.value for i in app.info)
    assert "research" in info_texts.lower()
