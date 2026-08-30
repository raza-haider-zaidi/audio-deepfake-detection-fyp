"""Cross-cutting language guards for the analysis-platform expansion:
no "confidence" terminology, no unsupported authenticity claims, and the
About page reflects the actual active model configuration."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.model_loader import DEPLOYMENT_MODEL_ID
from audio_deepfake_detector.config.models_config import load_models_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_AUTHENTICITY_CLAIMS = ("verified authentic", "confirmed deepfake", "forensically validated")


def _run_app() -> AppTest:
    at = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"))
    at.run(timeout=30)
    assert not at.exception
    return at


def _page(view_module: str) -> None:
    import importlib

    module = importlib.import_module(view_module)
    module.render()


def _run_view(view_module: str) -> AppTest:
    at = AppTest.from_function(_page, args=(view_module,))
    at.run(timeout=30)
    assert not at.exception
    return at


def _all_text(at: AppTest) -> str:
    text = " ".join(m.value for m in at.markdown)
    text += " ".join(c.value for c in at.caption)
    return text


def test_analyze_page_never_says_model_confidence():
    at = _run_app()
    assert "model confidence" not in _all_text(at).lower()


def test_about_page_never_makes_unsupported_authenticity_claims():
    at = _run_view("app.views.about")
    text = _all_text(at).lower()
    for phrase in FORBIDDEN_AUTHENTICITY_CLAIMS:
        assert phrase not in text


def test_about_page_shows_active_model_repository_and_revision():
    at = _run_view("app.views.about")
    config = load_models_config().get(DEPLOYMENT_MODEL_ID)
    text = _all_text(at)
    assert config.repository in text
    assert config.revision in text


def test_about_page_states_unpublished_status():
    at = _run_view("app.views.about")
    text = _all_text(at).lower()
    assert "pre-release" in text or "unpublished" in text


def test_evaluation_page_labels_project_measured_not_universal():
    at = _run_view("app.views.evaluation")
    text = _all_text(at).lower()
    assert "project-measured" in text
    assert "99% accurate" not in text
    assert "universal" not in text or "not a claim of universal accuracy" in text


def test_methodology_page_distinguishes_project_extension_from_model_native():
    at = _run_view("app.views.methodology")
    text = _all_text(at)
    assert "project-level" in text or "application-level extension" in text
