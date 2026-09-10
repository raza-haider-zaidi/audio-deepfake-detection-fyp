"""AI Voice Deepfake Detector — Streamlit entrypoint / top-level router.

Presentation layer only. All preprocessing/model/inference logic lives
under src/audio_deepfake_detector; this file (and the helpers under app/)
never implement model logic directly. Visual design lives in
app/styles.py (tokens + CSS) and app/components.py (reusable markup).
Supplementary, descriptive analysis (audio quality, segment evidence,
evidence summary, robustness, reporting) lives under app/analysis/ and
app/reporting/ -- see docs/analysis_platform_v2.md for the full
architecture and the explicit statement that none of it alters the
frozen Spectra-AASIST3 classification.

Navigation is native Streamlit top navigation (st.navigation with
position="top"), NOT the default sidebar and NOT the legacy file-based
pages/ multipage convention. Each destination is a thin `render()`
function under app/views/. Fresh page objects are built for every script run
by app/nav.py so page execution state is never shared between sessions.

Run locally with:
    .\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.components import render_page_header  # noqa: E402
from app.styles import inject_global_styles  # noqa: E402

st.set_page_config(
    page_title="Voice Analysis — Audio Deepfake Detection",
    page_icon=":material/graphic_eq:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(inject_global_styles(), unsafe_allow_html=True)

from app.nav import create_navigation_pages  # noqa: E402

render_page_header()

page = st.navigation(create_navigation_pages(), position="top")
page.run()
