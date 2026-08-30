"""Shared st.Page definitions -- defined once so both the top-level router
(streamlit_app.py) and individual views (for cross-page st.page_link
calls) reference the exact same Page objects."""

from __future__ import annotations

import streamlit as st

from app.views import about, analyze, batch, evaluation, methodology, robustness

analyze_page = st.Page(analyze.render, title="Analyze", url_path="analyze", default=True, icon=":material/graphic_eq:")
batch_page = st.Page(batch.render, title="Batch", url_path="batch", icon=":material/dataset:")
robustness_page = st.Page(robustness.render, title="Robustness", url_path="robustness", icon=":material/science:")
evaluation_page = st.Page(evaluation.render, title="Evaluation", url_path="evaluation", icon=":material/monitoring:")
methodology_page = st.Page(methodology.render, title="Methodology", url_path="methodology", icon=":material/schema:")
about_page = st.Page(about.render, title="About", url_path="about", icon=":material/info:")

ALL_PAGES = [analyze_page, batch_page, robustness_page, evaluation_page, methodology_page, about_page]
