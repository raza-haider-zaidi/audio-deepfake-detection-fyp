"""Factories for the callable pages used by the top-level router.

Page instances are deliberately created per script run. Streamlit marks the
page returned from ``st.navigation`` as callable only once, so sharing those
mutable instances between sessions can make one run consume another run's
page-call permission.
"""

from __future__ import annotations

import streamlit as st

from app.views import about, analyze, batch, evaluation, methodology, robustness

def create_robustness_page() -> st.Page:
    """Create the Robustness destination for navigation or a page link."""
    return st.Page(
        robustness.render,
        title="Robustness",
        url_path="robustness",
        icon=":material/science:",
    )


def create_navigation_pages() -> list[st.Page]:
    """Return fresh page objects for one Streamlit script run."""
    return [
        st.Page(
            analyze.render,
            title="Analyze",
            url_path="analyze",
            default=True,
            icon=":material/graphic_eq:",
        ),
        st.Page(batch.render, title="Batch", url_path="batch", icon=":material/dataset:"),
        create_robustness_page(),
        st.Page(
            evaluation.render,
            title="Evaluation",
            url_path="evaluation",
            icon=":material/monitoring:",
        ),
        st.Page(
            methodology.render,
            title="Methodology",
            url_path="methodology",
            icon=":material/schema:",
        ),
        st.Page(about.render, title="About", url_path="about", icon=":material/info:"),
    ]
