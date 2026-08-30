"""Reusable UI building blocks for the Streamlit presentation layer.

Pure HTML/data-building functions (no `streamlit` import, trivially unit
testable) are kept separate from the small number of functions that
actually call `st.*` to render them. This keeps app/components.py the
single place that assembles markup, while streamlit_app.py stays a thin
page-flow orchestrator.
"""

from __future__ import annotations

from app.styles import STATE_TOKENS

# ---------------------------------------------------------------------------
# Brand mark -- an original, simple inline SVG signal/waveform glyph.
# No external assets, no third-party logo.
# ---------------------------------------------------------------------------


def brand_mark_svg(size: int = 26, color: str = "#4f8ff7") -> str:
    """A minimal abstract waveform mark: five bars of varying height,
    evoking an audio signal without any figurative (brain/robot/shield)
    imagery."""
    heights = [6, 14, 22, 14, 8]
    bar_width = 3
    gap = 2
    total_width = len(heights) * bar_width + (len(heights) - 1) * gap
    svg_height = 24
    bars = []
    x = 0
    for h in heights:
        y = (svg_height - h) / 2
        bars.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{h}" rx="1.5" fill="{color}"/>')
        x += bar_width + gap
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {total_width} {svg_height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Audio signal mark">'
        f"{''.join(bars)}</svg>"
    )


# ---------------------------------------------------------------------------
# Pure markup builders (testable without a Streamlit runtime).
# ---------------------------------------------------------------------------


def probability_bar_html(label: str, value: float, color: str) -> str:
    """One labeled horizontal probability bar. `value` is 0-1."""
    pct = max(0.0, min(1.0, value)) * 100
    return (
        '<div class="adf-prob-row">'
        f'<span class="adf-prob-label">{label}</span>'
        '<div class="adf-prob-track" role="img" '
        f'aria-label="{label} probability {pct:.1f} percent">'
        f'<div class="adf-prob-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        "</div>"
        f'<span class="adf-prob-value">{pct:.1f}%</span>'
        "</div>"
    )


def probability_comparison_html(bonafide_prob: float, spoof_prob: float) -> str:
    from app.styles import COLORS

    return probability_bar_html("Bonafide", bonafide_prob, COLORS["success"]) + probability_bar_html(
        "Spoof", spoof_prob, COLORS["danger"]
    )


def threshold_visualization_html(observed_spoof_prob: float, threshold: float) -> str:
    """A single rail: filled up to the observed spoof probability, with a
    marker at the calibrated threshold. Only used for INCONCLUSIVE."""
    observed_pct = max(0.0, min(1.0, observed_spoof_prob)) * 100
    threshold_pct = max(0.0, min(1.0, threshold)) * 100
    return (
        '<div class="adf-threshold-rail">'
        f'<div class="adf-threshold-fill" style="width:{observed_pct:.1f}%;"></div>'
        f'<div class="adf-threshold-marker" style="left:{threshold_pct:.1f}%;" '
        f'title="Calibrated spoof threshold {threshold_pct:.1f}%"></div>'
        "</div>"
        '<div class="adf-mono">Observed spoof probability '
        f"{observed_pct:.1f}% &nbsp;·&nbsp; calibrated spoof threshold {threshold_pct:.1f}%</div>"
    )


def result_panel_html(state: str, title: str, explanation: str, extra_html: str = "") -> str:
    """The single result panel. `state` must be one of STATE_TOKENS' keys."""
    tokens = STATE_TOKENS.get(state, STATE_TOKENS["INCONCLUSIVE"])
    eyebrow = "ANALYSIS RESULT"
    style_vars = (
        f"--adf-state-fg:{tokens['fg']};--adf-state-bg:{tokens['bg']};--adf-state-border:{tokens['border']};"
    )
    return (
        f'<div class="adf-result" style="{style_vars}">'
        f'<div class="adf-result-eyebrow">{eyebrow}</div>'
        f'<div class="adf-result-title">{title}</div>'
        f'<div class="adf-result-explain">{explanation}</div>'
        f"{extra_html}"
        "</div>"
    )


def metrics_row_html(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="adf-metric"><div class="adf-metric-value">{value}</div>'
        f'<div class="adf-metric-label">{label}</div></div>'
        for label, value in items
    )
    return f'<div class="adf-metrics-row">{cells}</div>'


def step_flow_html(steps: list[tuple[str, str, str]]) -> str:
    """steps: list of (number, title, body)."""
    cards = "".join(
        '<div class="adf-step">'
        f'<div class="adf-step-number">{n}</div>'
        f'<div class="adf-step-title">{title}</div>'
        f'<div class="adf-step-body">{body}</div>'
        "</div>"
        for n, title, body in steps
    )
    return f'<div class="adf-steps">{cards}</div>'


HOW_IT_WORKS_STEPS = [
    ("01", "Upload", "Audio is decoded and standardized to a mono 16&nbsp;kHz waveform."),
    ("02", "Preprocess", "The waveform is pre-emphasized and split into the detector's required input segments."),
    ("03", "Analyze", "Spectra-AASIST3 evaluates each segment using ONNX Runtime on CPU."),
    ("04", "Aggregate", "Segment-level evidence is combined into a single clip-level result."),
]

WHY_MODEL_POINTS = [
    "Combines XLS-R-300M speech representations with an AASIST-style anti-spoofing back-end.",
    "Selected after comparative generalization testing against the previously deployed detector.",
    "Deployed as a dynamically quantized INT8 ONNX model, optimized for CPU inference.",
    "Runs without a dedicated GPU, on Streamlit Community Cloud's standard CPU allocation.",
]


# ---------------------------------------------------------------------------
# Streamlit-rendering wrappers (thin -- all real markup logic lives above).
# ---------------------------------------------------------------------------


def render_page_header() -> None:
    import streamlit as st

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.25rem;">'
        f"{brand_mark_svg()}"
        '<span class="adf-eyebrow" style="margin:0;">AI Voice Analysis</span>'
        '<span class="adf-badge" style="margin-left:auto;">'
        '<span class="adf-badge-dot"></span>Research Prototype</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    import streamlit as st

    st.markdown(
        '<div class="adf-headline">Detect AI-generated<br/>and cloned speech.</div>'
        '<div class="adf-subtext">Analyze speech using a CPU-deployed anti-spoofing model designed to '
        "identify characteristics associated with synthetic or manipulated voices.</div>",
        unsafe_allow_html=True,
    )


def render_section_title(title: str, subtitle: str | None = None) -> None:
    import streamlit as st

    html = f'<div class="adf-section-title">{title}</div>'
    if subtitle:
        html += f'<div class="adf-section-sub">{subtitle}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_result_panel(state: str, title: str, explanation: str, extra_html: str = "") -> None:
    import streamlit as st

    st.markdown(result_panel_html(state, title, explanation, extra_html), unsafe_allow_html=True)


def render_probability_comparison(bonafide_prob: float, spoof_prob: float) -> None:
    import streamlit as st

    st.markdown(probability_comparison_html(bonafide_prob, spoof_prob), unsafe_allow_html=True)


def render_metrics_row(items: list[tuple[str, str]]) -> None:
    import streamlit as st

    st.markdown(metrics_row_html(items), unsafe_allow_html=True)


def render_how_it_works() -> None:
    import streamlit as st

    render_section_title("How the analysis works")
    st.markdown(step_flow_html(HOW_IT_WORKS_STEPS), unsafe_allow_html=True)
    st.caption(
        "Multi-segment aggregation is an application-level extension implemented for this "
        "project, not native behavior of the underlying model."
    )


def render_why_model() -> None:
    import streamlit as st

    with st.expander("Why Spectra-AASIST3?"):
        st.markdown("\n".join(f"- {point}" for point in WHY_MODEL_POINTS))
        st.caption(
            "The model is currently pre-release/unpublished; project results were evaluated "
            "independently."
        )


def render_evaluation_section() -> None:
    import streamlit as st

    with st.expander("Evaluation"):
        st.caption("Project-measured evaluation on a fixed held-out balanced In-the-Wild subset.")
        render_metrics_row(
            [
                ("3.0%", "EER"),
                ("0.9819", "ROC-AUC"),
                ("0.9375", "F1"),
                ("2.0%", "Bonafide FPR"),
            ]
        )


def render_disclaimer() -> None:
    import streamlit as st

    st.markdown(
        '<div class="adf-section-sub" style="margin-top:0.5rem;">'
        "<strong>Research prototype.</strong> This detector should not be used as the sole basis "
        "for forensic, legal, security, disciplinary, or identity decisions. Detection "
        "performance can vary with synthesis methods, recording conditions, compression, "
        "language, and background noise."
        "</div>",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    import streamlit as st

    st.markdown(
        '<div class="adf-footer">'
        "<span>Audio Deepfake Detection Research Prototype</span>"
        "<span>CPU inference &middot; ONNX Runtime &middot; Spectra-AASIST3 INT8</span>"
        "</div>",
        unsafe_allow_html=True,
    )
