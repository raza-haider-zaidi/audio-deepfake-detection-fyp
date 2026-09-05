"""Reusable UI building blocks for the Streamlit presentation layer.

Pure HTML/data-building functions (no `streamlit` import, trivially unit
testable) are kept separate from the small number of functions that
actually call `st.*` to render them. This keeps app/components.py the
single place that assembles markup, while streamlit_app.py stays a thin
navigation router and app/views/*.py stay thin page-flow orchestrators.
"""

from __future__ import annotations

import html as _html

from app.styles import STATE_TOKENS

# ---------------------------------------------------------------------------
# Brand mark -- an original, simple inline SVG signal/waveform glyph.
# No external assets, no third-party logo.
# ---------------------------------------------------------------------------


def brand_mark_svg(size: int = 26, color: str = "#4568F2") -> str:
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


def hero_signal_svg(width: int = 220, height: int = 140) -> str:
    """A larger abstract multi-bar signal visualization for the hero
    section -- CSS/SVG only, no stock imagery, no figurative icons."""
    import math

    bars = []
    n = 24
    bar_w = width / (n * 1.6)
    gap = bar_w * 0.6
    for i in range(n):
        h = height * (0.25 + 0.55 * abs(math.sin(i * 0.45)) * (0.6 + 0.4 * math.cos(i * 0.2)))
        x = i * (bar_w + gap)
        y = (height - h) / 2
        opacity = 0.35 + 0.65 * (i / n)
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="{bar_w/2:.1f}" fill="url(#adfHeroGrad)" opacity="{opacity:.2f}"/>')
    return (
        f'<svg width="100%" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-label="Abstract audio signal visualization">'
        '<defs><linearGradient id="adfHeroGrad" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#4568F2"/><stop offset="100%" stop-color="#7C6CF2"/>'
        "</linearGradient></defs>"
        f"{''.join(bars)}</svg>"
    )


# ---------------------------------------------------------------------------
# Pure markup builders (testable without a Streamlit runtime).
# ---------------------------------------------------------------------------


def probability_bar_html(label: str, value: float, color: str) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return (
        '<div class="adf-prob-row">'
        f'<span class="adf-prob-label">{label}</span>'
        f'<div class="adf-prob-track" role="img" aria-label="{label} probability {pct:.1f} percent">'
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
    tokens = STATE_TOKENS.get(state, STATE_TOKENS["INCONCLUSIVE"])
    style_vars = f"--adf-state-fg:{tokens['fg']};--adf-state-bg:{tokens['bg']};--adf-state-border:{tokens['border']};"
    return (
        f'<div class="adf-result" style="{style_vars}">'
        '<div class="adf-result-eyebrow">ANALYSIS RESULT</div>'
        f'<div class="adf-result-title">{title}</div>'
        f'<div class="adf-result-explain">{explanation}</div>'
        f"{extra_html}"
        "</div>"
    )


def status_badge_html(state: str, label: str) -> str:
    tokens = STATE_TOKENS.get(state, STATE_TOKENS["INCONCLUSIVE"])
    style_vars = f"--adf-state-fg:{tokens['fg']};--adf-state-bg:{tokens['bg']};--adf-state-border:{tokens['border']};"
    return f'<span class="adf-status-badge" style="{style_vars}">{_html.escape(label)}</span>'


def metrics_row_html(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="adf-metric"><div class="adf-metric-value">{value}</div>'
        f'<div class="adf-metric-label">{label}</div></div>'
        for label, value in items
    )
    return f'<div class="adf-metrics-row">{cells}</div>'


def metric_card_html(value: str, label: str) -> str:
    return f'<div class="adf-metric-card"><div class="value">{_html.escape(str(value))}</div><div class="label">{_html.escape(label)}</div></div>'


def feature_card_html(number: str, title: str, body: str, url_path: str | None = None) -> str:
    tag = "a" if url_path else "div"
    href = f' href="{url_path}"' if url_path else ""
    return (
        f'<{tag} class="adf-feature-card"{href}>'
        f'<div class="adf-feature-number">{number}</div>'
        f'<div class="adf-feature-title">{_html.escape(title)}</div>'
        f'<div class="adf-feature-body">{_html.escape(body)}</div>'
        f"</{tag}>"
    )


def feature_grid_html(cards: list[tuple[str, str, str, str | None]]) -> str:
    return '<div class="adf-feature-grid">' + "".join(feature_card_html(*c) for c in cards) + "</div>"


def step_flow_html(steps: list[tuple[str, str, str]], tags: list[str] | None = None) -> str:
    """steps: list of (number, title, body). `tags`, if given, is a
    parallel list of "native" | "project" | "presentation" for the
    model-native vs. project-implemented vs. presentation-logic legend."""
    cards = []
    for i, (n, title, body) in enumerate(steps):
        tag_html = ""
        if tags and i < len(tags):
            tag = tags[i]
            tag_label = {"native": "Model-native", "project": "Project-implemented", "presentation": "Presentation logic"}.get(tag, tag)
            tag_html = f'<span class="adf-step-tag adf-step-tag--{tag}">{tag_label}</span>'
        cards.append(
            '<div class="adf-step">'
            f'<div class="adf-step-number">{n}</div>'
            f'<div class="adf-step-title">{title}</div>'
            f'<div class="adf-step-body">{body}</div>'
            f"{tag_html}"
            "</div>"
        )
    return f'<div class="adf-steps">{"".join(cards)}</div>'


def status_chip_html(value: str) -> str:
    """A compact semantic chip for a table status/result cell. Falls back
    to a neutral chip for any value not in STATUS_CHIP_TOKENS (e.g. a
    filename or a free-text label accidentally passed here) -- text is
    always shown, color is never the only signal."""
    from app.styles import STATUS_CHIP_TOKENS

    key = str(value).strip().upper()
    tokens = STATUS_CHIP_TOKENS.get(key)
    if tokens is None:
        return _html.escape(str(value))
    style_vars = f"--adf-chip-fg:{tokens['fg']};--adf-chip-bg:{tokens['bg']};--adf-chip-border:{tokens['border']};"
    return f'<span class="adf-chip" style="{style_vars}">{_html.escape(str(value))}</span>'


def professional_table_html(
    columns: list[str],
    rows: list[dict],
    *,
    numeric_columns: frozenset[str] | set[str] = frozenset(),
    status_columns: frozenset[str] | set[str] = frozenset(),
    stack_on_mobile: bool = False,
    empty_text: str = "—",
) -> str:
    """Builds a single professionally-styled HTML table (`.adf-table`)
    shared by every tabular result in the app (segment evidence, batch
    analysis, robustness, evaluation comparisons, session history) so no
    page hand-rolls its own table markup or relies on the default
    Streamlit dataframe look.

    `numeric_columns` right-aligns and mono-spaces a column (percentages,
    durations, counts). `status_columns` renders values in that column as
    semantic chips via `status_chip_html` (BONAFIDE/SPOOF/INCONCLUSIVE/
    GOOD/LIMITED/POOR/etc.) -- any other value degrades to plain escaped
    text, never a blank cell.
    """
    header_cells = "".join(
        f'<th class="{"adf-th-num" if col in numeric_columns else ""}">{_html.escape(col)}</th>' for col in columns
    )
    body_rows = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if value is None or value == "":
                cells.append(f'<td class="{"adf-td-num" if col in numeric_columns else ""}">{empty_text}</td>')
                continue
            if col in status_columns:
                cells.append(f"<td>{status_chip_html(value)}</td>")
            else:
                css_class = "adf-td-num" if col in numeric_columns else ""
                cells.append(f'<td class="{css_class}">{_html.escape(str(value))}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    table_class = "adf-table adf-table--stack-on-mobile" if stack_on_mobile else "adf-table"
    table_html = f'<table class="{table_class}"><thead><tr>{header_cells}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'

    kv_cards_html = ""
    if stack_on_mobile:
        cards = []
        for row in rows:
            kv_rows = "".join(
                f'<div class="adf-kv-row"><span class="adf-kv-key">{_html.escape(col)}</span>'
                f'<span class="adf-kv-val">{status_chip_html(row.get(col)) if col in status_columns and row.get(col) not in (None, "") else _html.escape(str(row.get(col))) if row.get(col) not in (None, "") else empty_text}</span></div>'
                for col in columns
            )
            cards.append(f'<div class="adf-kv-card">{kv_rows}</div>')
        kv_cards_html = "".join(cards)

    return f'<div class="adf-table-wrap">{table_html}</div>{kv_cards_html}'


def info_callout_html(text: str) -> str:
    return f'<div class="adf-callout">{text}</div>'


def empty_state_html(title: str, body: str) -> str:
    return (
        '<div class="adf-empty-state">'
        f'<div class="adf-section-title">{_html.escape(title)}</div>'
        f'<div class="adf-section-sub">{body}</div>'
        "</div>"
    )


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

CAPABILITY_STRIP = [
    "Audio Analysis",
    "Segment Evidence",
    "Audio Diagnostics",
    "Robustness Testing",
    "Research Evaluation",
    "Exportable Reports",
]

PLATFORM_FEATURE_CARDS = [
    ("01", "Audio Analysis", "Classify uploaded speech using the deployed Spectra detector.", "analyze"),
    ("02", "Segment Evidence", "Inspect how spoof scores vary across the recording.", "analyze"),
    ("03", "Audio Diagnostics", "Review codec, signal and recording-condition information.", "analyze"),
    ("04", "Robustness Analysis", "Test prediction stability under controlled degradation.", "robustness"),
    ("05", "Research Evaluation", "Inspect project-measured performance and model comparisons.", "evaluation"),
    ("06", "Analysis Reports", "Export reproducible technical analysis summaries.", "analyze"),
]


# ---------------------------------------------------------------------------
# Streamlit-rendering wrappers (thin -- all real markup logic lives above).
# ---------------------------------------------------------------------------


def render_page_header() -> None:
    import streamlit as st

    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.75rem;">'
        f"{brand_mark_svg()}"
        '<span style="font-weight:750;font-size:1.05rem;color:#1B2440;">Voice Analysis</span>'
        '<span class="adf-badge" style="margin-left:auto;">'
        '<span class="adf-badge-dot"></span>Research Prototype</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    import streamlit as st

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            '<div class="adf-eyebrow">Audio Deepfake Analysis</div>'
            '<div class="adf-headline">Detect synthetic and cloned speech <span class="accent">with evidence you can inspect</span>.</div>'
            '<div class="adf-subtext">Analyze voice recordings using a CPU-deployed anti-spoofing system, inspect '
            "segment-level evidence, evaluate recording conditions, and explore the project's measured research "
            "performance.</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="padding-top:1rem;">{hero_signal_svg()}</div>',
            unsafe_allow_html=True,
        )


def render_capability_strip() -> None:
    import streamlit as st

    cols = st.columns(len(CAPABILITY_STRIP))
    for col, label in zip(cols, CAPABILITY_STRIP):
        col.markdown(f'<div class="adf-badge" style="justify-content:center;width:100%;">{label}</div>', unsafe_allow_html=True)


def render_feature_grid() -> None:
    import streamlit as st

    render_section_title("Platform capabilities")
    st.markdown(feature_grid_html(PLATFORM_FEATURE_CARDS), unsafe_allow_html=True)


def render_section_title(title: str, subtitle: str | None = None) -> None:
    import streamlit as st

    markup = f'<div class="adf-section-title">{title}</div>'
    if subtitle:
        markup += f'<div class="adf-section-sub">{subtitle}</div>'
    st.markdown(markup, unsafe_allow_html=True)


def render_result_panel(state: str, title: str, explanation: str, extra_html: str = "") -> None:
    import streamlit as st

    st.markdown(result_panel_html(state, title, explanation, extra_html), unsafe_allow_html=True)


def render_probability_comparison(bonafide_prob: float, spoof_prob: float) -> None:
    import streamlit as st

    st.markdown(probability_comparison_html(bonafide_prob, spoof_prob), unsafe_allow_html=True)


def render_metrics_row(items: list[tuple[str, str]]) -> None:
    import streamlit as st

    st.markdown(metrics_row_html(items), unsafe_allow_html=True)


def render_metric_cards(items: list[tuple[str, str]]) -> None:
    import streamlit as st

    cols = st.columns(len(items))
    for col, (value, label) in zip(cols, items):
        col.markdown(metric_card_html(value, label), unsafe_allow_html=True)


def render_metadata_grid(rows: list[tuple[str, str]]) -> None:
    import streamlit as st

    table = "\n".join(f"| {label} | {value} |" for label, value in rows)
    st.markdown(f"| | |\n|---|---|\n{table}")


def render_evidence_panel(sentences: list[str]) -> None:
    import streamlit as st

    for sentence in sentences:
        st.markdown(f"- {sentence}")


def render_professional_table(
    columns: list[str],
    rows: list[dict],
    *,
    numeric_columns: frozenset[str] | set[str] = frozenset(),
    status_columns: frozenset[str] | set[str] = frozenset(),
    stack_on_mobile: bool = False,
) -> None:
    """The single reusable table renderer for every result table in the
    app -- see professional_table_html for the styling contract. Use this
    instead of st.dataframe wherever results are shown."""
    import streamlit as st

    st.markdown(
        professional_table_html(
            columns, rows, numeric_columns=numeric_columns, status_columns=status_columns, stack_on_mobile=stack_on_mobile
        ),
        unsafe_allow_html=True,
    )


def render_info_callout(text: str) -> None:
    import streamlit as st

    st.markdown(info_callout_html(text), unsafe_allow_html=True)


def render_empty_state(title: str, body: str) -> None:
    import streamlit as st

    st.markdown(empty_state_html(title, body), unsafe_allow_html=True)


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
