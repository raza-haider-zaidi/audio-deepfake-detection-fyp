"""Centralized design tokens and injected CSS for the Streamlit UI.

Single source of truth for color/spacing/typography so streamlit_app.py
and app/components.py never hardcode arbitrary hex values or ad-hoc CSS.
Only ONE stylesheet is injected (`inject_global_styles()`), scoped to
custom, stable class names (`.adf-*`) wherever practical rather than
targeting Streamlit's own auto-generated DOM structure, which can change
between versions.

No external fonts, no CDN assets, no JS frameworks -- see
docs/ui_ux_design.md for the full rationale.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color tokens -- sophisticated dark interface, restrained semantic accents.
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0b0f14",  # page background -- very dark navy/graphite
    "surface": "#131922",  # primary elevated surface (cards, panels)
    "surface_alt": "#1a2230",  # secondary/nested surface
    "border": "#232c3a",  # subtle hairline borders
    "border_strong": "#2e3a4d",
    "text": "#e8ecf1",  # off-white, not pure white
    "text_muted": "#8b96a8",  # cool gray
    "text_faint": "#5c6b80",
    "accent": "#4f8ff7",  # controlled electric/cool blue
    "accent_strong": "#6fa4ff",
    "accent_cyan": "#38c6d9",  # secondary accent, used sparingly
    "success": "#3ecf8e",  # REAL -- subtle green/teal
    "success_bg": "rgba(62, 207, 142, 0.10)",
    "success_border": "rgba(62, 207, 142, 0.35)",
    "danger": "#f2765a",  # SPOOF -- controlled coral/red
    "danger_bg": "rgba(242, 118, 90, 0.10)",
    "danger_border": "rgba(242, 118, 90, 0.35)",
    "warning": "#e0ad4f",  # INCONCLUSIVE -- amber/gold
    "warning_bg": "rgba(224, 173, 79, 0.10)",
    "warning_border": "rgba(224, 173, 79, 0.35)",
}

SPACING = {
    "xs": "0.375rem",
    "sm": "0.75rem",
    "md": "1.25rem",
    "lg": "2rem",
    "xl": "3rem",
}

RADIUS = {
    "sm": "6px",
    "md": "10px",
    "lg": "14px",
}

SHADOW = {
    "card": "0 1px 2px rgba(0, 0, 0, 0.4), 0 8px 24px -12px rgba(0, 0, 0, 0.5)",
}

CONTENT_MAX_WIDTH = "1040px"

FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)
MONO_STACK = (
    "ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', "
    "Menlo, monospace"
)

# Semantic-state -> token lookup, used by components.py so no widget ever
# hardcodes a raw color for BONAFIDE/SPOOF/INCONCLUSIVE.
STATE_TOKENS = {
    "BONAFIDE": {"fg": COLORS["success"], "bg": COLORS["success_bg"], "border": COLORS["success_border"]},
    "SPOOF": {"fg": COLORS["danger"], "bg": COLORS["danger_bg"], "border": COLORS["danger_border"]},
    "INCONCLUSIVE": {"fg": COLORS["warning"], "bg": COLORS["warning_bg"], "border": COLORS["warning_border"]},
}


def inject_global_styles() -> str:
    """Return the single global <style> block. Call once via st.markdown
    with unsafe_allow_html=True. Kept as a plain function (not a Streamlit
    call) so it stays trivially unit-testable without a Streamlit runtime.
    """
    c = COLORS
    return f"""
<style>
/* ==========================================================================
   1. Page shell -- background, max content width, base typography.
   ========================================================================== */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {c['bg']} !important;
    color: {c['text']};
    font-family: {FONT_STACK};
}}
[data-testid="stHeader"] {{
    background-color: transparent;
}}
.block-container {{
    max-width: {CONTENT_MAX_WIDTH};
    padding-top: {SPACING['lg']};
    padding-bottom: {SPACING['xl']};
}}
#MainMenu, footer[data-testid="stFooter"] {{
    visibility: hidden;
    height: 0;
}}

/* ==========================================================================
   2. Typography hierarchy.
   ========================================================================== */
.adf-eyebrow {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {c['accent_strong']};
    margin-bottom: {SPACING['xs']};
}}
.adf-headline {{
    font-size: 2.15rem;
    line-height: 1.18;
    font-weight: 650;
    color: {c['text']};
    margin: 0 0 {SPACING['sm']} 0;
    letter-spacing: -0.01em;
}}
.adf-subtext {{
    font-size: 1.02rem;
    color: {c['text_muted']};
    max-width: 640px;
    line-height: 1.55;
    margin-bottom: {SPACING['md']};
}}
.adf-section-title {{
    font-size: 1.05rem;
    font-weight: 650;
    color: {c['text']};
    margin: 0 0 0.25rem 0;
}}
.adf-section-sub {{
    font-size: 0.88rem;
    color: {c['text_muted']};
    margin-bottom: {SPACING['sm']};
}}
.adf-mono {{
    font-family: {MONO_STACK};
    font-size: 0.82rem;
    color: {c['text_muted']};
}}
.adf-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {c['text_muted']};
    background: {c['surface_alt']};
    border: 1px solid {c['border']};
    border-radius: 999px;
    padding: 0.3rem 0.75rem;
}}
.adf-badge-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {c['accent']};
}}

/* ==========================================================================
   3. Cards / surfaces.
   ========================================================================== */
.adf-card {{
    background: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS['lg']};
    padding: {SPACING['md']};
    box-shadow: {SHADOW['card']};
    transition: border-color 0.15s ease;
}}
.adf-card--flush {{
    padding: 0;
    overflow: hidden;
}}

/* ==========================================================================
   4. Result panel -- semantic state via left accent bar + tinted background,
      never a full-page background flood.
   ========================================================================== */
.adf-result {{
    border-radius: {RADIUS['lg']};
    border: 1px solid var(--adf-state-border);
    background: var(--adf-state-bg);
    padding: {SPACING['lg']};
    position: relative;
    overflow: hidden;
    animation: adf-fade-in 0.25s ease;
}}
.adf-result::before {{
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--adf-state-fg);
}}
.adf-result-eyebrow {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--adf-state-fg);
    margin-bottom: 0.5rem;
}}
.adf-result-title {{
    font-size: 1.55rem;
    font-weight: 650;
    color: {c['text']};
    margin-bottom: 0.5rem;
    letter-spacing: -0.01em;
}}
.adf-result-explain {{
    color: {c['text_muted']};
    font-size: 0.95rem;
    line-height: 1.5;
    max-width: 620px;
}}

/* ==========================================================================
   5. Probability bars.
   ========================================================================== */
.adf-prob-row {{
    display: flex;
    align-items: center;
    gap: {SPACING['sm']};
    margin-bottom: {SPACING['xs']};
}}
.adf-prob-label {{
    width: 92px;
    flex-shrink: 0;
    font-size: 0.85rem;
    color: {c['text_muted']};
    font-weight: 550;
}}
.adf-prob-track {{
    flex-grow: 1;
    height: 8px;
    border-radius: 999px;
    background: {c['surface_alt']};
    overflow: hidden;
}}
.adf-prob-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.35s ease;
}}
.adf-prob-value {{
    width: 54px;
    flex-shrink: 0;
    text-align: right;
    font-family: {MONO_STACK};
    font-size: 0.85rem;
    color: {c['text']};
}}

/* ==========================================================================
   6. Threshold visualization (INCONCLUSIVE only).
   ========================================================================== */
.adf-threshold-rail {{
    position: relative;
    height: 6px;
    border-radius: 999px;
    background: {c['surface_alt']};
    margin: {SPACING['sm']} 0 0.5rem 0;
}}
.adf-threshold-fill {{
    position: absolute;
    left: 0; top: 0; bottom: 0;
    border-radius: 999px;
    background: {c['warning']};
}}
.adf-threshold-marker {{
    position: absolute;
    top: -4px;
    width: 2px;
    height: 14px;
    background: {c['text']};
}}

/* ==========================================================================
   7. Metrics row.
   ========================================================================== */
.adf-metrics-row {{
    display: flex;
    flex-wrap: wrap;
    gap: {SPACING['lg']};
    margin-top: {SPACING['sm']};
}}
.adf-metric {{
    min-width: 110px;
}}
.adf-metric-value {{
    font-size: 1.15rem;
    font-weight: 650;
    color: {c['text']};
    font-family: {MONO_STACK};
}}
.adf-metric-label {{
    font-size: 0.76rem;
    color: {c['text_faint']};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.15rem;
}}

/* ==========================================================================
   8. Step flow ("How it works").
   ========================================================================== */
.adf-steps {{
    display: flex;
    flex-wrap: wrap;
    gap: {SPACING['sm']};
}}
.adf-step {{
    flex: 1 1 200px;
    background: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS['md']};
    padding: {SPACING['sm']} {SPACING['md']};
}}
.adf-step-number {{
    font-family: {MONO_STACK};
    font-size: 0.78rem;
    color: {c['accent_strong']};
    margin-bottom: 0.3rem;
}}
.adf-step-title {{
    font-weight: 600;
    font-size: 0.92rem;
    color: {c['text']};
    margin-bottom: 0.2rem;
}}
.adf-step-body {{
    font-size: 0.82rem;
    color: {c['text_muted']};
    line-height: 1.45;
}}

/* ==========================================================================
   9. Footer.
   ========================================================================== */
.adf-footer {{
    margin-top: {SPACING['xl']};
    padding-top: {SPACING['md']};
    border-top: 1px solid {c['border']};
    color: {c['text_faint']};
    font-size: 0.8rem;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: {SPACING['sm']};
}}

/* ==========================================================================
   10. Streamlit widget restyling -- minimal, targets stable public
       selectors only (buttons, uploader, metrics, expander).
   ========================================================================== */
.stButton > button {{
    background: {c['accent']};
    color: #ffffff;
    border: none;
    border-radius: {RADIUS['sm']};
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    transition: background 0.15s ease, transform 0.1s ease;
}}
.stButton > button:hover {{
    background: {c['accent_strong']};
}}
.stButton > button:focus-visible {{
    outline: 2px solid {c['accent_strong']};
    outline-offset: 2px;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: {c['surface']} !important;
    border: 1.5px dashed {c['border_strong']} !important;
    border-radius: {RADIUS['lg']} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: {MONO_STACK};
}}
[data-testid="stExpander"] {{
    border: 1px solid {c['border']} !important;
    border-radius: {RADIUS['md']} !important;
    background: {c['surface']} !important;
}}
[data-testid="stAlert"] {{
    border-radius: {RADIUS['md']};
}}

/* ==========================================================================
   11. Motion -- subtle only, respects prefers-reduced-motion.
   ========================================================================== */
@keyframes adf-fade-in {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@media (prefers-reduced-motion: reduce) {{
    .adf-result {{ animation: none; }}
    .stButton > button, .adf-prob-fill {{ transition: none; }}
}}

/* ==========================================================================
   12. Responsive adjustments.
   ========================================================================== */
@media (max-width: 640px) {{
    .adf-headline {{ font-size: 1.6rem; }}
    .adf-result-title {{ font-size: 1.25rem; }}
    .adf-prob-label {{ width: 68px; font-size: 0.78rem; }}
    .adf-metrics-row {{ gap: {SPACING['md']}; }}
    .adf-step {{ flex: 1 1 100%; }}
}}
</style>
"""
