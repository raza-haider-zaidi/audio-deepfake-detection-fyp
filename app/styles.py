"""Centralized design tokens and injected CSS for the Streamlit UI.

Single source of truth for color/spacing/typography so views/components
never hardcode arbitrary hex values or ad-hoc CSS. Only ONE stylesheet is
injected (`inject_global_styles()`), scoped to custom, stable class names
(`.adf-*`) wherever practical rather than targeting Streamlit's own
auto-generated DOM structure.

Visual direction (docs/analysis_platform_v2.md): a light, cool, technical
SaaS/research-tool palette -- replaces the earlier dark theme entirely.
No external fonts, no CDN assets, no JS frameworks.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color tokens -- light cool technical theme.
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#F5F7FB",  # page background -- light cool gray/blue-gray
    "surface": "#FFFFFF",  # primary card surface
    "surface_alt": "#EEF1F8",  # secondary/nested surface, subtle cool blue-gray
    "border": "#DEE3EE",  # hairline borders
    "border_strong": "#C7CEE0",
    "text": "#1B2440",  # deep navy/graphite
    "text_muted": "#4A5578",  # stronger-presence muted text than a pale gray
    "text_faint": "#7680A0",
    "accent": "#4568F2",  # vibrant cobalt/indigo
    "accent_strong": "#3B5BDB",
    "accent_soft_bg": "rgba(69, 104, 242, 0.08)",
    "accent_cyan": "#0EA5B7",  # secondary accent, used sparingly
    "gradient": "linear-gradient(135deg, #4568F2 0%, #7C6CF2 100%)",  # hero/CTA/selected-nav only
    "success": "#0E9F6E",  # BONAFIDE -- emerald/teal
    "success_bg": "rgba(14, 159, 110, 0.09)",
    "success_border": "rgba(14, 159, 110, 0.35)",
    "danger": "#E23E57",  # SPOOF -- professional red/coral
    "danger_bg": "rgba(226, 62, 87, 0.08)",
    "danger_border": "rgba(226, 62, 87, 0.32)",
    "warning": "#C88A1C",  # INCONCLUSIVE -- amber
    "warning_bg": "rgba(200, 138, 28, 0.10)",
    "warning_border": "rgba(200, 138, 28, 0.35)",
    "info": "#4568F2",  # research/info -- blue/indigo
    "info_bg": "rgba(69, 104, 242, 0.07)",
    "info_border": "rgba(69, 104, 242, 0.28)",
}

SPACING = {"xs": "0.375rem", "sm": "0.75rem", "md": "1.25rem", "lg": "2rem", "xl": "3rem"}
RADIUS = {"sm": "8px", "md": "12px", "lg": "16px"}
SHADOW = {
    "card": "0 1px 2px rgba(27, 36, 64, 0.04), 0 6px 16px -8px rgba(27, 36, 64, 0.10)",
    "card_hover": "0 2px 4px rgba(27, 36, 64, 0.06), 0 14px 28px -10px rgba(27, 36, 64, 0.16)",
    "nav": "0 1px 0 rgba(27, 36, 64, 0.06)",
}
CONTENT_MAX_WIDTH = "1120px"
TRANSITION = "180ms cubic-bezier(0.2, 0.8, 0.2, 1)"

FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO_STACK = "ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace"

STATE_TOKENS = {
    "BONAFIDE": {"fg": COLORS["success"], "bg": COLORS["success_bg"], "border": COLORS["success_border"]},
    "SPOOF": {"fg": COLORS["danger"], "bg": COLORS["danger_bg"], "border": COLORS["danger_border"]},
    "INCONCLUSIVE": {"fg": COLORS["warning"], "bg": COLORS["warning_bg"], "border": COLORS["warning_border"]},
}

# Chip tokens for tabular status cells -- superset of STATE_TOKENS covering
# the "Good/Limited/Poor" analysis-condition vocabulary and generic
# queue/batch status words used across tables.
STATUS_CHIP_TOKENS = {
    **STATE_TOKENS,
    "GOOD": STATE_TOKENS["BONAFIDE"],
    "LIMITED": STATE_TOKENS["INCONCLUSIVE"],
    "POOR": STATE_TOKENS["SPOOF"],
    "COMPLETE": STATE_TOKENS["BONAFIDE"],
    "QUEUED": {"fg": COLORS["text_muted"], "bg": COLORS["surface_alt"], "border": COLORS["border_strong"]},
    "PROCESSING": {"fg": COLORS["accent_strong"], "bg": COLORS["info_bg"], "border": COLORS["info_border"]},
    "FAILED": STATE_TOKENS["SPOOF"],
    "HIGH": STATE_TOKENS["BONAFIDE"],
    "MODERATE": STATE_TOKENS["INCONCLUSIVE"],
    "MIXED": STATE_TOKENS["SPOOF"],
}


def inject_global_styles() -> str:
    c = COLORS
    return f"""
<style>
/* ==========================================================================
   1. Page shell.
   ========================================================================== */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {c['bg']} !important;
    color: {c['text']};
    font-family: {FONT_STACK};
}}
[data-testid="stHeader"] {{ background-color: transparent; }}
.block-container {{
    max-width: {CONTENT_MAX_WIDTH};
    padding-top: {SPACING['md']};
    padding-bottom: {SPACING['xl']};
}}
#MainMenu, footer[data-testid="stFooter"] {{ visibility: hidden; height: 0; }}
/* Sidebar is not used as primary navigation (top nav instead); collapse its
   default chrome so no residual bare "streamlit app" sidebar is visible. */
[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}

/* ==========================================================================
   2. Top navigation (st.navigation(position="top")) restyling.
   ========================================================================== */
div[data-testid="stTopNav"], [data-testid="stAppViewBlockContainer"] > div:first-child nav {{
    background: {c['surface']};
}}
[data-testid="stTopNav"] a, [role="tablist"] button {{
    color: {c['text_muted']} !important;
    font-weight: 550 !important;
    transition: color {TRANSITION} !important;
}}
[data-testid="stTopNav"] a[aria-current="page"], [role="tablist"] button[aria-selected="true"] {{
    color: {c['accent_strong']} !important;
}}

/* ==========================================================================
   3. Typography hierarchy.
   ========================================================================== */
.adf-eyebrow {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: {c['accent_strong']}; margin-bottom: {SPACING['xs']};
}}
.adf-headline {{
    font-size: 2.3rem; line-height: 1.16; font-weight: 700; color: {c['text']};
    margin: 0 0 {SPACING['sm']} 0; letter-spacing: -0.015em;
}}
.adf-headline .accent {{
    background: {c['gradient']}; -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.adf-subtext {{ font-size: 1.05rem; color: {c['text_muted']}; max-width: 660px; line-height: 1.6; margin-bottom: {SPACING['md']}; }}
.adf-section-title {{ font-size: 1.08rem; font-weight: 700; color: {c['text']}; margin: 0 0 0.25rem 0; }}
.adf-section-sub {{ font-size: 0.9rem; color: {c['text_muted']}; margin-bottom: {SPACING['sm']}; }}
.adf-mono {{ font-family: {MONO_STACK}; font-size: 0.82rem; color: {c['text_muted']}; }}
.adf-badge {{
    display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase; color: {c['text_muted']};
    background: {c['surface_alt']}; border: 1px solid {c['border']}; border-radius: 999px; padding: 0.3rem 0.75rem;
}}
.adf-badge-dot {{ width: 6px; height: 6px; border-radius: 50%; background: {c['accent']}; }}

/* ==========================================================================
   4. Cards / surfaces / elevation hierarchy.
   ========================================================================== */
.adf-card {{
    background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {RADIUS['lg']};
    padding: {SPACING['md']}; box-shadow: {SHADOW['card']};
    transition: box-shadow {TRANSITION}, transform {TRANSITION}, border-color {TRANSITION};
}}
.adf-card--interactive:hover {{
    transform: translateY(-2px); box-shadow: {SHADOW['card_hover']}; border-color: {c['border_strong']};
}}
.adf-card--compact {{ padding: {SPACING['sm']} {SPACING['md']}; }}
.adf-card--muted {{ background: {c['surface_alt']}; box-shadow: none; }}

/* Feature/capability cards */
.adf-feature-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: {SPACING['sm']}; }}
.adf-feature-card {{
    background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {RADIUS['md']};
    padding: {SPACING['md']}; text-decoration: none; display: block; color: inherit;
    transition: box-shadow {TRANSITION}, transform {TRANSITION}, border-color {TRANSITION};
}}
.adf-feature-card:hover {{
    transform: translateY(-2px); box-shadow: {SHADOW['card_hover']}; border-color: {c['accent']};
}}
.adf-feature-number {{ font-family: {MONO_STACK}; font-size: 0.75rem; color: {c['accent_strong']}; font-weight: 700; }}
.adf-feature-title {{ font-size: 0.98rem; font-weight: 700; color: {c['text']}; margin: 0.35rem 0 0.3rem 0; }}
.adf-feature-body {{ font-size: 0.85rem; color: {c['text_muted']}; line-height: 1.5; }}

/* ==========================================================================
   5. Result panel -- semantic state via accent bar + tinted background.
   ========================================================================== */
.adf-result {{
    border-radius: {RADIUS['lg']}; border: 1px solid var(--adf-state-border); background: var(--adf-state-bg);
    padding: {SPACING['lg']}; position: relative; overflow: hidden; animation: adf-fade-in 260ms ease;
    box-shadow: {SHADOW['card']};
}}
.adf-result::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--adf-state-fg); }}
.adf-result-eyebrow {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--adf-state-fg); margin-bottom: 0.5rem; }}
.adf-result-title {{ font-size: 1.65rem; font-weight: 750; color: {c['text']}; margin-bottom: 0.5rem; letter-spacing: -0.01em; }}
.adf-result-explain {{ color: {c['text_muted']}; font-size: 0.96rem; line-height: 1.55; max-width: 640px; }}

/* ==========================================================================
   6. Probability bars.
   ========================================================================== */
.adf-prob-row {{ display: flex; align-items: center; gap: {SPACING['sm']}; margin-bottom: {SPACING['xs']}; }}
.adf-prob-label {{ width: 92px; flex-shrink: 0; font-size: 0.85rem; color: {c['text_muted']}; font-weight: 600; }}
.adf-prob-track {{ flex-grow: 1; height: 9px; border-radius: 999px; background: {c['surface_alt']}; overflow: hidden; }}
.adf-prob-fill {{ height: 100%; border-radius: 999px; transition: width 400ms ease; }}
.adf-prob-value {{ width: 54px; flex-shrink: 0; text-align: right; font-family: {MONO_STACK}; font-size: 0.85rem; color: {c['text']}; }}

/* ==========================================================================
   7. Threshold visualization.
   ========================================================================== */
.adf-threshold-rail {{ position: relative; height: 6px; border-radius: 999px; background: {c['surface_alt']}; margin: {SPACING['sm']} 0 0.5rem 0; }}
.adf-threshold-fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: {c['warning']}; }}
.adf-threshold-marker {{ position: absolute; top: -4px; width: 2px; height: 14px; background: {c['text']}; }}

/* ==========================================================================
   8. Metrics row / metric card.
   ========================================================================== */
.adf-metrics-row {{ display: flex; flex-wrap: wrap; gap: {SPACING['lg']}; margin-top: {SPACING['sm']}; }}
.adf-metric {{ min-width: 110px; }}
.adf-metric-value {{ font-size: 1.2rem; font-weight: 750; color: {c['text']}; font-family: {MONO_STACK}; }}
.adf-metric-label {{ font-size: 0.76rem; color: {c['text_faint']}; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.15rem; }}

.adf-metric-card {{
    background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {RADIUS['md']};
    padding: {SPACING['sm']} {SPACING['md']}; text-align: left;
}}
.adf-metric-card .value {{ font-size: 1.5rem; font-weight: 750; color: {c['accent_strong']}; font-family: {MONO_STACK}; }}
.adf-metric-card .label {{ font-size: 0.78rem; color: {c['text_faint']}; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }}

/* ==========================================================================
   9. Step flow / pipeline diagram.
   ========================================================================== */
.adf-steps {{ display: flex; flex-wrap: wrap; gap: {SPACING['sm']}; align-items: stretch; }}
.adf-step {{
    flex: 1 1 190px; background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {RADIUS['md']};
    padding: {SPACING['sm']} {SPACING['md']}; position: relative;
}}
.adf-step-number {{ font-family: {MONO_STACK}; font-size: 0.78rem; color: {c['accent_strong']}; margin-bottom: 0.3rem; }}
.adf-step-title {{ font-weight: 700; font-size: 0.92rem; color: {c['text']}; margin-bottom: 0.2rem; }}
.adf-step-body {{ font-size: 0.82rem; color: {c['text_muted']}; line-height: 1.45; }}
.adf-step-tag {{
    display: inline-block; margin-top: 0.4rem; font-size: 0.66rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; padding: 0.12rem 0.5rem; border-radius: 999px;
}}
.adf-step-tag--native {{ background: {c['info_bg']}; color: {c['info']}; }}
.adf-step-tag--project {{ background: {c['success_bg']}; color: {c['success']}; }}
.adf-step-tag--presentation {{ background: {c['warning_bg']}; color: {c['warning']}; }}

/* ==========================================================================
   9b. Professional data tables (replaces default st.dataframe look).
   ========================================================================== */
.adf-table-wrap {{
    border: 1px solid {c['border']}; border-radius: {RADIUS['md']}; overflow-x: auto;
    background: {c['surface']}; box-shadow: {SHADOW['card']};
}}
table.adf-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; min-width: 420px; }}
table.adf-table thead th {{
    position: sticky; top: 0; background: {c['surface_alt']}; color: {c['text']}; font-weight: 650;
    text-align: left; padding: 0.6rem 0.85rem; border-bottom: 1px solid {c['border']}; white-space: nowrap;
}}
table.adf-table tbody td {{
    padding: 0.55rem 0.85rem; border-bottom: 1px solid {c['border']}; color: {c['text']};
    vertical-align: middle;
}}
table.adf-table tbody tr:last-child td {{ border-bottom: none; }}
table.adf-table tbody tr {{ transition: background-color {TRANSITION}; }}
table.adf-table tbody tr:hover {{ background-color: {c['accent_soft_bg']}; }}
table.adf-table td.adf-td-num, table.adf-table th.adf-th-num {{ text-align: right; font-family: {MONO_STACK}; font-variant-numeric: tabular-nums; }}
table.adf-table td.adf-td-label {{ text-align: left; }}
.adf-chip {{
    display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.72rem; font-weight: 700;
    padding: 0.18rem 0.55rem; border-radius: 999px; border: 1px solid var(--adf-chip-border);
    background: var(--adf-chip-bg); color: var(--adf-chip-fg); white-space: nowrap;
}}
.adf-kv-card {{
    display: none; background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {RADIUS['md']};
    padding: {SPACING['sm']} {SPACING['md']}; margin-bottom: {SPACING['xs']};
}}
.adf-kv-card .adf-kv-row {{ display: flex; justify-content: space-between; gap: {SPACING['sm']}; padding: 0.25rem 0; border-bottom: 1px dashed {c['border']}; font-size: 0.85rem; }}
.adf-kv-card .adf-kv-row:last-child {{ border-bottom: none; }}
.adf-kv-card .adf-kv-key {{ color: {c['text_muted']}; font-weight: 600; }}
.adf-kv-card .adf-kv-val {{ color: {c['text']}; text-align: right; font-family: {MONO_STACK}; }}
@media (max-width: 480px) {{
    table.adf-table.adf-table--stack-on-mobile {{ display: none; }}
    .adf-kv-card {{ display: block; }}
}}

/* ==========================================================================
   10. Status badges / info callouts / empty states.
   ========================================================================== */
.adf-status-badge {{
    display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.76rem; font-weight: 700;
    padding: 0.2rem 0.6rem; border-radius: 999px; border: 1px solid var(--adf-state-border);
    background: var(--adf-state-bg); color: var(--adf-state-fg);
}}
.adf-callout {{
    border-radius: {RADIUS['md']}; border: 1px solid {c['info_border']}; background: {c['info_bg']};
    padding: {SPACING['sm']} {SPACING['md']}; font-size: 0.88rem; color: {c['text']};
}}
.adf-empty-state {{
    text-align: center; padding: {SPACING['xl']} {SPACING['md']}; border: 1px dashed {c['border_strong']};
    border-radius: {RADIUS['lg']}; background: {c['surface']};
}}
.adf-empty-state .adf-section-title {{ margin-top: {SPACING['sm']}; }}

/* ==========================================================================
   11. Hero signal mark background accent.
   ========================================================================== */
.adf-hero-wrap {{ display: flex; align-items: center; gap: {SPACING['lg']}; flex-wrap: wrap; }}
.adf-hero-copy {{ flex: 1 1 420px; }}
.adf-hero-visual {{ flex: 0 0 220px; }}

/* ==========================================================================
   12. Footer.
   ========================================================================== */
.adf-footer {{
    margin-top: {SPACING['xl']}; padding-top: {SPACING['md']}; border-top: 1px solid {c['border']};
    color: {c['text_faint']}; font-size: 0.8rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: {SPACING['sm']};
}}

/* ==========================================================================
   13. Streamlit widget restyling -- stable public selectors only.
   ========================================================================== */
.stButton > button {{
    background: {c['gradient']}; color: #ffffff; border: none; border-radius: {RADIUS['sm']};
    font-weight: 650; padding: 0.6rem 1.5rem; box-shadow: 0 2px 8px -2px rgba(69,104,242,0.45);
    transition: box-shadow {TRANSITION}, transform {TRANSITION};
}}
.stButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 6px 16px -4px rgba(69,104,242,0.5); }}
.stButton > button:active {{ transform: translateY(0px); box-shadow: 0 2px 6px -2px rgba(69,104,242,0.4); }}
.stButton > button:focus-visible {{ outline: 2px solid {c['accent_strong']}; outline-offset: 2px; }}
.stButton > button[kind="secondary"] {{
    background: {c['surface']}; color: {c['accent_strong']}; border: 1px solid {c['border_strong']}; box-shadow: none;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: {c['surface']} !important; border: 1.5px dashed {c['border_strong']} !important; border-radius: {RADIUS['lg']} !important;
    transition: border-color {TRANSITION} !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: {c['accent']} !important; }}
[data-testid="stMetricValue"] {{ font-family: {MONO_STACK}; color: {c['text']}; }}
[data-testid="stExpander"] {{
    border: 1px solid {c['border']} !important; border-radius: {RADIUS['md']} !important; background: {c['surface']} !important;
    transition: border-color {TRANSITION} !important;
}}
[data-testid="stExpander"]:hover {{ border-color: {c['border_strong']} !important; }}
[data-testid="stAlert"] {{ border-radius: {RADIUS['md']}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {c['border']}; border-radius: {RADIUS['md']}; overflow: hidden; }}

/* ==========================================================================
   14. Motion -- subtle only, respects prefers-reduced-motion.
   ========================================================================== */
@keyframes adf-fade-in {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@media (prefers-reduced-motion: reduce) {{
    .adf-result {{ animation: none; }}
    .stButton > button, .adf-prob-fill, .adf-card--interactive, .adf-feature-card,
    [data-testid="stFileUploaderDropzone"], [data-testid="stExpander"], table.adf-table tbody tr {{ transition: none !important; }}
    .adf-card--interactive:hover, .adf-feature-card:hover, .stButton > button:hover {{ transform: none !important; }}
}}

/* ==========================================================================
   15. Responsive adjustments.
   ========================================================================== */
@media (max-width: 900px) {{
    .adf-feature-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
@media (max-width: 640px) {{
    .adf-headline {{ font-size: 1.7rem; }}
    .adf-result-title {{ font-size: 1.3rem; }}
    .adf-prob-label {{ width: 68px; font-size: 0.78rem; }}
    .adf-metrics-row {{ gap: {SPACING['md']}; }}
    .adf-step {{ flex: 1 1 100%; }}
    .adf-feature-grid {{ grid-template-columns: 1fr; }}
    .adf-hero-visual {{ flex-basis: 100%; order: -1; }}
}}
</style>
"""
