# UI/UX Design Phase

Branch: `feat/spectra-streamlit-candidate`. `main` is unmodified.
**This phase touches presentation only** — no model, artifact, HF
revision, SHA256, threshold, binary decision logic, inconclusive-state
logic, preprocessing, evaluation, windowing, or ONNX Runtime
configuration was changed. Every file touched lives in `streamlit_app.py`,
`app/styles.py` (new), `app/components.py` (new), or `app/visualizations.py`
(plot styling only).

## 1. Visual direction

A sophisticated dark interface: very dark navy/graphite background
(`#0b0f14`), a slightly elevated charcoal-navy surface for cards
(`#131922`), a controlled cool-blue accent (`#4f8ff7`) used for
interactive elements only, off-white text (`#e8ecf1`, never pure white),
and three restrained semantic accents (teal-green, coral, amber) reserved
for the three result states. No neon, no glow, no gradients, no emoji,
no cyberpunk/gaming aesthetic. Semantic state is communicated via a small
left accent bar plus a tinted (not flooded) background on the single
result panel — the page itself never turns red or green.

## 2. Design system (`app/styles.py`, `app/components.py`)

`app/styles.py` is the single source of truth for color/spacing/
typography/radius/shadow tokens (`COLORS`, `SPACING`, `RADIUS`, `SHADOW`,
`STATE_TOKENS`) and the one injected `<style>` block
(`inject_global_styles()`). No other file defines a raw hex color or
ad-hoc CSS block. `app/components.py` builds all reusable markup
(probability bars, the result panel, the metrics row, the step flow, the
inline SVG brand mark) as **pure, Streamlit-free functions** wherever
possible — `probability_bar_html`, `result_panel_html`,
`threshold_visualization_html`, `metrics_row_html`, `step_flow_html`,
`brand_mark_svg` — with a thin layer of `render_*` wrappers that actually
call `st.markdown`/`st.expander`. This split makes the markup-building
logic unit-testable without a Streamlit runtime (`tests/test_app_components.py`).

CSS targets custom `.adf-*` classes almost everywhere; the handful of
Streamlit-native selectors touched (`.stButton > button`,
`[data-testid="stFileUploaderDropzone"]`, `[data-testid="stMetricValue"]`,
`[data-testid="stExpander"]`, `[data-testid="stAlert"]`) are Streamlit's
own documented `data-testid` attributes, which are far more stable across
versions than un-prefixed auto-generated class names.

## 3. Typography

System font stack only (`-apple-system, BlinkMacSystemFont, 'Segoe UI',
Inter, Roboto, ...`), no external font file or Google Fonts dependency.
Hierarchy: an uppercase, letter-spaced "eyebrow" label; a restrained
~2.15rem headline (not a giant hero); section titles at ~1.05rem;
technical metadata in a monospace stack for numbers/hashes/revisions.

## 4. Page structure and progressive disclosure

```
Header (brand mark + eyebrow + "Research Prototype" badge)
  -> Hero (headline + subtext)                          [initial state only]
  -> Upload card (native st.file_uploader, restyled) + privacy microcopy
  -> Analysis workspace (audio player | metadata table)  [after upload]
  -> Analyze Audio button
  -> Result panel (BONAFIDE / SPOOF / INCONCLUSIVE)      [after analysis]
  -> Result metrics row
  -> Segment analysis (only rendered if windows_analyzed > 1)
  -> Audio characteristics (waveform | spectrogram, side by side)
  -> How the analysis works (always visible, educational)
  -> Why Spectra-AASIST3? (expander, always visible)
  -> Evaluation (expander, always visible)
  -> Technical details (expander, only after analysis)
  -> Disclaimer + footer (always visible, always last)
```

The hero headline is hidden once a file is uploaded (State 2 replaces the
empty-state emphasis with the audio workspace, per the brief); the result
panel, segment analysis, visual-analysis section, and technical details
only appear after a successful analysis — nothing data-dependent is shown
before there is data.

## 5. Result-state semantics — why "confidence" language was avoided

Carried forward unchanged from the previous prediction-semantics phases
(`docs/spectra_prediction_semantics_fix.md`,
`docs/spectra_inconclusive_state.md`): the calibrated decision threshold
(0.9397) sits far from a naive 50% softmax split, so the "confidence" of
the predicted class can legitimately be a minority probability. Calling
that value "confidence" reads as self-contradictory. This phase's UI
therefore always shows **both** class probabilities as a labeled
comparison ("Bonafide" / "Spoof", each as a percentage and a bar), never
a single "confidence" number standing in for the decision. For
INCONCLUSIVE specifically, an additional threshold-rail visualization
shows the observed spoof probability against the calibrated threshold
directly, so the result is intuitively understandable without implying
the score is a calibrated statistical probability beyond what it actually
is (a raw softmax value).

## 6. Component-by-section notes

- **Upload**: the native `st.file_uploader` is kept (no fragile custom
  drag-and-drop reimplementation, per the explicit "do not break upload
  interaction" instruction) but its dropzone is restyled via the stable
  `data-testid="stFileUploaderDropzone"` selector, and its label carries
  the requested "Drop an audio sample here" copy plus format/size limits.
- **Loading**: `st.status(...)` with real, sequential stage labels
  ("Preparing detector...", "Processing audio...", "Running anti-spoof
  analysis...", "Combining segment results...", "Preparing visual
  analysis...", "Analysis complete"). No fabricated percentage or fake
  progress bar — each label transition corresponds to a real point in the
  actual `get_detector()` / `detector.predict()` call sequence.
- **Result panel**: one `adf-result` card per analysis, with state-specific
  copy exactly as specified (BONAFIDE/SPOOF/INCONCLUSIVE headline +
  explanation), plus the probability comparison and (INCONCLUSIVE only)
  the threshold rail.
- **Metrics row**: audio duration, segments analyzed, analysis time,
  runtime ("CPU / ONNX") — four values, no more.
- **Segment analysis**: only rendered when `windows_analyzed > 1`; shows a
  one-line summary (N segments, spoof-leaning vs. bonafide-leaning count)
  before the detailed table, which is behind an expander rather than
  dumped immediately.
- **Waveform / spectrogram**: `app/visualizations.py` now renders both
  plots with a dark facecolor/grid/text matching the surrounding card
  (`PLOT_BG`/`PLOT_GRID`/`PLOT_TEXT` tokens), smaller default figure size
  (fits a two-column layout without oversized whitespace), and explicit
  axis labels. The mel-spectrogram's scientific parameters (n_fft=400,
  win_length=400, hop_length=160, n_mels=80) and colormap (`magma`, a
  perceptually-uniform, professional default) are unchanged — only
  presentation (colors, sizing, tick styling) was touched.
- **How it works**: a 4-step horizontal flow (Upload → Preprocess →
  Analyze → Aggregate) built from `step_flow_html`, with an explicit
  caption stating multi-segment aggregation is a project-level
  application extension, not the model authors' own methodology.
- **Why Spectra-AASIST3?**: replaces the outdated "Why Wav2Vec2?" expander
  entirely with accurate copy (XLS-R-300M + AASIST-style back-end,
  comparative generalization testing, dynamically quantized INT8 ONNX,
  CPU-only) and the required "pre-release/unpublished... evaluated
  independently" caveat — no claim of peer-reviewed or published status.
- **Technical details**: unchanged in spirit from the previous phase — it
  remains entirely `model_info()`-driven (`docs/spectra_inconclusive_state.md`),
  now additionally showing segments analyzed and rendered inside the new
  card styling.
- **Evaluation**: a small, collapsed-by-default expander leading with "the
  fixed held-out balanced In-the-Wild subset" framing, never "99%
  accurate" language — shows EER/ROC-AUC/F1/bonafide-FPR as plain metric
  tiles, explicitly project-measured, never presented as the model
  authors' own benchmark.
- **Disclaimer / footer**: kept concise and visually secondary (smaller,
  muted text, below a divider), always the last thing on the page.

## 7. Accessibility

- Text/background contrast: body text `#e8ecf1` on `#0b0f14`/`#131922`
  backgrounds exceeds WCAG AA for normal text; muted text `#8b96a8` was
  chosen to still clear AA for the smaller caption sizes used.
- State is never color-only: each result state also carries a distinct
  eyebrow position, headline copy, and (for INCONCLUSIVE) a visibly
  different rail visualization — a color-blind user reading the text
  still gets the full result.
- All buttons keep their native accessible label ("Analyze Audio"); no
  icon-only controls were introduced.
- `st.file_uploader`, `st.button`, and all `st.expander` calls keep
  descriptive, non-generic labels ("Technical details", "Evaluation",
  "Why Spectra-AASIST3?", "View segment-level detail").
- Keyboard navigation is unaffected — no custom click-handling or
  JavaScript was introduced; all interaction remains native Streamlit
  widgets.
- `.stButton > button:focus-visible` gets an explicit visible outline
  (Streamlit's own default focus ring is preserved/reinforced, not
  removed).
- `@media (prefers-reduced-motion: reduce)` disables the result-panel
  fade-in and button/bar transitions.
- Error messages remain short, plain-English strings (see Section 9) —
  no tracebacks are ever shown to the user.

## 8. Responsive behavior

The `.block-container` is capped at `1040px` max width so the layout
doesn't stretch awkwardly on large desktop displays, while still using
Streamlit's native `st.columns` (which stack automatically below
Streamlit's own responsive breakpoint) for the two-column workspace,
metadata, and waveform/spectrogram sections — no custom JS media-query
logic was needed for column stacking, since this is Streamlit's built-in
behavior. Custom CSS media queries (`@media (max-width: 640px)`) reduce
the headline/result-title font sizes, narrow the probability-bar labels,
and force the step-flow cards to full width on narrow viewports, so
nothing overflows or requires horizontal scrolling.

**Honest limitation**: no automated browser (Playwright/Chrome) was
available in this session, so pixel-level screenshots at 1440/1280/1024/
768/390px were **not captured** — this is stated plainly rather than
fabricated. What was verified instead: (1) the actual `streamlit run`
server starts cleanly and serves the page (HTTP 200, confirmed via
`curl`); (2) the Streamlit `AppTest` suite exercises the full render tree
without exception; (3) the CSS media queries and Streamlit's native
column-stacking were reasoned through manually against each target
breakpoint. If Playwright or a connected browser tool becomes available,
this should be the first thing added as a follow-up visual-QA pass.

## 9. Error states

All error copy stays in `streamlit_app.py`'s `_render_error()` (a thin
`st.error(..., icon=...)` wrapper) and the pre-existing
`app/validation.py`/`app/errors.py` (unchanged — validation logic is not
part of "inference" but was left alone regardless, out of an abundance of
caution). Covered: unsupported format, over-duration, over-size, corrupt/
empty audio (all pre-existing, friendly `UserFacingError` messages), plus
two new explicit failure paths this phase added by splitting the single
try/except into two:

- **Model preparation failure** (`get_detector()` raises): *"Detector
  unavailable. The analysis model could not be prepared. Please try
  again shortly."* — the exception is logged, never shown.
- **Inference failure** (`detector.predict()` raises): the existing
  generic *"Something went wrong while analyzing this audio..."* message.

No Python traceback is ever rendered to the user in either case.

## 10. Performance

Pre-model RSS (importing `streamlit_app.py` in bare mode, before any
detector is loaded) was measured before and after this phase:

| | RSS (MB) |
|---|---|
| Before redesign | 95.9 |
| After redesign | 94.5 |

**No material increase** — within normal run-to-run variance. No new
heavyweight dependency (no React/Next.js/Node/Three.js/GSAP/Lottie/large
frontend framework) was added; the entire design system is CSS + inline
SVG + Python string building.

## 11. What was explicitly NOT touched

Model artifact, Hugging Face repository/revision/SHA256, calibrated
threshold, `decide_label()`/`presentation_state()` decision logic,
preprocessing (`author_compatible_preprocess`, pre-emphasis, windowing),
evaluation/calibration scripts and their frozen results, ONNX Runtime
session configuration, and `configs/models.yaml`. Verified via `git diff
--stat` before commit and via a full frozen-evaluation replay producing
byte-identical metrics (EER 3.0%, ROC-AUC 0.9819, F1 0.9375, bonafide FPR
2.0%, spoof FNR 10.0%).
