# Analysis Platform v2 — UI/UX and Platform Expansion

Branch: `feat/spectra-streamlit-candidate`. `main` is unmodified.

This phase supersedes the visual direction and page-routing approach
described in `docs/analysis_platform.md` (v1, dark theme, file-based
`pages/` multipage). It does **not** supersede v1's scientific content —
the frozen-detector guarantees, segment-evidence methodology, and
evaluation data sources documented there still apply. Read this file
alongside `docs/analysis_platform.md` for full context.

**The detector is frozen.** No change to the Spectra-AASIST3 model, INT8
artifact, Hugging Face revision, SHA256, ONNX Runtime inference,
preprocessing, pre-emphasis, sample rate, model input length, calibrated
threshold (`0.939693808555603`), binary decision logic, inconclusive
presentation rule, aggregation implementation, or any scientific
evaluation result. Confirmed for this phase by: zero diff under `src/`
and `configs/` (`git diff --stat -- src/ configs/` is empty), and a full
integration-test re-run (`tests/test_candidate_e.py`,
`tests/test_robustness.py`) against the real cached INT8 artifact.

## 1. Why this phase happened

The v1 dark theme, once deployed and visually reviewed, had specific,
named problems: a flat near-black theme with insufficient depth, a
default (bright) Streamlit sidebar clashing with the dark app, a
generic "streamlit app" sidebar label, the uploader appearing before any
hero/headline hierarchy, low-contrast muted text, and the overall
impression of a single-purpose upload tool rather than a platform. This
phase replaces the visual system and information architecture end to
end; it does not patch the old one.

## 2. Visual system

`app/styles.py` is the single source of truth for every color, spacing,
radius, shadow, and transition token (`COLORS`, `SPACING`, `RADIUS`,
`SHADOW`, `TRANSITION`, `STATE_TOKENS`). Light, cool, technical palette:
off-white page background (`#F5F7FB`), white elevated cards, deep navy
text, a cobalt/indigo accent (`#4568F2`/`#3B5BDB`) reserved for
interactive elements and gradients (hero, primary buttons, selected nav,
selected chart accents — never whole cards). Semantic state colors
(emerald/teal = Bonafide, red/coral = Spoof, amber = Inconclusive,
indigo = informational) drive the result panel and status badges via
CSS custom properties, so adding a new state only means adding one
`STATE_TOKENS` entry.

Depth comes from a small shadow/border/elevation scale
(`SHADOW["card"]`/`SHADOW["card_hover"]`), not from color alone, and
content is organized into four hierarchy tiers (primary result →
secondary analysis → technical details → reference/methodology) via
heading weight, card elevation, and section ordering.

Micro-interactions are restrained and CSS-only: card hover
(`translateY(-2px)` + shadow, ~180ms), button lift/press, probability
bar width transitions, expander/result-panel fade-slide entrance. No
looping, pulsing, spinning, or fake-progress animation exists anywhere
in the stylesheet. A `@media (prefers-reduced-motion: reduce)` block
disables every transition/transform/animation project-wide.

## 3. Navigation

The default Streamlit sidebar is fully hidden
(`[data-testid="stSidebar"] { display: none }` and the collapse control
alongside it) and replaced by `st.navigation(ALL_PAGES, position="top")`
— Streamlit 1.62's officially supported top-navigation mode, confirmed
via `inspect.signature(st.navigation)` before adoption rather than any
CSS/DOM hack. `app/nav.py` defines every destination once, as `st.Page`
objects wrapping plain callables (`app/views/*.py::render`), and is
imported by both the router (`streamlit_app.py`) and any view that needs
a cross-page `st.page_link` (e.g. Analyze → Robustness).

Navigation destinations: **Analyze** (default), **Batch**,
**Robustness**, **Evaluation**, **Methodology**, **About**. Every
destination renders real, working functionality — none is a stub page
that exists only to fill out the nav bar.

## 4. New/expanded functionality this phase added

- **Batch Analysis** (`app/views/batch.py`, `MAX_BATCH_FILES = 5`) —
  strictly sequential processing of multiple uploaded files against the
  single cached detector (never concurrent, to protect Streamlit Cloud
  memory), a live queue/status table, a results table, CSV export, and
  an HTML batch report. The file-count cap is a documented, conservative
  choice based on measured per-clip INT8 latency
  (`results/metrics/spectra_resource_fp32_vs_int8.json`), not a
  guess.
- **Analysis Session** (`app/analysis/session.py`) — a temporary,
  session-scoped (not persistent, not a database) record of each
  analysis performed in the current browser session: filename, SHA-256,
  metadata, result, probabilities, timestamp, model version, segment
  summary. Never stores raw audio. Bounded to
  `MAX_SESSION_ENTRIES = 25` to cap memory growth. Viewable, per-entry
  removable, clearable, and exportable as JSON from the Analyze page.
- **Audio Metadata Inspector** (`app/analysis/audio_quality.py::probe_media_metadata`)
  — real container/codec/bitrate/duration extraction via `ffprobe`
  (already present via this project's ffmpeg installation), with an
  explicit `"Not available"` fallback for any field ffprobe can't
  determine — nothing is guessed.
- **Segment Evidence Navigator** — extends the v1 segment-evidence
  timeline with a segment selector that renders the local
  waveform/spectrogram excerpt and probabilities for one chosen segment.
  Explicitly labeled "segment evidence," never "detected at," since this
  is window classification, not manipulation localization.
- **Analysis Consistency stats** — `SegmentAgreement` gained
  `score_mean`/`score_median` (alongside the existing
  `agreement_percent`/`score_std`), surfaced as a 5-metric row above the
  segment timeline. Descriptive only; never modifies the final
  classification.
- **Robustness Lab** (`app/views/robustness.py`) — reuses the existing,
  unmodified degradation methodology
  (`app/analysis/robustness.py`, itself a reuse of
  `scripts/run_spectra_evaluation.py`'s exact transforms) against
  whichever clip is currently loaded (from Analyze, or uploaded
  directly). `run_robustness_analysis()` gained an `on_condition_start`
  callback (real "Testing MP3 128 kbps..." status text, never a
  fabricated percentage) and a `processing_time_ms` field per condition.
  Results table: Condition / Result / Spoof probability / Difference
  from original / Processing time, plus a bar-chart comparison
  (`plot_robustness_comparison`) and a descriptive stability summary.
- **Threshold Explorer** (`app/analysis/threshold_explorer.py`,
  Evaluation page) — a research-only, descriptive threshold sweep over
  real per-clip INT8 spoof scores
  (`app/data/int8_evaluation_scores.json`), recomputed via a read-only
  re-run of the frozen, unmodified `predict()` path over the same
  200-clip evaluation set already used for the accepted frozen metrics.
  Cross-validated: sweeping at the production threshold
  (`0.939693808555603`) reproduces the exact accepted metrics (EER 3.0%,
  ROC-AUC 0.9819, F1 0.9375, FPR 2.0%, FNR 10.0%). The tool never writes
  back to production and states so explicitly in its own UI copy. If
  this data file is ever absent from a deployment, the section degrades
  to an explanatory message rather than fabricating numbers.
- **Reporting** — extended to batch (CSV + HTML batch report) alongside
  the existing single-analysis HTML/JSON report; the Analysis Session
  panel adds a JSON session export. No report ever claims "verified
  authentic," "confirmed deepfake," or "forensically validated."

## 5. Component system

`app/components.py` centralizes reusable markup builders
(`metric_card_html`, `feature_card_html`/`feature_grid_html`,
`status_badge_html`, `info_callout_html`, `empty_state_html`,
`step_flow_html` with native/project/presentation tags, plus the
pre-existing `result_panel_html`/`probability_bar_html`/
`metrics_row_html`) so pages assemble markup from named functions
instead of scattering raw HTML. Pure markup-building functions (no
`streamlit` import) are kept separate from thin `render_*` wrappers that
call `st.*`, matching this project's established testability pattern.

## 6. Empty, loading, and error states

Every non-Analyze destination has a designed empty state
(`render_empty_state`) rather than a bare "please upload a file" —
Batch and Robustness both explain what will happen and what's supported
before any file exists. Loading states use `st.status`/`st.progress`
with real, derived stage text (segment count computed from actual
duration, per-condition robustness labels) — no fabricated percentages
anywhere. Errors go through the existing `UserFacingError` /
`app/errors.py` path, never a raw traceback.

## 7. Known limitations of this phase

- No browser-automation tool was available in this environment (as in
  prior phases). Visual QA was performed by launching the app locally,
  confirming it serves HTTP 200 with a clean startup log, and manually
  reasoning through the CSS/markup for the 5 target breakpoints. A
  human visual pass in an actual browser is recommended before treating
  this as fully QA'd.
- The Threshold Explorer's underlying `app/data/int8_evaluation_scores.json`
  was generated by a one-off local recomputation script (not itself
  committed as a permanent script under `scripts/`) run against this
  developer's local model cache; regenerating it on another machine
  requires the same cached INT8 ONNX artifact.
