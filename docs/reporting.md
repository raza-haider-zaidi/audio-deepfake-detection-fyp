# Reporting — Tables and the Premium PDF Report

Branch: `feat/spectra-streamlit-candidate`. `main` is unmodified.

## 1. Professional tables

Every tabular result in the app (segment evidence, batch analysis queue
and results, robustness results, evaluation/FP32-vs-INT8/resource/Sara
comparison tables, session history) is rendered through one shared
component: `app.components.render_professional_table` /
`professional_table_html`, backed by CSS in `app/styles.py`
(`.adf-table*`) — white surface, light blue/gray sticky header, cool-gray
row separators, a subtle cobalt row-hover highlight, right-aligned
tabular-number columns for percentages/durations/counts, and semantic
status chips (`status_chip_html`) for result/condition columns
(BONAFIDE/SPOOF/INCONCLUSIVE/GOOD/LIMITED/POOR/etc.) — chip text is
always shown, never color alone. No page hand-rolls its own table markup.

Mobile: tables scroll horizontally within their own `.adf-table-wrap`
container (the page itself never overflows horizontally). Wide tables
that would be unreadable at very small widths (batch results, session
history) additionally opt into `stack_on_mobile=True`, which renders a
parallel stacked key/value card per row, shown only under a 480px
breakpoint via CSS (the wide table is hidden at that width instead).

**Not implemented in this phase:** interactive click-to-sort column
headers and client-side pagination. Per the "no heavy JS data-grid"
guidance, tables are static, server-rendered HTML; this keeps the
implementation small and dependency-free but is a real limitation for
very large tables (mitigated today only by the existing batch file cap
and segment-count limits). See the phase's final report for this
disclosed as an incomplete item.

## 2. PDF report

### Library

`reportlab` (pure-Python `platypus` layout engine + `pdfgen` canvas for
header/footer), chosen after inspecting installed packages and finding no
existing PDF library. Selected over WeasyPrint/xhtml2pdf because it has
no system library dependency (WeasyPrint needs Pango/Cairo, which are not
guaranteed on Streamlit Community Cloud), and over a browser/Chrome-based
renderer because that was explicitly disallowed. Package footprint is
small (`reportlab` + `chardet`), added to `requirements.txt` pinned at
`4.2.5`. Generation happens entirely in memory
(`app/reporting/pdf_report.py::build_pdf_report(report_data) -> bytes`)
and is handed directly to `st.download_button`; nothing is written to
disk, and the report is not retained after the request.

### Branding

The report header draws the SAME five-bar abstract signal mark used as
the web app's brand SVG (`app/components.py::brand_mark_svg`), redrawn as
vector `roundRect` shapes via `reportlab.pdfgen.canvas` (`_brand_mark` in
`pdf_report.py`) — no rasterized image, no external asset, no third-party
logo. Colors are the exact same hex values as `app/styles.py::COLORS`
(navy `#1B2440`, cobalt `#4568F2`, muted `#4A5578`, semantic
success/danger/warning), so the PDF and the web UI read as one product.

### Visual style

White background, dark navy typography, cobalt accents used sparingly
(result banner border/eyebrow, headings), subtle gray/blue dividers, a
compact technical table style shared by every section, a page
header/footer with the report ID, generation timestamp, and page number
on every page, and a secondary (small, italic) disclaimer line in the
footer of every page. No gradients, no decorative imagery, no emoji.

### Structure (typical report, 4 pages)

1. **Executive analysis** — brand header, report ID/generated
   timestamp/filename, a prominent result panel (Likely Real / Bonafide,
   Likely AI-Generated / Spoofed, or Inconclusive / Mixed Evidence) with
   a state-tinted background, bonafide/spoof class probabilities (never
   called "confidence"), the calibrated threshold, a one-sentence
   interpretation, and File Information (filename, SHA-256, duration,
   sample rate, format, channels, plus source-specific fields).
2. **Evidence** — segment evidence summary (segments analyzed, dominant
   direction, agreement, mean/median spoof probability, probability
   range), an optional "Spoof Probability Over Time" chart with the
   calibrated threshold line (only rendered when real segment data
   exists — never fabricated for a single-window clip), and the
   segment-level table (capped at 60 rows for report length, with a
   note if truncated).
3. **Audio conditions** — peak amplitude, RMS, silence proportion,
   clipping, analysis-suitability level, approximate bitrate where
   available, and an explicit note that these diagnostics are
   supplementary and never alter the frozen classification.
4. **Model & reproducibility** — model display name, architecture,
   runtime, model artifact/repository, revision, threshold description,
   analysis sample rate, and inference time — all read live from
   `detector.model_info()` / the loaded model config, never hardcoded or
   copied from a prior (e.g. Sara) model — plus the full disclaimer text.

### Source-specific metadata

`build_report_data(..., source_type=..., source_metadata=...)` carries
the same `source_type` used everywhere else in the app
(`audio_file`/`microphone`/`voice_note`/`video_audio`) and a source
adapter's `source_metadata` dict straight into both the PDF and the HTML
report's File Information section:

- **Audio file:** format only (standard, already-known metadata).
- **Microphone:** capture type ("Live microphone recording"), requested
  sample rate.
- **Voice note:** original format/container, "Normalized for analysis:
  Mono · 16 kHz".
- **Video:** original video filename, container, video codec, audio
  codec, full video duration, and the selected analysis interval (e.g.
  "5s–35s") — the report always states the *analyzed interval*, never
  implies the whole video was analyzed.

The report never claims "Video detected as deepfake" — result language
is always phrased around "the analyzed audio track" / "the analyzed
speech", regardless of source.

### Report ID and privacy

The report ID (`AD-YYYYMMDD-XXXX`, derived from the input SHA-256 plus
date — unchanged from the pre-existing HTML/JSON report logic in
`app/reporting/report.py`) is a local, session-scoped reproducibility
identifier, not a database reference. The original audio/video is never
embedded in the PDF. No uploaded file, recording, extracted audio, or
generated report is written to persistent storage by this application.

### Disclaimer

Every report — PDF, HTML, and JSON — carries the same
`DISCLAIMER_TEXT` ("Research prototype. This detector should not be used
as the sole basis for forensic, legal, security, disciplinary, or
identity decisions...") from `app/reporting/report.py`, so all three
formats stay consistent by construction.
