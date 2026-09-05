"""Premium downloadable PDF analysis report.

Built with ReportLab (pure-Python, no LibreOffice/Chrome/Puppeteer, no
system dependency beyond the already-installed Pillow used for the
optional embedded chart image) directly from the SAME `report_data` dict
already assembled by `app.reporting.report.build_report_data` -- no
number here is computed independently of the values already produced
elsewhere in the app. Generated entirely in memory; nothing is written to
disk and nothing is retained after the request completes. The original
audio is never embedded.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import ParagraphStyle

from app.branding import PROJECT_AUTHOR
from app.reporting.report import DISCLAIMER_TEXT, SOURCE_DISPLAY_LABELS

# Brand palette -- mirrors app/styles.py COLORS exactly so the PDF and the
# web UI read as the same product.
NAVY = colors.HexColor("#1B2440")
MUTED = colors.HexColor("#4A5578")
FAINT = colors.HexColor("#7680A0")
ACCENT = colors.HexColor("#4568F2")
ACCENT_STRONG = colors.HexColor("#3B5BDB")
BORDER = colors.HexColor("#DEE3EE")
SURFACE_ALT = colors.HexColor("#EEF1F8")
WHITE = colors.white
SUCCESS = colors.HexColor("#0E9F6E")
SUCCESS_BG = colors.HexColor("#E6F6F0")
DANGER = colors.HexColor("#E23E57")
DANGER_BG = colors.HexColor("#FBE9EC")
WARNING = colors.HexColor("#C88A1C")
WARNING_BG = colors.HexColor("#FBF1DE")

STATE_COLORS = {
    "BONAFIDE": (SUCCESS, SUCCESS_BG),
    "SPOOF": (DANGER, DANGER_BG),
    "INCONCLUSIVE": (WARNING, WARNING_BG),
}

PAGE_SIZE = A4
MARGIN = 20 * mm

_body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.2, leading=13.5, textColor=NAVY)
_muted = ParagraphStyle("muted", fontName="Helvetica", fontSize=8.2, leading=12, textColor=MUTED)
_h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceAfter=2)
_h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=NAVY, spaceBefore=14, spaceAfter=6)
_eyebrow = ParagraphStyle("eyebrow", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=ACCENT_STRONG, spaceAfter=4)
_result_title = ParagraphStyle("result_title", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=NAVY, spaceAfter=4)
_interp = ParagraphStyle("interp", fontName="Helvetica-Oblique", fontSize=9, leading=13, textColor=MUTED, spaceBefore=4)


def _brand_mark(canvas: pdf_canvas.Canvas, x: float, y: float, size: float = 9) -> None:
    """Draws the same five-bar abstract signal mark used as the brand SVG
    in the web UI (app/components.py::brand_mark_svg), as vector shapes --
    no external image, no third-party logo."""
    heights = [0.3, 0.65, 1.0, 0.65, 0.4]
    bar_w = size * 0.16
    gap = size * 0.09
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    cx = x
    for h_frac in heights:
        h = size * h_frac
        canvas.roundRect(cx, y - h / 2, bar_w, h, bar_w / 2.2, stroke=0, fill=1)
        cx += bar_w + gap
    canvas.restoreState()


def _header_footer(canvas: pdf_canvas.Canvas, doc, *, report_id: str, generated_at: str) -> None:
    canvas.saveState()
    width, height = PAGE_SIZE

    # Header
    _brand_mark(canvas, MARGIN, height - 16 * mm, size=8)
    canvas.setFont("Helvetica-Bold", 10.5)
    canvas.setFillColor(NAVY)
    canvas.drawString(MARGIN + 12 * mm, height - 14.5 * mm, "AUDIO DEEPFAKE ANALYSIS")
    canvas.setFont("Helvetica", 7.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN + 12 * mm, height - 18.2 * mm, "Audio Analysis Report · Research Prototype")
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, height - 21 * mm, width - MARGIN, height - 21 * mm)

    # Footer
    canvas.setStrokeColor(BORDER)
    canvas.line(MARGIN, 16 * mm, width - MARGIN, 16 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(FAINT)
    canvas.drawString(MARGIN, 12.5 * mm, f"Report ID {report_id} · Generated {generated_at}")
    canvas.drawRightString(width - MARGIN, 12.5 * mm, f"Page {doc.page}")
    canvas.setFont("Helvetica-Oblique", 6.6)
    canvas.drawString(MARGIN, 9.2 * mm, "Research prototype — not for forensic, legal, security, disciplinary, or identity decisions.")
    canvas.restoreState()


def _kv_table(rows: list[tuple[str, str]], col_widths: tuple[float, float] = (55 * mm, None)) -> Table:
    width2 = col_widths[1] or (PAGE_SIZE[0] - 2 * MARGIN - col_widths[0])
    data = [[Paragraph(f"<b>{label}</b>", _muted), Paragraph(str(value), _body)] for label, value in rows]
    t = Table(data, colWidths=[col_widths[0], width2])
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, BORDER),
            ]
        )
    )
    return t


def _data_table(header: list[str], rows: list[list[str]]) -> Table:
    data = [header] + rows
    t = Table(data, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SURFACE_ALT),
                ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("TEXTCOLOR", (0, 1), (-1, -1), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ]
        )
    )
    return t


def _result_interpretation(presentation_state: str) -> str:
    if presentation_state == "SPOOF":
        return "The analyzed speech exceeded the detector's calibrated spoof operating threshold."
    if presentation_state == "INCONCLUSIVE":
        return "The detector identified elevated spoof indicators, but the result remained below the calibrated spoof threshold."
    return "The analyzed speech was more consistent with genuine human speech under the model's calibrated operating threshold."


def _segment_chart_png(segment_rows: list[dict], threshold: float | None) -> bytes | None:
    if not segment_rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure

    fig = matplotlib.figure.Figure(figsize=(6.2, 2.1), facecolor="#FFFFFF")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#FFFFFF")
    x = [r["start_seconds"] for r in segment_rows]
    y = [r["spoof_probability"] * 100 for r in segment_rows]
    ax.plot(x, y, color="#4568F2", marker="o", markersize=3, linewidth=1.4)
    if threshold is not None:
        ax.axhline(threshold * 100, color="#C88A1C", linestyle="--", linewidth=1, label="Calibrated threshold")
        ax.legend(loc="upper right", fontsize=6.5, frameon=False)
    ax.set_xlabel("Time (s)", fontsize=7.5, color="#4A5578")
    ax.set_ylabel("Spoof probability (%)", fontsize=7.5, color="#4A5578")
    ax.tick_params(labelsize=6.5, colors="#4A5578")
    for spine in ax.spines.values():
        spine.set_color("#DEE3EE")
    ax.set_ylim(0, 100)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    return buf.getvalue()


def build_pdf_report(report_data: dict[str, Any]) -> bytes:
    """Renders the full multi-page PDF report and returns raw bytes,
    suitable for `st.download_button(..., data=...)`. Never touches disk."""
    ai = report_data["analysis_information"]
    result = report_data["result"]
    seg = report_data["segment_analysis"]
    conditions = report_data["audio_conditions"]
    model = report_data["model"]
    perf = report_data["performance"]
    presentation_state = result["presentation_state"]
    fg, bg = STATE_COLORS.get(presentation_state, STATE_COLORS["INCONCLUSIVE"])

    buffer = io.BytesIO()
    doc = BaseDocTemplate(
        buffer,
        pagesize=PAGE_SIZE,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        title="Audio Deepfake Analysis Report",
        author=PROJECT_AUTHOR,
        subject="Audio Deepfake Detection Research Prototype",
    )
    frame = Frame(MARGIN, 20 * mm, PAGE_SIZE[0] - 2 * MARGIN, PAGE_SIZE[1] - 45 * mm, id="body")

    def _on_page(canvas, doc):
        _header_footer(canvas, doc, report_id=report_data["report_id"], generated_at=report_data["generated_at"])

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_on_page)])

    story: list = []

    # -------------------- Page 1: Executive analysis --------------------
    story.append(Paragraph("AUDIO DEEPFAKE ANALYSIS REPORT", _h1))
    story.append(
        Paragraph(
            f"Report ID {report_data['report_id']} &nbsp;·&nbsp; Generated {report_data['generated_at']} &nbsp;·&nbsp; "
            f"Filename {ai['filename']}",
            _muted,
        )
    )
    story.append(Spacer(1, 10))

    result_table = Table(
        [[Paragraph("ANALYSIS RESULT", _eyebrow)], [Paragraph(result["presentation_result"], _result_title)],
         [Paragraph(_result_interpretation(presentation_state), _interp)]],
        colWidths=[PAGE_SIZE[0] - 2 * MARGIN],
    )
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.7, fg),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(result_table)
    story.append(Spacer(1, 8))
    story.append(
        _data_table(
            ["Metric", "Value"],
            [
                ["Bonafide class probability", f"{result['bonafide_class_probability'] * 100:.1f}%"],
                ["Spoof class probability", f"{result['spoof_class_probability'] * 100:.1f}%"],
                ["Calibrated spoof threshold", f"{result['calibrated_threshold'] * 100:.2f}%" if result.get("calibrated_threshold") is not None else "Not available"],
                ["Binary model decision", str(result["binary_model_decision"])],
            ],
        )
    )
    story.append(Paragraph("These values are class probabilities, not a \"confidence\" score.", _muted))

    story.append(Paragraph("File Information", _h2))
    file_rows = [
        ("Input source", SOURCE_DISPLAY_LABELS.get(ai.get("source_type"), ai.get("source_type", "Audio File"))),
        ("Filename", ai["filename"]),
        ("SHA-256", ai["sha256"]),
        ("Duration", f"{ai['duration_seconds']:.2f} s"),
        ("Analysis sample rate", f"{ai['sample_rate_hz']} Hz"),
        ("Format", ai["format"]),
        ("Channels", str(ai["channels"])),
    ]
    for k, v in (ai.get("source_metadata") or {}).items():
        if k == "format":
            continue  # already shown above as the top-level "Format" row
        file_rows.append((k.replace("_", " ").title(), str(v)))
    story.append(_kv_table(file_rows))

    story.append(NextPageTemplate("main"))
    story.append(PageBreak())

    # -------------------- Page 2: Evidence --------------------
    story.append(Paragraph("Evidence", _h1))
    n_segments = seg["n_segments"]
    story.append(Paragraph(f"{n_segments} segment(s) analyzed as a project-level, non-overlapping extension.", _muted))

    agreement = seg.get("segment_agreement")
    if agreement:
        story.append(
            _kv_table(
                [
                    ("Segments analyzed", str(agreement["n_segments"])),
                    ("Dominant direction", str(agreement["dominant_direction"]).title()),
                    ("Agreement", f"{agreement['agreement_percent']:.0f}%"),
                    ("Mean spoof probability", f"{agreement['score_mean'] * 100:.1f}%"),
                    ("Median spoof probability", f"{agreement['score_median'] * 100:.1f}%"),
                    ("Probability range", f"{agreement['score_range'] * 100:.1f} pp"),
                ]
            )
        )
        story.append(Spacer(1, 6))

    segment_rows = seg.get("segments") or []
    chart_png = _segment_chart_png(segment_rows, result.get("calibrated_threshold"))
    if chart_png:
        from reportlab.platypus import Image

        story.append(Paragraph("Spoof Probability Over Time", _h2))
        story.append(Image(io.BytesIO(chart_png), width=PAGE_SIZE[0] - 2 * MARGIN, height=(PAGE_SIZE[0] - 2 * MARGIN) * 2.1 / 6.2))

    if segment_rows:
        story.append(Paragraph("Segment Evidence", _h2))
        table_rows = [
            [
                str(r["segment"]),
                f"{r['start_seconds']:.2f}s–{r['end_seconds']:.2f}s",
                f"{r['bonafide_probability'] * 100:.1f}%",
                f"{r['spoof_probability'] * 100:.1f}%",
                str(r["interpretation"]),
            ]
            for r in segment_rows[:60]
        ]
        story.append(_data_table(["Segment", "Time range", "Bonafide", "Spoof", "Interpretation"], table_rows))
        if len(segment_rows) > 60:
            story.append(Paragraph(f"...and {len(segment_rows) - 60} further segments (truncated for report length).", _muted))
    else:
        story.append(Paragraph("Only a single window was analyzed; no per-segment breakdown is available for this clip.", _muted))

    story.append(PageBreak())

    # -------------------- Page 3: Audio conditions --------------------
    story.append(Paragraph("Audio Conditions", _h1))
    quality = conditions.get("quality_metrics") or {}
    rows = [
        ("Peak amplitude", f"{quality.get('peak_amplitude', 0):.3f}" if quality else "Not available"),
        ("RMS level", f"{quality.get('rms_level', 0):.4f}" if quality else "Not available"),
        ("Silence proportion", f"{quality.get('silence_ratio', 0) * 100:.0f}%" if quality else "Not available"),
        (
            "Clipping",
            f"{quality.get('clipping_ratio', 0) * 100:.2f}%" if quality else "Not available",
        ),
        ("Analysis conditions", str(conditions.get("analysis_suitability") or "Not available")),
    ]
    if quality.get("approximate_bitrate_kbps") is not None:
        rows.append(("Approximate bitrate", f"{quality['approximate_bitrate_kbps']:.0f} kbps"))
    story.append(_kv_table(rows))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Audio-condition diagnostics are supplementary and do not alter the frozen detector classification.",
            _muted,
        )
    )
    story.append(PageBreak())

    # -------------------- Page 4: Model & reproducibility --------------------
    story.append(Paragraph("Model & Reproducibility", _h1))
    story.append(
        _kv_table(
            [
                ("Model", str(model.get("display_name") or "Spectra-AASIST3 INT8")),
                ("Architecture", str(model.get("architecture") or "XLS-R-300M + KAN-enhanced AASIST")),
                ("Runtime", str(model.get("runtime") or "ONNX Runtime CPU")),
                ("Model artifact", str(model.get("repository") or "Not available")),
                ("Model revision", str(model.get("revision") or "Not available")),
                ("Decision threshold", str(model.get("threshold_description") or "Not available")),
                ("Analysis sample rate", f"{ai['sample_rate_hz']} Hz"),
                ("Inference time", f"{perf['inference_time_ms']:.0f} ms"),
            ]
        )
    )
    story.append(Spacer(1, 14))
    story.append(Paragraph("RESEARCH PROTOTYPE", _eyebrow))
    story.append(Paragraph(report_data["disclaimer"] or DISCLAIMER_TEXT, _muted))

    doc.build(story)
    return buffer.getvalue()
