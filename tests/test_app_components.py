"""Tests for app/components.py's pure markup-building functions.

No Streamlit runtime needed — these are plain string builders.
"""

from app.components import (
    HOW_IT_WORKS_STEPS,
    analysis_condition_html,
    brand_mark_svg,
    diagnostics_grid_html,
    media_identity_html,
    metadata_card_html,
    metadata_grid_html,
    metric_strip_html,
    metrics_row_html,
    probability_bar_html,
    probability_comparison_html,
    result_panel_html,
    step_flow_html,
    threshold_visualization_html,
)
from app.styles import STATE_TOKENS


def test_brand_mark_svg_is_valid_inline_svg():
    svg = brand_mark_svg()
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "<rect" in svg


def test_probability_bar_html_formats_percentage():
    html = probability_bar_html("Bonafide", 0.734, "#3ecf8e")
    assert "73.4%" in html
    assert "width:73.4%" in html
    assert "Bonafide" in html


def test_probability_bar_html_clamps_out_of_range_values():
    assert "100.0%" in probability_bar_html("Spoof", 1.5, "#f2765a")
    assert "0.0%" in probability_bar_html("Spoof", -0.2, "#f2765a")


def test_probability_comparison_html_contains_both_classes():
    html = probability_comparison_html(0.078, 0.922)
    assert "7.8%" in html
    assert "92.2%" in html
    assert "Bonafide" in html
    assert "Spoof" in html


def test_threshold_visualization_html_contains_observed_and_threshold():
    html = threshold_visualization_html(0.922, 0.939693808555603)
    assert "92.2%" in html
    assert "94.0%" in html


def test_result_panel_html_uses_correct_state_tokens_for_each_state():
    for state in ("BONAFIDE", "SPOOF", "INCONCLUSIVE"):
        html = result_panel_html(state, "Title", "Explanation")
        tokens = STATE_TOKENS[state]
        assert tokens["fg"] in html
        assert tokens["bg"] in html
        assert "ANALYSIS RESULT" in html
        assert "Title" in html
        assert "Explanation" in html


def test_result_panel_html_unknown_state_falls_back_to_inconclusive_tokens():
    html = result_panel_html("SOMETHING_UNEXPECTED", "Title", "Explain")
    assert STATE_TOKENS["INCONCLUSIVE"]["fg"] in html


def test_result_panel_html_groups_source_metadata_and_advisory():
    html = result_panel_html(
        "SPOOF",
        "Likely AI-Generated / Spoofed",
        "Explanation",
        source_label="Audio File",
        metadata=[("Duration", "3.6 sec"), ("Runtime", "CPU / ONNX")],
        advisory="Capture advisory",
    )
    assert "adf-result-source" in html
    assert "Audio File" in html
    assert "3.6 sec" in html
    assert "CPU / ONNX" in html
    assert "Capture advisory" in html


def test_diagnostics_grid_wraps_groups_and_escapes_values():
    html = diagnostics_grid_html([("File", [("Filename", "clip<take>.wav")]), ("Signal", [("RMS level", "0.120")])])
    assert html.count("adf-metadata-card") == 2
    assert "clip&lt;take&gt;.wav" in html
    assert "clip<take>.wav" not in html


def test_metadata_system_preserves_full_technical_value_in_title():
    full_hash = "a" * 64
    html = metadata_card_html("Reproducibility", [("SHA-256", "aaaaaaaaaaaaaaaa…", full_hash)])
    assert "aaaaaaaaaaaaaaaa…" in html
    assert f'title="{full_hash}"' in html


def test_metadata_grid_strip_and_identity_use_shared_classes():
    grid = metadata_grid_html([("Media", [("Codec", "PCM")])])
    strip = metric_strip_html([("Sample rate", "16000 Hz")])
    identity = media_identity_html("a-very-long-recording-name.wav", [("File size", "188 KB")])
    assert "adf-metadata-grid" in grid
    assert "adf-metadata-strip" in strip
    assert "adf-media-identity-name" in identity


def test_analysis_condition_keeps_warning_detail_descriptive():
    html = analysis_condition_html("Limited", ["high silence", "low level"])
    assert "Analysis conditions: Limited" in html
    assert "high silence and low level" in html
    assert "interpreted cautiously" in html


def test_metrics_row_html_renders_all_items():
    html = metrics_row_html([("28.4 sec", "Audio duration"), ("14", "Segments analyzed")])
    assert "28.4 sec" in html
    assert "Audio duration" in html
    assert "14" in html


def test_step_flow_html_renders_all_steps():
    html = step_flow_html(HOW_IT_WORKS_STEPS)
    for number, title, _body in HOW_IT_WORKS_STEPS:
        assert number in html
        assert title in html
