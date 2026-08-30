"""Tests for app/components.py's pure markup-building functions.

No Streamlit runtime needed — these are plain string builders.
"""

from app.components import (
    HOW_IT_WORKS_STEPS,
    brand_mark_svg,
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
