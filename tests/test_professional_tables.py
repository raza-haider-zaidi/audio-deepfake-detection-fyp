"""Tests for the shared professional table components (app/components.py)
used across segment evidence, batch analysis, robustness, evaluation
comparisons, and session history -- replacing the default st.dataframe
look everywhere in the app."""

from app.components import professional_table_html, status_chip_html


def test_status_chip_html_known_state_renders_chip_with_text():
    html = status_chip_html("SPOOF")
    assert "adf-chip" in html
    assert "SPOOF" in html


def test_status_chip_html_unknown_value_falls_back_to_escaped_text():
    html = status_chip_html("Some free text")
    assert "adf-chip" not in html
    assert "Some free text" in html


def test_status_chip_html_never_relies_on_color_alone():
    # Text must always be present verbatim, regardless of chip styling.
    for value in ("BONAFIDE", "SPOOF", "INCONCLUSIVE", "GOOD", "LIMITED", "POOR"):
        assert value in status_chip_html(value)


def test_professional_table_html_renders_header_and_rows():
    html = professional_table_html(["Name", "Result"], [{"Name": "a.wav", "Result": "BONAFIDE"}], status_columns={"Result"})
    assert "<table" in html and "adf-table" in html
    assert "a.wav" in html
    assert "adf-chip" in html  # Result rendered as a semantic chip


def test_professional_table_html_wraps_in_scrollable_container():
    html = professional_table_html(["A"], [{"A": "1"}])
    assert "adf-table-wrap" in html


def test_professional_table_html_numeric_column_right_aligned():
    html = professional_table_html(["Count"], [{"Count": "5"}], numeric_columns={"Count"})
    assert "adf-td-num" in html


def test_professional_table_html_missing_value_shows_placeholder_not_blank():
    html = professional_table_html(["A", "B"], [{"A": "x", "B": None}])
    assert "—" in html


def test_professional_table_html_escapes_html_in_values():
    html = professional_table_html(["A"], [{"A": "<script>alert(1)</script>"}])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_professional_table_html_empty_rows_still_renders_header():
    html = professional_table_html(["A", "B"], [])
    assert "<th" in html and "A" in html and "B" in html


def test_professional_table_html_stack_on_mobile_emits_kv_cards():
    html = professional_table_html(["A"], [{"A": "1"}], stack_on_mobile=True)
    assert "adf-kv-card" in html
    assert "adf-table--stack-on-mobile" in html
