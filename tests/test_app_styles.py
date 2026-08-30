"""Tests for app/styles.py — design tokens and the injected stylesheet."""

from app.styles import COLORS, STATE_TOKENS, inject_global_styles


def test_inject_global_styles_returns_style_block():
    css = inject_global_styles()
    assert css.strip().startswith("<style>") or "<style>" in css
    assert "</style>" in css


def test_inject_global_styles_contains_core_tokens():
    css = inject_global_styles()
    assert COLORS["bg"] in css
    assert COLORS["accent"] in css


def test_inject_global_styles_respects_reduced_motion():
    css = inject_global_styles()
    assert "prefers-reduced-motion" in css


def test_state_tokens_defined_for_all_three_states():
    assert set(STATE_TOKENS.keys()) == {"BONAFIDE", "SPOOF", "INCONCLUSIVE"}
    for tokens in STATE_TOKENS.values():
        assert "fg" in tokens and "bg" in tokens and "border" in tokens
