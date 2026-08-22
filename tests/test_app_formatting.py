"""Tests for app/formatting.py — pure functions, no Streamlit needed."""

from app.formatting import format_percentage, friendly_label, result_summary, window_table_rows
from audio_deepfake_detector.utils.datatypes import PredictionResult, WindowPrediction


def _make_result(**overrides) -> PredictionResult:
    defaults = dict(
        raw_label="spoof",
        normalized_label="SPOOF",
        confidence=0.874,
        probabilities={"bonafide": 0.126, "spoof": 0.874},
        model_id="sara_wav2vec2",
        model_repository="Sara1708/deepfake-audio-wav2vec2",
        device="cpu",
        inference_time_ms=140.2,
        audio_duration_seconds=4.0,
        windows_analyzed=1,
    )
    defaults.update(overrides)
    return PredictionResult(**defaults)


def test_friendly_label_spoof():
    assert friendly_label("SPOOF") == "Likely AI-Generated / Spoofed"


def test_friendly_label_bonafide():
    assert friendly_label("BONAFIDE") == "Likely Real / Bonafide"


def test_friendly_label_unknown_passthrough():
    assert friendly_label("SOMETHING_ELSE") == "SOMETHING_ELSE"


def test_format_percentage():
    assert format_percentage(0.874) == "87.4%"
    assert format_percentage(0.0) == "0.0%"
    assert format_percentage(1.0) == "100.0%"


def test_result_summary_fields():
    result = _make_result()
    summary = result_summary(result)
    assert summary["prediction"] == "Likely AI-Generated / Spoofed"
    assert summary["confidence"] == "87.4%"
    assert summary["bonafide_probability"] == "12.6%"
    assert summary["spoof_probability"] == "87.4%"


def test_window_table_rows_empty_when_single_window():
    result = _make_result(windows_analyzed=1, window_predictions=[])
    assert window_table_rows(result) == []


def test_window_table_rows_multiple_windows():
    windows = [
        WindowPrediction(
            window_index=0,
            start_sample=0,
            end_sample=64000,
            raw_label="bonafide",
            probabilities={"bonafide": 0.9, "spoof": 0.1},
        ),
        WindowPrediction(
            window_index=1,
            start_sample=32000,
            end_sample=96000,
            raw_label="spoof",
            probabilities={"bonafide": 0.2, "spoof": 0.8},
        ),
    ]
    result = _make_result(windows_analyzed=2, window_predictions=windows)
    rows = window_table_rows(result)

    assert len(rows) == 2
    assert rows[0]["Window"] == 1
    assert rows[0]["Start time (s)"] == "0.00"
    assert rows[0]["End time (s)"] == "4.00"
    assert rows[0]["Prediction"] == "Likely Real / Bonafide"
    assert rows[1]["Window"] == 2
    assert rows[1]["Start time (s)"] == "2.00"
    assert rows[1]["Prediction"] == "Likely AI-Generated / Spoofed"
