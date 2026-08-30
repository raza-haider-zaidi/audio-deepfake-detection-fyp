"""Tests for app/analysis/threshold_explorer.py -- a research-only,
descriptive threshold sweep over frozen evaluation scores. Never touches
the production classifier or its threshold."""

from app.analysis.threshold_explorer import (
    load_int8_evaluation_scores,
    sweep_thresholds,
)

RECORDS = [
    {"utterance_id": "a", "label": "spoof", "spoof_score": 0.9},
    {"utterance_id": "b", "label": "spoof", "spoof_score": 0.6},
    {"utterance_id": "c", "label": "bonafide", "spoof_score": 0.2},
    {"utterance_id": "d", "label": "bonafide", "spoof_score": 0.4},
]


def test_sweep_thresholds_low_threshold_maximizes_fpr():
    points = sweep_thresholds(RECORDS, [0.01])
    assert points[0].fpr == 1.0
    assert points[0].fnr == 0.0


def test_sweep_thresholds_high_threshold_maximizes_fnr():
    points = sweep_thresholds(RECORDS, [0.99])
    assert points[0].fnr == 1.0
    assert points[0].fpr == 0.0


def test_sweep_thresholds_mid_threshold_separates_correctly():
    points = sweep_thresholds(RECORDS, [0.5])
    p = points[0]
    assert p.fpr == 0.0
    assert p.fnr == 0.0
    assert p.accuracy == 1.0
    assert p.balanced_accuracy == 1.0


def test_sweep_thresholds_returns_one_point_per_threshold():
    points = sweep_thresholds(RECORDS, [0.1, 0.5, 0.9])
    assert [p.threshold for p in points] == [0.1, 0.5, 0.9]


def test_load_int8_evaluation_scores_returns_none_or_valid_shape():
    data = load_int8_evaluation_scores()
    if data is not None:
        assert "records" in data
        assert isinstance(data["records"], list)
        if data["records"]:
            record = data["records"][0]
            assert "label" in record
            assert "spoof_score" in record


def test_load_int8_evaluation_scores_reproduces_frozen_production_metrics():
    data = load_int8_evaluation_scores()
    if data is None:
        return  # data file not present in this checkout -- nothing to verify
    threshold = data["_provenance"]["threshold_used_in_production"]
    points = sweep_thresholds(data["records"], [threshold])
    p = points[0]
    assert round(p.fpr, 2) == 0.02
    assert round(p.fnr, 2) == 0.10
