"""Tests for app/evaluation_loader.py -- authoritative research-data
loading, with graceful handling of missing data (never fabricated)."""

from app.evaluation_loader import get_section, load_evaluation_summary


def test_load_evaluation_summary_returns_real_committed_data():
    summary = load_evaluation_summary()
    assert summary is not None
    assert "_provenance" in summary
    assert "int8_evaluation" in summary


def test_int8_evaluation_metrics_match_frozen_project_results():
    summary = load_evaluation_summary()
    int8 = get_section(summary, "int8_evaluation")
    # These must match results/metrics/spectra_int8_evaluation.json exactly --
    # any drift here means the committed summary has gone stale.
    assert int8["eer"] == 0.030000000000000013
    assert int8["roc_auc"] == 0.9819
    assert int8["f1"] == 0.9375
    assert int8["bonafide_fpr"] == 0.02
    assert int8["spoof_fnr"] == 0.09999999999999998
    assert int8["n_samples"] == 200


def test_get_section_missing_key_returns_none_not_placeholder():
    summary = load_evaluation_summary()
    assert get_section(summary, "does_not_exist") is None


def test_get_section_none_summary_returns_none():
    assert get_section(None, "int8_evaluation") is None


def test_provenance_documents_source_files():
    summary = load_evaluation_summary()
    provenance = summary["_provenance"]
    assert len(provenance["sources"]) > 0
    assert all("results/metrics/" in s for s in provenance["sources"])
