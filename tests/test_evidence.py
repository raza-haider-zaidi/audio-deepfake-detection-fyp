"""Tests for app/analysis/evidence.py -- segment table, segment
agreement, and evidence-summary sentence generation. All descriptive;
none of this touches the frozen decision logic."""

from audio_deepfake_detector.models.candidate_e import INT8_DYNAMIC_CALIBRATED_THRESHOLD

from app.analysis.evidence import (
    SEGMENT_DURATION_SECONDS,
    compute_segment_agreement,
    evidence_summary_sentences,
    segment_table,
)

THRESHOLD = INT8_DYNAMIC_CALIBRATED_THRESHOLD


def test_segment_table_row_count_matches_input():
    rows = segment_table([0.1, 0.9, 0.5], THRESHOLD, clip_duration_seconds=12.0)
    assert len(rows) == 3
    assert [r["segment"] for r in rows] == [1, 2, 3]


def test_segment_table_time_ranges_sequential_non_overlapping():
    rows = segment_table([0.1, 0.9], THRESHOLD, clip_duration_seconds=8.0)
    assert rows[0]["start_seconds"] == 0.0
    assert rows[0]["end_seconds"] == SEGMENT_DURATION_SECONDS
    assert rows[1]["start_seconds"] == SEGMENT_DURATION_SECONDS


def test_segment_table_last_segment_clipped_to_clip_duration():
    rows = segment_table([0.1, 0.9], THRESHOLD, clip_duration_seconds=6.0)
    assert rows[-1]["end_seconds"] == 6.0


def test_segment_table_probabilities_sum_to_one():
    rows = segment_table([0.3], THRESHOLD, clip_duration_seconds=4.0)
    assert abs(rows[0]["bonafide_probability"] + rows[0]["spoof_probability"] - 1.0) < 1e-9


def test_segment_table_interpretation_uses_frozen_presentation_rule():
    rows = segment_table([0.2, 0.7, THRESHOLD + 0.001], THRESHOLD, clip_duration_seconds=12.0)
    assert rows[0]["interpretation"] == "BONAFIDE"
    assert rows[1]["interpretation"] == "INCONCLUSIVE"
    assert rows[2]["interpretation"] == "SPOOF"


def test_compute_segment_agreement_none_for_single_segment():
    assert compute_segment_agreement([0.9]) is None


def test_compute_segment_agreement_high_when_unanimous():
    agreement = compute_segment_agreement([0.9, 0.95, 0.99, 0.85])
    assert agreement.level == "High"
    assert agreement.dominant_direction == "spoof"
    assert agreement.agreement_percent == 100.0
    assert agreement.n_segments == 4


def test_compute_segment_agreement_mixed_when_split():
    agreement = compute_segment_agreement([0.9, 0.1, 0.8, 0.2])
    assert agreement.level == "Mixed"
    assert agreement.agreement_percent == 50.0


def test_compute_segment_agreement_moderate_level():
    # 3 of 4 = 75% -> Moderate (65-85% band)
    agreement = compute_segment_agreement([0.9, 0.9, 0.9, 0.1])
    assert agreement.level == "Moderate"
    assert agreement.agreement_percent == 75.0


def test_evidence_summary_sentences_are_factual_no_speculative_claims():
    agreement = compute_segment_agreement([0.9, 0.95, 0.99])
    sentences = evidence_summary_sentences("SPOOF", 0.97, THRESHOLD, agreement)
    joined = " ".join(sentences).lower()
    forbidden = ["breathing", "robotic", "prosody", "unnatural pause", "ai artifact"]
    for phrase in forbidden:
        assert phrase not in joined


def test_evidence_summary_sentences_inconclusive_wording():
    sentences = evidence_summary_sentences("INCONCLUSIVE", 0.918, THRESHOLD, None)
    assert any("mixed-evidence zone" in s for s in sentences)


def test_evidence_summary_sentences_reports_threshold_and_observed_score():
    sentences = evidence_summary_sentences("BONAFIDE", 0.03, THRESHOLD, None)
    joined = " ".join(sentences)
    assert "3.0%" in joined
    assert f"{THRESHOLD * 100:.2f}%" in joined
