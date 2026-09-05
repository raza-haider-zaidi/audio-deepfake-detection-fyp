"""Segment-evidence timeline, segment agreement, and factual evidence
summaries -- all DESCRIPTIVE, supplementary presentation built from the
frozen detector's own outputs.

Segment-level scores come from `candidate_e.CandidateSpectraAasist3OnnxDetector.predict_full_clip()`,
an existing, UNCHANGED project-level extension (sequential, non-overlapping
64,600-sample windows) that is separate from the production single-window
`predict()` decision. This module NEVER derives the presented BONAFIDE/
SPOOF/INCONCLUSIVE result -- that always comes from `predict()`'s own
`presentation_state`. Per-segment "interpretation" labels below apply the
SAME calibrated threshold/presentation rule to each individual segment
score for descriptive purposes only; this is explicitly NOT an
independently validated per-segment threshold, and is labeled as such.

Segment-agreement categories (High/Moderate/Mixed) are a documented,
deterministic rule over the segment interpretations -- they are advisory
text only and never alter the classifier's threshold, decision, or
presentation state (see docs/analysis_platform.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from audio_deepfake_detector.models.candidate_e import REQUIRED_SAMPLES

SEGMENT_DURATION_SECONDS = REQUIRED_SAMPLES / 16000  # ~4.0375s, non-overlapping


def presentation_state_for_score(spoof_prob: float, threshold: float) -> str:
    """Applies the SAME rule as candidate_e.presentation_state() to a
    single segment score, for descriptive display only. Duplicated here
    (rather than imported) is not the goal -- it IS imported, see below;
    this docstring exists only to make explicit that no new rule is
    invented for segments."""
    from audio_deepfake_detector.models.candidate_e import presentation_state

    return presentation_state(spoof_prob, threshold)


def segment_table(window_spoof_probs: list[float], threshold: float, clip_duration_seconds: float) -> list[dict]:
    """One row per analyzed segment: number, time range, both class
    probabilities, and a descriptive interpretation using the frozen
    presentation rule. Segments are sequential and non-overlapping."""
    rows = []
    for i, spoof_prob in enumerate(window_spoof_probs):
        start = i * SEGMENT_DURATION_SECONDS
        end = min((i + 1) * SEGMENT_DURATION_SECONDS, clip_duration_seconds) if i == len(window_spoof_probs) - 1 else (i + 1) * SEGMENT_DURATION_SECONDS
        rows.append(
            {
                "segment": i + 1,
                "start_seconds": start,
                "end_seconds": end,
                "bonafide_probability": 1.0 - spoof_prob,
                "spoof_probability": spoof_prob,
                "interpretation": presentation_state_for_score(spoof_prob, threshold),
            }
        )
    return rows


@dataclass
class SegmentAgreement:
    n_segments: int
    n_spoof_leaning: int
    n_bonafide_leaning: int
    dominant_direction: str  # "spoof" | "bonafide" | "tied"
    agreement_percent: float
    score_mean: float
    score_median: float
    score_std: float
    score_range: float
    level: str  # "High" | "Moderate" | "Mixed"


AGREEMENT_HIGH_THRESHOLD = 0.85
AGREEMENT_MODERATE_THRESHOLD = 0.65


def compute_segment_agreement(window_spoof_probs: list[float]) -> SegmentAgreement | None:
    """Descriptive-only summary of how consistently segments leaned
    spoof vs. bonafide (spoof_prob > 0.5 = spoof-leaning, else
    bonafide-leaning -- a plain majority split, NOT the calibrated
    threshold, since this measures raw directional agreement across
    segments, not the deployment decision).

    Documented rule for `level`:
      High     -- agreement_percent >= 85%
      Moderate -- agreement_percent >= 65%
      Mixed    -- otherwise

    Returns None for a single-segment clip (agreement is not a
    meaningful concept with only one data point).
    """
    import numpy as np

    n = len(window_spoof_probs)
    if n <= 1:
        return None

    scores = np.asarray(window_spoof_probs, dtype=np.float64)
    n_spoof_leaning = int(np.sum(scores > 0.5))
    n_bonafide_leaning = n - n_spoof_leaning

    if n_spoof_leaning > n_bonafide_leaning:
        dominant_direction = "spoof"
        agreement_percent = 100.0 * n_spoof_leaning / n
    elif n_bonafide_leaning > n_spoof_leaning:
        dominant_direction = "bonafide"
        agreement_percent = 100.0 * n_bonafide_leaning / n
    else:
        dominant_direction = "tied"
        agreement_percent = 50.0

    if agreement_percent >= AGREEMENT_HIGH_THRESHOLD * 100:
        level = "High"
    elif agreement_percent >= AGREEMENT_MODERATE_THRESHOLD * 100:
        level = "Moderate"
    else:
        level = "Mixed"

    return SegmentAgreement(
        n_segments=n,
        n_spoof_leaning=n_spoof_leaning,
        n_bonafide_leaning=n_bonafide_leaning,
        dominant_direction=dominant_direction,
        agreement_percent=agreement_percent,
        score_mean=float(np.mean(scores)),
        score_median=float(np.median(scores)),
        score_std=float(np.std(scores)),
        score_range=float(np.max(scores) - np.min(scores)),
        level=level,
    )


def evidence_summary_sentences(
    presentation_state: str,
    spoof_probability: float,
    threshold: float,
    agreement: SegmentAgreement | None,
) -> list[str]:
    """Factual, measurable sentences only -- no speculative claims about
    breathing, pitch, prosody, or "AI artifacts" the system does not
    actually compute."""
    sentences = [
        f"Overall spoof probability was {spoof_probability * 100:.1f}%.",
        f"The calibrated spoof threshold is {threshold * 100:.2f}%.",
    ]
    if presentation_state == "INCONCLUSIVE":
        sentences.append(
            "The recording therefore fell inside the application's mixed-evidence zone."
        )
    elif presentation_state == "SPOOF":
        sentences.append("The observed spoof probability crossed the calibrated threshold.")
    else:
        sentences.append("The observed spoof probability did not cross the calibrated threshold.")

    if agreement is not None:
        sentences.append(
            f"{agreement.n_spoof_leaning if agreement.dominant_direction == 'spoof' else agreement.n_bonafide_leaning} "
            f"of {agreement.n_segments} analyzed segments leaned {agreement.dominant_direction} "
            f"({agreement.agreement_percent:.0f}% agreement)."
        )
        if agreement.level == "High":
            sentences.append("Segment scores were highly consistent across the recording.")
        elif agreement.level == "Mixed":
            sentences.append("Segment results were inconsistent across portions of the recording.")

    return sentences
