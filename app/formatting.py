"""Pure formatting helpers for presenting PredictionResult in the UI.

Kept free of any Streamlit import so these are trivially unit-testable.
"""

from __future__ import annotations

from audio_deepfake_detector.utils.datatypes import PredictionResult

FRIENDLY_LABELS = {
    "BONAFIDE": "Likely Real / Bonafide",
    "SPOOF": "Likely AI-Generated / Spoofed",
    "INCONCLUSIVE": "Inconclusive / Mixed Evidence",
}

INCONCLUSIVE_EXPLANATION = (
    "The model detected elevated spoof indicators, but the score did not "
    "cross the calibrated spoof threshold."
)


def friendly_label(state: str) -> str:
    """Map a normalized label or presentation state
    (BONAFIDE/SPOOF/INCONCLUSIVE) to a user-facing phrase."""
    return FRIENDLY_LABELS.get(state, state)


def format_percentage(value: float) -> str:
    """Format a 0-1 probability as a percentage string, e.g. '87.4%'."""
    return f"{value * 100:.1f}%"


CONFIDENCE_LABELS = {
    "BONAFIDE": "Bonafide class probability",
    "SPOOF": "Spoof class probability",
}


def result_summary(result: PredictionResult, calibrated_threshold: float | None = None) -> dict[str, str | bool]:
    """Return the primary result-card fields as display-ready strings.

    `presentation_state` (BONAFIDE/SPOOF/INCONCLUSIVE) drives the headline
    prediction text -- it falls back to `result.normalized_label` (binary,
    no INCONCLUSIVE) for adapters that don't populate it, e.g. Sara. The
    underlying scientific decision (`result.binary_model_decision` /
    `result.raw_label` / `result.normalized_label`) is NEVER changed by
    this -- INCONCLUSIVE is a presentation-only abstention state for the
    narrow zone where a naive 50/50 softmax split would disagree with the
    calibrated (low-bonafide-FPR) decision. See
    docs/spectra_inconclusive_state.md.

    `confidence` is always the probability of the underlying binary
    decision's predicted class (never the losing class) -- but a
    calibrated threshold far from 50% means that probability can
    legitimately be below 50%, which is exactly the INCONCLUSIVE trigger
    condition. `confidence_label` gives a class-specific, non-misleading
    label ("Bonafide/Spoof class probability") to pair with it; it is not
    shown as the headline metric when the result is INCONCLUSIVE (the
    calibrated-threshold context is shown instead -- see
    `calibrated_threshold_percent`/`observed_spoof_probability_percent`/
    `inconclusive_explanation`).
    """
    bonafide_prob = result.probabilities.get("bonafide", 0.0)
    spoof_prob = result.probabilities.get("spoof", 0.0)
    state = result.presentation_state or result.normalized_label
    summary: dict[str, str | bool] = {
        "prediction": friendly_label(state),
        "presentation_state": state,
        "is_inconclusive": state == "INCONCLUSIVE",
        "confidence": format_percentage(result.confidence),
        "confidence_label": CONFIDENCE_LABELS.get(result.normalized_label, "Predicted class probability"),
        "bonafide_probability": format_percentage(bonafide_prob),
        "spoof_probability": format_percentage(spoof_prob),
    }
    if state == "INCONCLUSIVE":
        summary["inconclusive_explanation"] = INCONCLUSIVE_EXPLANATION
        summary["observed_spoof_probability"] = format_percentage(spoof_prob)
        if calibrated_threshold is not None:
            summary["calibrated_threshold"] = format_percentage(calibrated_threshold)
    return summary


def window_table_rows(result: PredictionResult) -> list[dict[str, str]]:
    """Build rows for the optional window-level analysis table."""
    rows = []
    for wp in result.window_predictions:
        rows.append(
            {
                "Window": wp.window_index + 1,
                "Start time (s)": f"{wp.start_sample / _sample_rate_hint(result):.2f}",
                "End time (s)": f"{wp.end_sample / _sample_rate_hint(result):.2f}",
                "Bonafide probability": format_percentage(wp.probabilities.get("bonafide", 0.0)),
                "Spoof probability": format_percentage(wp.probabilities.get("spoof", 0.0)),
                "Prediction": friendly_label(
                    "BONAFIDE" if wp.raw_label == "bonafide" else "SPOOF"
                ),
            }
        )
    return rows


def _sample_rate_hint(result: PredictionResult) -> float:
    """Window start/end are stored in samples; assume the standard 16 kHz
    pipeline rate used throughout this project for display purposes."""
    return 16000.0
