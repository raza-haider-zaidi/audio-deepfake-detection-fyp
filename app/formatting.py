"""Pure formatting helpers for presenting PredictionResult in the UI.

Kept free of any Streamlit import so these are trivially unit-testable.
"""

from __future__ import annotations

from audio_deepfake_detector.utils.datatypes import PredictionResult

FRIENDLY_LABELS = {
    "BONAFIDE": "Likely Real / Bonafide",
    "SPOOF": "Likely AI-Generated / Spoofed",
}


def friendly_label(normalized_label: str) -> str:
    """Map a normalized label (BONAFIDE/SPOOF) to a user-facing phrase."""
    return FRIENDLY_LABELS.get(normalized_label, normalized_label)


def format_percentage(value: float) -> str:
    """Format a 0-1 probability as a percentage string, e.g. '87.4%'."""
    return f"{value * 100:.1f}%"


CONFIDENCE_LABELS = {
    "BONAFIDE": "Bonafide class probability",
    "SPOOF": "Spoof class probability",
}


def result_summary(result: PredictionResult) -> dict[str, str | bool]:
    """Return the primary result-card fields as display-ready strings.

    `confidence` is always the probability of the PREDICTED class (never the
    losing class) -- but a model's deployment decision may use a calibrated
    threshold far from a naive 50% split (e.g. a low-bonafide-FPR operating
    point), so the predicted class's own probability can legitimately be
    below 50% while still being the correct calibrated decision. `confidence`
    was previously mislabeled "Model confidence" everywhere it was displayed,
    which reads as an internal contradiction in exactly that situation (see
    docs/spectra_prediction_semantics_fix.md). `confidence_label` gives the
    caller a class-specific, non-misleading label to pair with it, and
    `threshold_disagreement` flags the case for callers that want to add a
    clarifying note.
    """
    bonafide_prob = result.probabilities.get("bonafide", 0.0)
    spoof_prob = result.probabilities.get("spoof", 0.0)
    return {
        "prediction": friendly_label(result.normalized_label),
        "confidence": format_percentage(result.confidence),
        "confidence_label": CONFIDENCE_LABELS.get(result.normalized_label, "Predicted class probability"),
        "bonafide_probability": format_percentage(bonafide_prob),
        "spoof_probability": format_percentage(spoof_prob),
        "threshold_disagreement": result.confidence < 0.5,
    }


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
