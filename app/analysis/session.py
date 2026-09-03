"""Temporary, in-session "Analysis Session" tracking.

Stores only lightweight, already-computed summary fields in
`st.session_state` -- filename, SHA-256, metadata, result, probabilities,
timestamp, model/version info, and segment summary. Raw audio waveforms
are NEVER stored here. This is explicitly a temporary, per-browser-session
convenience, not a persistent case database -- see docs/analysis_platform_v2.md.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

SESSION_KEY = "analysis_session_entries"
MAX_SESSION_ENTRIES = 25  # bounds memory growth for a long-running session


def record_analysis(session_state: dict, entry: dict[str, Any]) -> None:
    entries = session_state.setdefault(SESSION_KEY, [])
    entries.append(entry)
    if len(entries) > MAX_SESSION_ENTRIES:
        del entries[0 : len(entries) - MAX_SESSION_ENTRIES]


def get_session_entries(session_state: dict) -> list[dict[str, Any]]:
    return session_state.get(SESSION_KEY, [])


def remove_entry(session_state: dict, index: int) -> None:
    entries = session_state.get(SESSION_KEY, [])
    if 0 <= index < len(entries):
        del entries[index]


def clear_session(session_state: dict) -> None:
    session_state[SESSION_KEY] = []


def build_session_entry(
    *,
    filename: str,
    sha256: str,
    duration_seconds: float,
    sample_rate: int,
    audio_format: str,
    presentation_state: str,
    bonafide_probability: float,
    spoof_probability: float,
    model_id: str,
    model_revision: str,
    n_segments: int,
    segment_agreement_level: str | None,
    source_type: str = "audio_file",
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "sha256": sha256,
        "duration_seconds": round(duration_seconds, 2),
        "sample_rate": sample_rate,
        "format": audio_format,
        "presentation_state": presentation_state,
        "bonafide_probability": bonafide_probability,
        "spoof_probability": spoof_probability,
        "model_id": model_id,
        "model_revision": model_revision,
        "n_segments": n_segments,
        "segment_agreement_level": segment_agreement_level,
        "source_type": source_type,
    }


def export_session_summary(entries: list[dict[str, Any]]) -> str:
    return json.dumps({"session_data_notice": "Temporary session data, not a persistent case database.", "analyses": entries}, indent=2)
