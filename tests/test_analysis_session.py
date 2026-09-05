"""Tests for app/analysis/session.py -- the temporary, session-scoped
Analysis Session store. Never touches Streamlit; operates on a plain dict
standing in for st.session_state."""

import json

from app.analysis.session import (
    MAX_SESSION_ENTRIES,
    SESSION_KEY,
    build_session_entry,
    clear_session,
    export_session_summary,
    get_session_entries,
    record_analysis,
    remove_entry,
)


def _entry(filename="clip.wav"):
    return build_session_entry(
        filename=filename,
        sha256="abc123",
        duration_seconds=3.456,
        sample_rate=16000,
        audio_format="wav",
        presentation_state="BONAFIDE",
        bonafide_probability=0.9,
        spoof_probability=0.1,
        model_id="spectra_aasist3_onnx_int8",
        model_revision="deadbeef",
        n_segments=1,
        segment_agreement_level="Unanimous",
    )


def test_record_analysis_appends_entry():
    state = {}
    record_analysis(state, _entry())
    assert len(get_session_entries(state)) == 1


def test_build_session_entry_never_includes_raw_audio_key():
    entry = _entry()
    assert "waveform" not in entry
    assert "audio" not in entry
    assert entry["duration_seconds"] == 3.46


def test_record_analysis_bounds_growth_to_max_entries():
    state = {}
    for i in range(MAX_SESSION_ENTRIES + 10):
        record_analysis(state, _entry(filename=f"clip{i}.wav"))
    entries = get_session_entries(state)
    assert len(entries) == MAX_SESSION_ENTRIES
    assert entries[-1]["filename"] == f"clip{MAX_SESSION_ENTRIES + 9}.wav"


def test_remove_entry_removes_by_index():
    state = {}
    record_analysis(state, _entry("a.wav"))
    record_analysis(state, _entry("b.wav"))
    remove_entry(state, 0)
    entries = get_session_entries(state)
    assert len(entries) == 1
    assert entries[0]["filename"] == "b.wav"


def test_clear_session_empties_entries():
    state = {}
    record_analysis(state, _entry())
    clear_session(state)
    assert get_session_entries(state) == []


def test_get_session_entries_empty_by_default():
    assert get_session_entries({}) == []


def test_export_session_summary_is_valid_json_with_notice():
    state = {}
    record_analysis(state, _entry())
    payload = json.loads(export_session_summary(get_session_entries(state)))
    assert "session_data_notice" in payload
    assert "temporary" in payload["session_data_notice"].lower()
    assert len(payload["analyses"]) == 1
    assert SESSION_KEY  # sanity: constant exists and is truthy
