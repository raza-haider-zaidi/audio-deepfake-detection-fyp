"""Tests for app/analysis/url_helper_client.py -- the Streamlit-side
client for the local URL ingestion helper. No network access: HTTP calls
are monkeypatched at the `requests` layer."""

from __future__ import annotations

import base64
import json

import pytest

from app.analysis import url_helper_client as client_module
from app.analysis.url_helper_client import (
    HelperConnection,
    HelperConnectionError,
    HelperRequestError,
    decode_connection_code,
    fetch_metadata_via_helper,
    prepare_via_helper,
    verify_connection,
)


def _make_code(endpoint="https://abc123.trycloudflare.com", token="secret-token", version=1) -> str:
    payload = {"endpoint": endpoint, "token": token, "version": version}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return f"ADA1:{encoded}"


# --------------------------------------------------------------------------
# decode_connection_code -- pure parsing/validation, no network
# --------------------------------------------------------------------------


def test_decode_connection_code_round_trips_a_valid_code():
    connection = decode_connection_code(_make_code())
    assert connection.endpoint == "https://abc123.trycloudflare.com"
    assert connection.token == "secret-token"


def test_decode_connection_code_rejects_missing_prefix():
    with pytest.raises(HelperConnectionError):
        decode_connection_code("not-a-real-code")


def test_decode_connection_code_rejects_garbage_base64():
    with pytest.raises(HelperConnectionError):
        decode_connection_code("ADA1:!!!not-base64!!!")


def test_decode_connection_code_rejects_non_json_payload():
    encoded = base64.urlsafe_b64encode(b"not json at all").decode("ascii")
    with pytest.raises(HelperConnectionError):
        decode_connection_code(f"ADA1:{encoded}")


def test_decode_connection_code_rejects_http_scheme():
    with pytest.raises(HelperConnectionError):
        decode_connection_code(_make_code(endpoint="http://abc123.trycloudflare.com"))


def test_decode_connection_code_rejects_non_trycloudflare_hostname():
    """Defense against a connection code pointing somewhere else entirely
    -- e.g. an attacker-controlled HTTPS endpoint."""
    with pytest.raises(HelperConnectionError):
        decode_connection_code(_make_code(endpoint="https://evil.example.com"))


def test_decode_connection_code_rejects_unsupported_version():
    with pytest.raises(HelperConnectionError):
        decode_connection_code(_make_code(version=2))


def test_decode_connection_code_rejects_missing_token():
    payload = {"endpoint": "https://abc123.trycloudflare.com", "token": "", "version": 1}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    with pytest.raises(HelperConnectionError):
        decode_connection_code(f"ADA1:{encoded}")


def test_decode_connection_code_never_executes_arbitrary_content():
    """No pickle, no eval -- a payload that LOOKS like it could be
    dangerous is just inert JSON data, never executed."""
    payload = {"endpoint": "https://abc123.trycloudflare.com", "token": "__import__('os').system('echo pwned')", "version": 1}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    connection = decode_connection_code(f"ADA1:{encoded}")
    assert connection.token == "__import__('os').system('echo pwned')"


# --------------------------------------------------------------------------
# HTTP calls -- requests is monkeypatched, no real network
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code, json_body):
        self.status_code = status_code
        self._json_body = json_body

    def json(self):
        return self._json_body


def test_verify_connection_raises_on_network_error(monkeypatch):
    import requests

    def _raise(*a, **k):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(client_module.requests, "get", _raise)
    with pytest.raises(HelperRequestError):
        verify_connection(HelperConnection(endpoint="https://x.trycloudflare.com", token="t"))


def test_verify_connection_raises_when_not_ready(monkeypatch):
    monkeypatch.setattr(client_module.requests, "get", lambda *a, **k: _FakeResponse(200, {"status": "starting"}))
    with pytest.raises(HelperRequestError):
        verify_connection(HelperConnection(endpoint="https://x.trycloudflare.com", token="t"))


def test_verify_connection_succeeds_when_ready(monkeypatch):
    monkeypatch.setattr(client_module.requests, "get", lambda *a, **k: _FakeResponse(200, {"status": "ready"}))
    verify_connection(HelperConnection(endpoint="https://x.trycloudflare.com", token="t"))  # no raise


def test_fetch_metadata_via_helper_sends_bearer_token(monkeypatch):
    captured = {}

    def _fake_post(url, json, headers, timeout):
        captured["headers"] = headers
        return _FakeResponse(200, {"platform": "YouTube", "title": "T", "duration_seconds": 10.0, "uploader": "u", "webpage_url": url, "thumbnail_url": None})

    monkeypatch.setattr(client_module.requests, "post", _fake_post)
    meta = fetch_metadata_via_helper(HelperConnection(endpoint="https://x.trycloudflare.com", token="the-token"), "https://www.youtube.com/watch?v=abc")
    assert captured["headers"]["Authorization"] == "Bearer the-token"
    assert meta.title == "T"


def test_fetch_metadata_via_helper_raises_safe_message_on_error_response(monkeypatch):
    monkeypatch.setattr(client_module.requests, "post", lambda *a, **k: _FakeResponse(422, {"detail": "bad url"}))
    with pytest.raises(HelperRequestError, match="bad url"):
        fetch_metadata_via_helper(HelperConnection(endpoint="https://x.trycloudflare.com", token="t"), "https://www.youtube.com/watch?v=abc")


def test_prepare_via_helper_decodes_base64_audio(monkeypatch):
    wav_bytes = b"RIFF....WAVEfmt fake bytes"
    body = {
        "sample_rate": 16000,
        "duration_seconds": 5.0,
        "source_extension": ".webm",
        "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
        "metadata": {"platform": "YouTube", "title": "T", "duration_seconds": 19.0, "uploader": "u", "webpage_url": "https://www.youtube.com/watch?v=abc"},
    }
    monkeypatch.setattr(client_module.requests, "post", lambda *a, **k: _FakeResponse(200, body))
    prepared = prepare_via_helper(HelperConnection(endpoint="https://x.trycloudflare.com", token="t"), "https://www.youtube.com/watch?v=abc", 0.0, 5.0)
    assert prepared.wav_bytes == wav_bytes
    assert prepared.metadata.title == "T"
