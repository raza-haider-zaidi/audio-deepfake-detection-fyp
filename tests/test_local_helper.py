"""Tests for local_helper/ (the local URL ingestion helper -- see
docs/local_url_helper.md). Requires the helper's own dependency set
(requirements-helper.txt); the whole module is skipped if `fastapi` is
not installed, so the deployed Streamlit application's own test/CI
environment (which does not install requirements-helper.txt) is
unaffected by this file's absence of coverage there.

Tests that need real network access to YouTube are marked
`@pytest.mark.integration` and skipped by the default suite, consistent
with tests/test_video_url.py's existing convention.
"""

from __future__ import annotations

import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from local_helper.api import create_app  # noqa: E402
from local_helper.security import RateLimiter, generate_session_token, tokens_match  # noqa: E402
from local_helper.tunnel import TRYCLOUDFLARE_URL_RE, CloudflaredNotFoundError, locate_cloudflared  # noqa: E402
from local_helper.youtube import HelperInputError  # noqa: E402


# --------------------------------------------------------------------------
# security.py
# --------------------------------------------------------------------------


def test_generate_session_token_is_random_and_long_enough():
    a = generate_session_token()
    b = generate_session_token()
    assert a != b
    assert len(a) >= 32


def test_tokens_match_requires_exact_match():
    token = generate_session_token()
    assert tokens_match(token, token) is True
    assert tokens_match(token, token + "x") is False
    assert tokens_match(token, "") is False
    assert tokens_match("", token) is False


def test_rate_limiter_blocks_after_max_requests():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    results = [limiter.allow() for _ in range(5)]
    assert results == [True, True, True, False, False]


def test_rate_limiter_allows_again_after_window_expires(monkeypatch):
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    fake_time = {"t": 0.0}
    monkeypatch.setattr("local_helper.security.time.monotonic", lambda: fake_time["t"])

    assert limiter.allow() is True
    assert limiter.allow() is False
    fake_time["t"] = 11.0
    assert limiter.allow() is True


# --------------------------------------------------------------------------
# tunnel.py -- no real cloudflared process is started here
# --------------------------------------------------------------------------


def test_trycloudflare_url_regex_extracts_the_hostname_from_cloudflared_output():
    line = "2026-09-06T00:00:00Z INF |  https://random-words-here.trycloudflare.com   |"
    match = TRYCLOUDFLARE_URL_RE.search(line)
    assert match is not None
    assert match.group(0) == "https://random-words-here.trycloudflare.com"


def test_locate_cloudflared_raises_actionable_error_when_not_found(monkeypatch):
    monkeypatch.setattr("local_helper.tunnel._bundled_cloudflared_path", lambda: None)
    monkeypatch.setattr("local_helper.tunnel.shutil.which", lambda name: None)
    with pytest.raises(CloudflaredNotFoundError):
        locate_cloudflared()


# --------------------------------------------------------------------------
# api.py -- FastAPI TestClient, no real network/yt-dlp calls
# --------------------------------------------------------------------------

TOKEN = "unit-test-token"


@pytest.fixture()
def client():
    app = create_app(get_token=lambda: TOKEN)
    return TestClient(app)


def test_health_requires_no_auth_and_returns_minimal_fields(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert set(body.keys()) == {"status", "service", "version"}


def test_metadata_rejects_missing_authorization(client):
    response = client.post("/metadata", json={"url": "https://www.youtube.com/watch?v=abc"})
    assert response.status_code == 401


def test_metadata_rejects_wrong_token(client):
    response = client.post(
        "/metadata", json={"url": "https://www.youtube.com/watch?v=abc"}, headers={"Authorization": "Bearer wrong-token"}
    )
    assert response.status_code == 401


def test_prepare_rejects_missing_authorization(client):
    response = client.post("/prepare", json={"url": "https://www.youtube.com/watch?v=abc", "interval_start": 0, "interval_duration": 5})
    assert response.status_code == 401


def test_prepare_rejects_non_youtube_hostname_even_when_authorized(client):
    response = client.post(
        "/prepare",
        json={"url": "https://example.com/watch?v=abc", "interval_start": 0, "interval_duration": 5},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 422


def test_prepare_rejects_interval_longer_than_helper_max(client):
    response = client.post(
        "/prepare",
        json={"url": "https://www.youtube.com/watch?v=abc", "interval_start": 0, "interval_duration": 999},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    # Pydantic's Field(gt=0) always allows 999 through validation; the
    # helper's OWN interval cap is enforced inside prepare_interval itself.
    assert response.status_code == 422


def test_rate_limiting_returns_429_after_too_many_requests():
    app = create_app(get_token=lambda: TOKEN)
    client = TestClient(app)
    statuses = []
    for _ in range(10):
        r = client.get("/health")
        statuses.append(r.status_code)
    # /health itself has no rate limit applied (only authenticated
    # endpoints do) -- confirm it never 429s regardless of volume.
    assert all(s == 200 for s in statuses)

    statuses = []
    for _ in range(10):
        r = client.post("/metadata", json={"url": "https://example.com"}, headers={"Authorization": f"Bearer {TOKEN}"})
        statuses.append(r.status_code)
    assert 429 in statuses


def test_oversized_request_body_is_rejected(client):
    huge_url = "https://www.youtube.com/watch?v=" + ("a" * 20000)
    response = client.post(
        "/metadata", json={"url": huge_url}, headers={"Authorization": f"Bearer {TOKEN}", "Content-Length": str(20000)}
    )
    assert response.status_code == 413


def test_only_one_prepare_job_runs_concurrently(monkeypatch):
    """Part E: 'one active preparation job is enough' -- a second /prepare
    call made while one is already in flight must be rejected, not
    queued."""
    release_event = threading.Event()
    started_event = threading.Event()

    def _slow_prepare(url, start, duration):
        started_event.set()
        release_event.wait(timeout=5)
        raise HelperInputError("stopped for test")

    monkeypatch.setattr("local_helper.api.prepare_interval", _slow_prepare)
    app = create_app(get_token=lambda: TOKEN)
    client = TestClient(app)

    results = {}

    def _first_call():
        results["first"] = client.post(
            "/prepare",
            json={"url": "https://www.youtube.com/watch?v=abc", "interval_start": 0, "interval_duration": 5},
            headers={"Authorization": f"Bearer {TOKEN}"},
        ).status_code

    thread = threading.Thread(target=_first_call)
    thread.start()
    assert started_event.wait(timeout=5), "expected the first job to start"

    second_status = client.post(
        "/prepare",
        json={"url": "https://www.youtube.com/watch?v=abc", "interval_start": 0, "interval_duration": 5},
        headers={"Authorization": f"Bearer {TOKEN}"},
    ).status_code
    assert second_status == 429

    release_event.set()
    thread.join(timeout=5)
    assert results["first"] == 422


# --------------------------------------------------------------------------
# Real end-to-end retrieval against YouTube -- network required, not part
# of the default suite.
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_prepare_real_video_end_to_end():
    app = create_app(get_token=lambda: TOKEN)
    client = TestClient(app)
    response = client.post(
        "/metadata",
        json={"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["title"]

    response = client.post(
        "/prepare",
        json={"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw", "interval_start": 1.0, "interval_duration": 4.0},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sample_rate"] == 16000
    assert body["duration_seconds"] == pytest.approx(4.0, abs=0.1)
    assert body["audio_base64"]
