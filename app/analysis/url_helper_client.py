"""Streamlit-side client for the LOCAL URL INGESTION HELPER (see
local_helper/ and docs/local_url_helper.md).

This module has NO dependency on yt-dlp, ffmpeg, or any media-decoding
library -- it only talks HTTPS JSON to whatever helper the user has
connected, via the `requests` library (already a transitive dependency of
this project through other packages). It is intentionally a thin,
stdlib-plus-requests client so importing it never risks pulling in a
helper-only dependency (see requirements-helper.txt) into the deployed
Streamlit environment.

Security: the connection code is opaque, non-executable JSON (base64url
of a JSON object) -- never `pickle`, never `eval`, and every field is
validated before use (scheme, hostname, version) so a malformed or
malicious paste fails safely rather than being trusted blindly. The
decoded endpoint/token are the caller's responsibility to store only in
`st.session_state` (see app/views/analyze.py) -- never logged, persisted,
or written to Streamlit secrets by this module.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

CONNECTION_CODE_PREFIX = "ADA1:"
SUPPORTED_API_VERSION = 1
ALLOWED_ENDPOINT_HOST_SUFFIX = "trycloudflare.com"

HEALTH_TIMEOUT_SECONDS = 8
METADATA_TIMEOUT_SECONDS = 20
PREPARE_TIMEOUT_SECONDS = 180


class HelperConnectionError(ValueError):
    """Raised for a malformed connection code or a helper that fails
    verification -- always safe to show directly to the user."""


class HelperRequestError(RuntimeError):
    """Raised when a verified, connected helper's API call itself fails
    (network error, non-2xx response, malformed response body)."""


@dataclass(frozen=True)
class HelperConnection:
    endpoint: str
    token: str


@dataclass(frozen=True)
class HelperVideoMetadata:
    platform: str
    title: str
    duration_seconds: float | None
    uploader: str | None
    webpage_url: str
    thumbnail_url: str | None


def decode_connection_code(code: str) -> HelperConnection:
    """Decodes and validates a connection code produced by
    local_helper.app.build_connection_code. Never executes anything from
    the value -- plain JSON parsing only."""
    code = (code or "").strip()
    if not code.startswith(CONNECTION_CODE_PREFIX):
        raise HelperConnectionError("That doesn't look like a valid connection code.")

    encoded = code[len(CONNECTION_CODE_PREFIX) :]
    padding_needed = (-len(encoded)) % 4
    try:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii") + b"=" * padding_needed)
        payload = json.loads(raw)
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise HelperConnectionError("That connection code could not be read. Please copy it again.") from exc

    if not isinstance(payload, dict):
        raise HelperConnectionError("That connection code is not in the expected format.")

    endpoint = payload.get("endpoint")
    token = payload.get("token")
    version = payload.get("version")

    if not isinstance(endpoint, str) or not isinstance(token, str) or not isinstance(version, int):
        raise HelperConnectionError("That connection code is missing required fields.")
    if version != SUPPORTED_API_VERSION:
        raise HelperConnectionError("This connection code was created by an incompatible helper version.")

    parsed = urlparse(endpoint)
    if parsed.scheme != "https":
        raise HelperConnectionError("The helper connection must use a secure (https) address.")
    hostname = (parsed.hostname or "").lower()
    if not (hostname == ALLOWED_ENDPOINT_HOST_SUFFIX or hostname.endswith("." + ALLOWED_ENDPOINT_HOST_SUFFIX)):
        raise HelperConnectionError("The helper connection address is not a recognized tunnel endpoint.")
    if not token:
        raise HelperConnectionError("That connection code is missing an access token.")

    return HelperConnection(endpoint=endpoint.rstrip("/"), token=token)


def verify_connection(connection: HelperConnection) -> None:
    """Calls the helper's /health endpoint. Raises HelperRequestError with
    a safe, user-facing message on any failure."""
    try:
        response = requests.get(f"{connection.endpoint}/health", timeout=HEALTH_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise HelperRequestError("Could not reach the helper. Check that it is running and try again.") from exc

    if response.status_code != 200:
        raise HelperRequestError("The helper did not respond as expected. Please reconnect.")
    try:
        body = response.json()
    except ValueError as exc:
        raise HelperRequestError("The helper returned an unexpected response.") from exc
    if body.get("status") != "ready":
        raise HelperRequestError("The helper is not ready yet. Please try again shortly.")


def _authorized_headers(connection: HelperConnection) -> dict:
    return {"Authorization": f"Bearer {connection.token}"}


def fetch_metadata_via_helper(connection: HelperConnection, url: str) -> HelperVideoMetadata:
    try:
        response = requests.post(
            f"{connection.endpoint}/metadata",
            json={"url": url},
            headers=_authorized_headers(connection),
            timeout=METADATA_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise HelperRequestError("Could not reach the helper. Check that it is running and try again.") from exc

    if response.status_code != 200:
        raise HelperRequestError(_safe_error_detail(response))

    body = response.json()
    return HelperVideoMetadata(
        platform=body.get("platform") or "Unknown",
        title=body.get("title") or "Untitled video",
        duration_seconds=body.get("duration_seconds"),
        uploader=body.get("uploader"),
        webpage_url=body.get("webpage_url") or url,
        thumbnail_url=body.get("thumbnail_url"),
    )


@dataclass(frozen=True)
class HelperPreparedAudio:
    wav_bytes: bytes
    sample_rate: int
    duration_seconds: float
    source_extension: str
    metadata: HelperVideoMetadata


def prepare_via_helper(connection: HelperConnection, url: str, start_seconds: float, window_seconds: float) -> HelperPreparedAudio:
    try:
        response = requests.post(
            f"{connection.endpoint}/prepare",
            json={"url": url, "interval_start": start_seconds, "interval_duration": window_seconds},
            headers=_authorized_headers(connection),
            timeout=PREPARE_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise HelperRequestError("Could not reach the helper. Check that it is running and try again.") from exc

    if response.status_code != 200:
        raise HelperRequestError(_safe_error_detail(response))

    body = response.json()
    try:
        wav_bytes = base64.b64decode(body["audio_base64"])
    except (KeyError, binascii.Error, ValueError) as exc:
        raise HelperRequestError("The helper returned an unexpected response.") from exc

    meta = body.get("metadata") or {}
    return HelperPreparedAudio(
        wav_bytes=wav_bytes,
        sample_rate=body.get("sample_rate", 16000),
        duration_seconds=body.get("duration_seconds", 0.0),
        source_extension=body.get("source_extension", ""),
        metadata=HelperVideoMetadata(
            platform=meta.get("platform") or "Unknown",
            title=meta.get("title") or "Untitled video",
            duration_seconds=meta.get("duration_seconds"),
            uploader=meta.get("uploader"),
            webpage_url=meta.get("webpage_url") or url,
            thumbnail_url=None,
        ),
    )


def _safe_error_detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    return "The helper could not complete this request."
