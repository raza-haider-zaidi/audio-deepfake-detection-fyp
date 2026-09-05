"""The local helper's HTTP API -- exposed ONLY through the Cloudflare
tunnel (tunnel.py), never directly reachable by anything other than
whatever can reach this machine's loopback interface plus the tunnel
itself (see docs/local_url_helper.md, "No localhost from Streamlit").

Endpoints:
  GET  /health    -- minimal, unauthenticated liveness/version check.
  POST /metadata  -- authenticated; title/duration/uploader for a URL.
  POST /prepare   -- authenticated; retrieves + extracts + normalizes the
                      requested interval, returns it as base64 WAV bytes.

Security (Part D of the architecture spec): bearer-token auth on every
endpoint except /health, constant-time token comparison, a request-size
cap, a per-process rate limiter, a hard cap of one concurrent preparation
job, a wall-clock timeout on that job, and strict URL validation delegated
to the already-tested app.analysis.video_url allow-list. No `shell=True`
anywhere in this codebase; every subprocess call uses an explicit argument
list (see app/analysis/media_ffmpeg.py and tunnel.py).
"""

from __future__ import annotations

import asyncio
import base64
import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from local_helper.config import (
    API_VERSION,
    MAX_REQUEST_BODY_BYTES,
    PREPARE_TIMEOUT_SECONDS,
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    SERVICE_ID,
)
from local_helper.security import RateLimiter, tokens_match
from local_helper.youtube import HelperInputError, PreparedAudio, VideoURLError, prepare_interval, validate_and_fetch_metadata

logger = logging.getLogger("local_helper")


class PrepareRequest(BaseModel):
    url: str
    interval_start: float = Field(ge=0)
    interval_duration: float = Field(gt=0)


class MetadataRequest(BaseModel):
    url: str


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Rejects any request whose declared (or actual) body exceeds the
    configured limit -- every request this API accepts is a tiny JSON
    object, never media, so this is a generous but real ceiling."""

    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self._max_bytes:
                    return JSONResponse({"detail": "Request body too large."}, status_code=413)
            except ValueError:
                pass
        return await call_next(request)


def create_app(*, get_token: callable) -> FastAPI:
    """`get_token` is a zero-argument callable returning the CURRENT
    session token -- passed in rather than imported as a module-level
    constant so app.py can rotate it if the helper is restarted without
    reloading this module."""
    app = FastAPI(title=SERVICE_ID, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)

    rate_limiter = RateLimiter(max_requests=RATE_LIMIT_MAX_REQUESTS, window_seconds=RATE_LIMIT_WINDOW_SECONDS)
    job_lock = asyncio.Lock()

    def _require_auth(authorization: str | None) -> None:
        if not rate_limiter.allow():
            raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment and try again.")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
        provided = authorization[len("Bearer ") :]
        if not tokens_match(get_token(), provided):
            raise HTTPException(status_code=401, detail="Invalid access token.")

    @app.get("/health")
    async def health():
        # Deliberately minimal -- no video/job/system detail, per Part D
        # ("health may expose only minimal non-sensitive status").
        return {"status": "ready", "service": SERVICE_ID, "version": API_VERSION}

    @app.post("/metadata")
    async def metadata(payload: MetadataRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        try:
            meta = await asyncio.to_thread(validate_and_fetch_metadata, payload.url)
        except HelperInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except VideoURLError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "platform": meta.platform,
            "title": meta.title,
            "duration_seconds": meta.duration_seconds,
            "uploader": meta.uploader,
            "video_id": meta.video_id,
            "webpage_url": meta.webpage_url,
            "thumbnail_url": meta.thumbnail_url,
        }

    @app.post("/prepare")
    async def prepare(payload: PrepareRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        if job_lock.locked():
            raise HTTPException(status_code=429, detail="Another video is already being prepared. Please wait for it to finish.")

        async with job_lock:
            try:
                prepared: PreparedAudio = await asyncio.wait_for(
                    asyncio.to_thread(prepare_interval, payload.url, payload.interval_start, payload.interval_duration),
                    timeout=PREPARE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise HTTPException(status_code=504, detail="Retrieving this video took too long. Please try again.") from exc
            except HelperInputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except VideoURLError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        return {
            "sample_rate": prepared.sample_rate,
            "duration_seconds": prepared.duration_seconds,
            "source_extension": prepared.source_extension,
            "audio_base64": base64.b64encode(prepared.wav_bytes).decode("ascii"),
            "metadata": {
                "platform": prepared.metadata.platform,
                "title": prepared.metadata.title,
                "duration_seconds": prepared.metadata.duration_seconds,
                "uploader": prepared.metadata.uploader,
                "webpage_url": prepared.metadata.webpage_url,
            },
        }

    return app


__all__ = ["MaxBodySizeMiddleware", "MetadataRequest", "PrepareRequest", "create_app"]
