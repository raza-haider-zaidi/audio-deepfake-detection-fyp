"""Fixed configuration for the local helper. No secrets or environment-
specific values live here -- the per-session access token is generated at
runtime (see security.py) and never stored on disk."""

from __future__ import annotations

APP_NAME = "Audio Deepfake URL Helper"
SERVICE_ID = "audio-url-helper"
API_VERSION = 1

# Bound to loopback only -- the Cloudflare tunnel (tunnel.py) is what
# exposes this publicly, never the process itself (see docs/
# local_url_helper.md, "No localhost from Streamlit").
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8765

# Resource limits (Part E of the architecture spec) -- kept conservative
# for an ordinary laptop running during a live demonstration.
MAX_SOURCE_DURATION_SECONDS = 30 * 60
MAX_INTERVAL_SECONDS = 30.0
MAX_CONCURRENT_JOBS = 1
PREPARE_TIMEOUT_SECONDS = 180
MAX_REQUEST_BODY_BYTES = 8 * 1024  # /prepare and /metadata bodies are tiny JSON, never media
RATE_LIMIT_WINDOW_SECONDS = 10.0
RATE_LIMIT_MAX_REQUESTS = 5

# Only these hosts are ever contacted -- see local_helper/youtube.py,
# which delegates the actual check to the same allow-list
# app.analysis.video_url already enforces (never duplicated/loosened
# here).
CONNECTION_CODE_PREFIX = "ADA1:"
