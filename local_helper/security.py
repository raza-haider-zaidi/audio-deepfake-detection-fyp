"""Session token generation/verification and a tiny in-memory rate
limiter for the local helper's API.

The token is generated fresh every time the helper starts, lives only in
memory, is never written to disk or printed to any log, and is shown only
inside the GUI for the user to copy explicitly (see gui.py)."""

from __future__ import annotations

import hmac
import secrets
import time
from collections import deque


def generate_session_token() -> str:
    """A fresh, cryptographically random token for this helper session
    only -- changes every time the helper (re)starts."""
    return secrets.token_urlsafe(32)


def tokens_match(expected: str, provided: str) -> bool:
    """Constant-time comparison -- never a plain `==`, which leaks timing
    information about how many leading characters matched."""
    if not expected or not provided:
        return False
    return hmac.compare_digest(expected, provided)


class RateLimiter:
    """A minimal fixed-window limiter: at most `max_requests` calls to
    `allow()` succeed within any rolling `window_seconds` period. Enough
    to blunt casual abuse of a single-purpose, short-lived, low-traffic
    demo endpoint -- not a substitute for the token requirement, which is
    the real access control."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max_requests:
            return False
        self._timestamps.append(now)
        return True
