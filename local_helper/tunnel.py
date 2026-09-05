"""Cloudflare Quick Tunnel management.

Uses `cloudflared tunnel --url http://127.0.0.1:<port>` to expose the
local API over a temporary, randomly-named `https://<random>.
trycloudflare.com` address -- no account, no domain, no persistent public
server, and the tunnel only exists while this process runs (see
docs/local_url_helper.md).

IMPORTANT (verified by inspecting this development machine before writing
this module): `cloudflared` was NOT found installed or on PATH here. This
module locates it at runtime in this order: (1) a `tools/cloudflared.exe`
directory shipped alongside the packaged executable, (2) the system PATH.
If neither is present, `start_tunnel` raises `CloudflaredNotFoundError`
with an actionable message -- `scripts/build_local_helper.ps1` is
responsible for obtaining a pinned `cloudflared.exe` release and placing
it in `tools/` at build time (see that script and docs/
local_url_helper.md for exact steps); this module never downloads
anything itself at runtime.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

TRYCLOUDFLARE_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
STARTUP_TIMEOUT_SECONDS = 30


class CloudflaredNotFoundError(RuntimeError):
    pass


class TunnelStartError(RuntimeError):
    pass


def _bundled_cloudflared_path() -> Path | None:
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        # Development mode: scripts/build_local_helper.ps1 places tools/
        # directly under local_helper/ (this file's own directory), NOT
        # the repository root.
        base = Path(__file__).resolve().parent
    candidate = base / "tools" / "cloudflared.exe"
    return candidate if candidate.is_file() else None


def locate_cloudflared() -> str:
    bundled = _bundled_cloudflared_path()
    if bundled is not None:
        return str(bundled)
    on_path = shutil.which("cloudflared")
    if on_path:
        return on_path
    raise CloudflaredNotFoundError(
        "cloudflared.exe was not found next to this application (tools/cloudflared.exe) "
        "or on the system PATH. Run scripts/build_local_helper.ps1 to obtain it, or place "
        "a cloudflared.exe from https://github.com/cloudflare/cloudflared/releases into "
        "the tools/ folder next to this executable."
    )


class QuickTunnel:
    """Owns one `cloudflared` child process for the life of the helper
    session. `public_url` is set once cloudflared reports its assigned
    trycloudflare.com hostname; `is_alive()` reflects the child process's
    actual state so a crashed tunnel is detected, not assumed healthy."""

    def __init__(self, local_port: int) -> None:
        self._local_port = local_port
        self._process: subprocess.Popen | None = None
        self._public_url: str | None = None
        self._ready_event = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._lines: list[str] = []

    @property
    def public_url(self) -> str | None:
        return self._public_url

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> str:
        cloudflared = locate_cloudflared()
        # No shell=True, explicit argument list only -- see security
        # requirements in docs/local_url_helper.md.
        self._process = subprocess.Popen(
            [
                cloudflared,
                "tunnel",
                "--url",
                f"http://127.0.0.1:{self._local_port}",
                "--no-autoupdate",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

        if not self._ready_event.wait(timeout=STARTUP_TIMEOUT_SECONDS):
            self.stop()
            raise TunnelStartError(
                "cloudflared did not report a public URL within the expected time. "
                "Check your internet connection and try again."
            )
        return self._public_url  # type: ignore[return-value]

    def _read_output(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.append(line)
            if self._public_url is None:
                match = TRYCLOUDFLARE_URL_RE.search(line)
                if match:
                    self._public_url = match.group(0)
                    self._ready_event.set()

    def stop(self, timeout: float = 5.0) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=timeout)
        self._process = None
        self._public_url = None
        self._ready_event.clear()

    def recent_output(self, max_lines: int = 40) -> list[str]:
        """For an optional developer log view only -- never shown in the
        main GUI (see gui.py)."""
        return self._lines[-max_lines:]
