"""Orchestrates one helper session: generates the access token, starts
the local API (Uvicorn, in a background thread), starts the Cloudflare
Quick Tunnel, and builds the connection code the GUI displays. Owns clean
shutdown of both child processes/threads (Part I of the architecture
spec)."""

from __future__ import annotations

import base64
import json
import logging
import threading
import time

import uvicorn

from local_helper.api import create_app
from local_helper.config import API_VERSION, CONNECTION_CODE_PREFIX, LOCAL_HOST, LOCAL_PORT
from local_helper.security import generate_session_token
from local_helper.tunnel import QuickTunnel

logger = logging.getLogger("local_helper")


def build_connection_code(endpoint: str, token: str) -> str:
    payload = {"endpoint": endpoint, "token": token, "version": API_VERSION}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return f"{CONNECTION_CODE_PREFIX}{encoded}"


class _UvicornThread(threading.Thread):
    """Runs a Uvicorn server in-process on a background thread so the GUI
    stays responsive, with a clean `stop()` via uvicorn's own shutdown
    signal rather than killing the thread."""

    def __init__(self, app, host: str, port: int) -> None:
        super().__init__(daemon=True)
        # log_config=None: uvicorn's default logging setup attaches a
        # StreamHandler to sys.stdout, which is None in a PyInstaller
        # `--windowed` build (no console) -- that raised "Unable to
        # configure formatter 'default'" and prevented the server (and
        # therefore the whole helper) from starting. Our own
        # logging.basicConfig (see launcher.py, a real log FILE) remains
        # in effect for anything uvicorn logs through the standard
        # `logging` module.
        self._config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False, log_config=None)
        self._server = uvicorn.Server(self._config)

    def run(self) -> None:
        self._server.run()

    def stop(self, timeout: float = 5.0) -> None:
        self._server.should_exit = True
        self.join(timeout=timeout)

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                return True
            time.sleep(0.05)
        return self._server.started


class HelperSession:
    """One run of: token -> local API -> tunnel -> connection code. A new
    `HelperSession` is created on every start/restart so the token and
    tunnel hostname are always fresh (Part D: "the token should change
    every helper session")."""

    def __init__(self) -> None:
        self.token = generate_session_token()
        self._app = create_app(get_token=lambda: self.token)
        self._server_thread: _UvicornThread | None = None
        self._tunnel = QuickTunnel(LOCAL_PORT)
        self.public_url: str | None = None

    def start(self) -> str:
        self._server_thread = _UvicornThread(self._app, LOCAL_HOST, LOCAL_PORT)
        self._server_thread.start()
        if not self._server_thread.wait_until_ready():
            raise RuntimeError("The local service did not start in time.")

        self.public_url = self._tunnel.start()
        return build_connection_code(self.public_url, self.token)

    def is_healthy(self) -> bool:
        return (
            self._server_thread is not None
            and self._server_thread.is_alive()
            and self._tunnel.is_alive()
        )

    def stop(self) -> None:
        self._tunnel.stop()
        if self._server_thread is not None:
            self._server_thread.stop()
            self._server_thread = None
        self.public_url = None
