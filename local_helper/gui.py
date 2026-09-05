"""The helper's Windows GUI -- Tkinter (already part of a standard Python
install, no extra frontend framework or Electron). Deliberately shows no
technical logs in the main window (Part H of the architecture spec); an
optional developer log window is available but not required for normal
use.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

from local_helper.app import HelperSession
from local_helper.config import APP_NAME
from local_helper.tunnel import CloudflaredNotFoundError, TunnelStartError

_POLL_INTERVAL_MS = 1000


class HelperGUI:
    def __init__(self) -> None:
        self._session: HelperSession | None = None
        self._connection_code: str | None = None
        self._events: queue.Queue[tuple[str, str]] = queue.Queue()

        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.geometry("440x320")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_widgets()
        self.root.after(_POLL_INTERVAL_MS, self._poll)
        self._start_session_async()

    def _build_widgets(self) -> None:
        pad = {"padx": 16, "pady": 6}

        ttk.Label(self.root, text=APP_NAME, font=("Segoe UI", 14, "bold")).pack(**pad)

        self._status_var = tk.StringVar(value="Starting...")
        ttk.Label(self.root, textvariable=self._status_var, font=("Segoe UI", 11)).pack(**pad)

        self._tunnel_status_var = tk.StringVar(value="")
        self._local_status_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self._tunnel_status_var).pack()
        ttk.Label(self.root, textvariable=self._local_status_var).pack()

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=16)

        self._copy_button = ttk.Button(button_frame, text="Copy Connection", command=self._copy_connection, state="disabled")
        self._copy_button.grid(row=0, column=0, padx=6)

        self._restart_button = ttk.Button(button_frame, text="Restart Service", command=self._restart_session)
        self._restart_button.grid(row=0, column=1, padx=6)

        self._stop_button = ttk.Button(button_frame, text="Stop Service", command=self._stop_session)
        self._stop_button.grid(row=0, column=2, padx=6)

        ttk.Label(
            self.root,
            text="Keep this window open while using Video URL analysis.",
            font=("Segoe UI", 9),
            foreground="#555555",
        ).pack(pady=(12, 0))

        ttk.Label(self.root, text="Audio Deepfake Detector — Syed Raza Haider Zaidi", font=("Segoe UI", 8)).pack(
            side="bottom", pady=8
        )

    def _start_session_async(self) -> None:
        self._status_var.set("Starting...")
        self._copy_button.configure(state="disabled")

        def _run() -> None:
            self._events.put(("status", "Starting secure tunnel..."))
            try:
                session = HelperSession()
                code = session.start()
                self._events.put(("status", "Checking connection..."))
                self._session = session
                self._connection_code = code
                self._events.put(("ready", ""))
            except CloudflaredNotFoundError as exc:
                self._events.put(("error", str(exc)))
            except TunnelStartError as exc:
                self._events.put(("error", str(exc)))
            except Exception as exc:  # noqa: BLE001 -- must never crash the GUI thread
                self._events.put(("error", f"The helper could not start: {exc}"))

        threading.Thread(target=_run, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                kind, detail = self._events.get_nowait()
                if kind == "status":
                    self._status_var.set(detail)
                elif kind == "ready":
                    self._status_var.set("Ready")
                    self._tunnel_status_var.set("Secure URL Service: Connected")
                    self._local_status_var.set("Local Service: Running")
                    self._copy_button.configure(state="normal")
                elif kind == "error":
                    self._status_var.set("Disconnected")
                    self._tunnel_status_var.set(detail)
                    self._local_status_var.set("")
                    self._copy_button.configure(state="disabled")
        except queue.Empty:
            pass

        if self._session is not None and self._status_var.get() == "Ready" and not self._session.is_healthy():
            self._status_var.set("Disconnected")
            self._tunnel_status_var.set("Secure URL Service: Disconnected")
            self._local_status_var.set("")
            self._copy_button.configure(state="disabled")

        self.root.after(_POLL_INTERVAL_MS, self._poll)

    def _copy_connection(self) -> None:
        if not self._connection_code:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self._connection_code)
        self.root.update()

    def _restart_session(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None
        self._connection_code = None
        self._start_session_async()

    def _stop_session(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None
        self._connection_code = None
        self._status_var.set("Stopped")
        self._tunnel_status_var.set("")
        self._local_status_var.set("")
        self._copy_button.configure(state="disabled")

    def _on_close(self) -> None:
        if self._session is not None:
            self._session.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    HelperGUI().run()
