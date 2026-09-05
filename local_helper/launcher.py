"""Entry point for the local helper -- what `Raza Audio URL Helper.exe`
actually runs (see scripts/build_local_helper.ps1). Also runnable directly
during development: `python -m local_helper.launcher`.

Logging is configured to a rotating file under the user's local app-data
directory rather than a console, since the packaged executable has no
attached console window -- and, per the security requirements, the
per-session access token is never written to this (or any other) log.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _prepend_bundled_tools_to_path() -> None:
    """If a `tools/` directory sits next to this executable (or, in
    development, next to the repository root), put it at the FRONT of
    PATH so `shutil.which("cloudflared"/"ffmpeg"/"ffprobe")` -- used
    unchanged by tunnel.py and media_ffmpeg.py -- finds the bundled copies
    first, with no code changes needed in either module. Falls back to
    whatever is already on the system PATH if `tools/` is absent (e.g. a
    developer already has ffmpeg installed, as this development machine
    does)."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent.parent
    tools_dir = base / "tools"
    if tools_dir.is_dir():
        os.environ["PATH"] = str(tools_dir) + os.pathsep + os.environ.get("PATH", "")


def _configure_logging() -> None:
    log_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AudioDeepfakeURLHelper"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / "helper.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    _prepend_bundled_tools_to_path()
    _configure_logging()
    from local_helper.gui import main as run_gui

    run_gui()


if __name__ == "__main__":
    main()
