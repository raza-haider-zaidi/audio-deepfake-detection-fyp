"""Proof-of-Origin (PO) token provider integration for public video URL
audio ingestion.

Background: YouTube increasingly requires a GVS (Google Video Server) PO
Token for media (not page/metadata) requests from some clients/networks --
see https://github.com/yt-dlp/yt-dlp-wiki/blob/master/PO%20Token%20Guide.md
(current upstream guidance, verified against live sources, not memory).
Without one, the `mweb` client's media requests can return HTTP 403. This
module wires in the maintained bgutil-ytdlp-pot-provider
(https://github.com/Brainicism/bgutil-ytdlp-pot-provider), pinned at
release 1.3.2 (GPL-3.0-only -- see third_party/bgutil-ytdlp-pot-provider/
LICENSE), using its Deno "script mode" (no persistent server process, no
Docker sidecar -- appropriate for Streamlit Community Cloud, which
provides neither).

Policy (see docs/input_sources.md, "Proof-of-Origin token support" and
CLAUDE.md): PUBLIC content only. This module never touches cookies, a
Google account, OAuth, or a proxy -- only the anonymous, public PO-token
generation path that unauthenticated `mweb` playback already uses.

Deployment note: script mode requires the vendored server/ directory's
native `canvas` npm dependency to be installed once via `deno install`.
This is done LAZILY, at most once per process lifetime (see
`ensure_provider_ready`), not at Streamlit Cloud build time (Community
Cloud has no arbitrary build-script hook -- only requirements.txt and
packages.txt) and not on every request (would be far too slow and
violates the "avoid aggressive retries" resource-safety requirement).
Whether the underlying `canvas` install actually succeeds on Streamlit
Cloud's specific Linux container is UNVERIFIED from this sandbox -- see
docs/input_sources.md for the exact verification steps still required
after a real deployment.
"""

from __future__ import annotations

import functools
import importlib.util
import shutil
import subprocess
from pathlib import Path

PROVIDER_NAME = "bgutil-ytdlp-pot-provider"
PROVIDER_VERSION = "1.3.2"
PROVIDER_MODE = "script-deno"
PLAYER_CLIENT = "mweb"

# Derived from this file's location, never from `~` -- Streamlit Cloud's
# home directory assumptions are not something this project depends on.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_HOME = _REPO_ROOT / "third_party" / "bgutil-ytdlp-pot-provider" / "server"

# Native-dependency install (`canvas`, a BotGuard-interfacing native npm
# module) can be slow on a cold container, especially if no prebuilt
# binary matches the platform and it must compile from source.
DENO_INSTALL_TIMEOUT_SECONDS = 300


def _plugin_importable() -> bool:
    """Whether the pip-installed yt-dlp plugin side (`bgutil-ytdlp-pot-
    provider` on PyPI, resolvable as `yt_dlp_plugins.extractor.
    getpot_bgutil_script`) is present. This is the yt-dlp-facing half of
    the provider; the vendored server/ directory (SERVER_HOME) is the
    other half that actually generates tokens.

    Uses `importlib.util.find_spec` rather than an actual `import` --
    yt-dlp's own plugin loader is what is supposed to import and register
    this module exactly once per process; importing it ourselves first
    causes yt-dlp's own plugin discovery to attempt (and fail, harmlessly
    but noisily) a duplicate provider registration."""
    try:
        return importlib.util.find_spec("yt_dlp_plugins.extractor.getpot_bgutil_script") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _deno_bin() -> str | None:
    try:
        import deno as _deno_pkg

        found = _deno_pkg.find_deno_bin()
        if found:
            return found
    except Exception:  # noqa: BLE001 -- best-effort only
        pass
    return shutil.which("deno")


def _native_deps_installed() -> bool:
    return (SERVER_HOME / "node_modules").is_dir()


def provider_installed() -> bool:
    """Cheap, side-effect-free check: is the provider's yt-dlp-side plugin
    installed, the vendored server source present, and Deno available?
    Does NOT check whether the (potentially slow) native `canvas`
    dependency install has run -- see `ensure_provider_ready` for that.
    Safe to call on every request for diagnostics."""
    return SERVER_HOME.is_dir() and _plugin_importable() and _deno_bin() is not None


@functools.lru_cache(maxsize=1)
def ensure_provider_ready() -> tuple[bool, str]:
    """Idempotent (memoized for the life of this process) lazy setup of the
    vendored provider's native dependencies. Runs `deno install` at most
    ONCE per process -- Streamlit Cloud keeps the app alive as one
    long-running process, so this one-time cost is paid on first use (the
    first time a 403/bot-challenge/PO-token failure makes a provider-backed
    retry worthwhile), never at import time and never per-request.

    Returns (ready, sanitized_reason) -- `reason` is empty when ready,
    otherwise a short, non-sensitive description safe to log."""
    if not SERVER_HOME.is_dir():
        return False, "vendored provider server directory not found"
    if not _plugin_importable():
        return False, "yt-dlp plugin package not installed"
    deno_bin = _deno_bin()
    if deno_bin is None:
        return False, "deno runtime not available"

    if _native_deps_installed():
        return True, ""

    try:
        proc = subprocess.run(
            [deno_bin, "install", "--allow-scripts=npm:canvas,npm:@swc/core", "--frozen"],
            cwd=str(SERVER_HOME),
            capture_output=True,
            text=True,
            timeout=DENO_INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.SubprocessError:
        return False, "deno install did not complete (timed out or failed to start)"

    if proc.returncode != 0 or not _native_deps_installed():
        return False, "deno install exited without producing the expected dependencies"
    return True, ""


def provider_extractor_args() -> dict:
    """The yt-dlp `extractor_args` dict that selects the `mweb` client and
    points yt-dlp's bgutil script-mode PO-token provider at the vendored,
    pinned server/ directory (an absolute path, never `~`-relative)."""
    return {
        "youtube": {"player_client": [PLAYER_CLIENT]},
        "youtubepot-bgutilscript": {"server_home": [str(SERVER_HOME)]},
    }
