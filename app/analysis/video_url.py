"""Public video URL audio input adapter.

Lets a user paste a link to a PUBLICLY ACCESSIBLE YouTube video (standard
watch links, Shorts, youtu.be) and analyze a selected portion of its audio
track through the SAME frozen detector used for every other input source.

This is AUDIO-ONLY. No video frame is ever decoded or analyzed here or
anywhere else in this project -- only the audio track is retrieved and
extracted.

Scope note: only YouTube is currently supported. This project does not
claim support for other platforms it has not verified against the
installed yt-dlp build -- see docs/input_sources.md.

Reliability note: this module is a CONVENIENCE input adapter. It must
never become a dependency of normal detector operation -- Audio File,
Microphone, Voice Note, and Video File analysis do not import from this
module and continue to work unaffected if YouTube changes its delivery
behavior or yt-dlp needs an update.

Security note: this module accepts arbitrary user-entered URLs, so it
implements defense-in-depth SSRF protection -- a scheme allow-list
(http/https only), a hostname allow-list (YouTube domains only), and a
DNS/IP validation step that rejects loopback/private/link-local/multicast/
reserved destinations before any network request is made. All yt-dlp/
ffmpeg invocations pass arguments as explicit lists -- never `shell=True`,
never a string-concatenated command.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shutil
import socket
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from app.analysis.media_ffmpeg import MediaDecodeError, extract_video_audio_window
from audio_deepfake_detector.utils.datatypes import AudioSample

logger = logging.getLogger(__name__)

# Network safety limits. These bound worst-case resource usage on
# Streamlit Community Cloud -- see docs/input_sources.md, "Minimizing
# media transfer".
YTDLP_METADATA_TIMEOUT_SECONDS = 20
# The native downloader now retrieves the FULL audio-only stream (see
# "Native download / local interval extraction split" below), not just a
# short ranged slice, so this allows more time than a short clip would need.
YTDLP_DOWNLOAD_TIMEOUT_SECONDS = 180
MAX_URL_DOWNLOAD_BYTES = 100 * 1024 * 1024  # safety ceiling, matches the video-file upload cap
# Because the whole audio-only track is now downloaded before the selected
# interval is cut locally, the old 4-hour ceiling (safe under the previous
# ranged-remote-retrieval design) would let Streamlit Cloud download and
# hold a very large temporary file. 30 minutes of typical YouTube
# audio-only formats (~128 kbps) is well under MAX_URL_DOWNLOAD_BYTES and
# a practical ceiling for a "select a short interval to analyze" feature.
MAX_SOURCE_DURATION_SECONDS = 30 * 60

# Only these hosts are accepted. Other public video platforms are
# intentionally NOT claimed as supported in this phase -- see the module
# docstring and docs/input_sources.md for why.
ALLOWED_HOST_SUFFIXES = ("youtube.com", "youtube-nocookie.com", "youtu.be")

UNAVAILABLE_MESSAGE = (
    "Unable to access this video. The source could not be retrieved from the supplied "
    "public URL. You can download or export the audio/video yourself and use the "
    "file-analysis option instead."
)

# Internal failure categories -- never shown to the user directly, only used
# for diagnostics logging and to pick a slightly more specific (but still
# generic and traceback-free) user-facing message. See
# docs/input_sources.md, "Failure classification" for what triggers each one.
URL_INVALID = "URL_INVALID"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
NO_AUDIO_STREAM = "NO_AUDIO_STREAM"
YOUTUBE_BOT_CHALLENGE = "YOUTUBE_BOT_CHALLENGE"
JS_RUNTIME_UNAVAILABLE = "JS_RUNTIME_UNAVAILABLE"
PO_TOKEN_REQUIRED = "PO_TOKEN_REQUIRED"
FORMAT_UNAVAILABLE = "FORMAT_UNAVAILABLE"
NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
EXTRACTION_FAILED = "EXTRACTION_FAILED"
FFMPEG_FAILED = "FFMPEG_FAILED"
# Two distinct failure boundaries for the two-stage architecture (see
# "Native download / local interval extraction split" below):
# NATIVE_DOWNLOAD_FAILED is the fallback category for the yt-dlp native
# audio-only download stage (a 403/bot-challenge/PO-token/timeout/etc. is
# still classified more specifically by classify_extraction_error below --
# this is only the catch-all for that stage). FFMPEG_LOCAL_PROCESSING_FAILED
# is used ONLY when the native download already succeeded and a LOCAL
# ffmpeg call (against a local temp file, never a remote URL) then failed.
NATIVE_DOWNLOAD_FAILED = "NATIVE_DOWNLOAD_FAILED"
FFMPEG_LOCAL_PROCESSING_FAILED = "FFMPEG_LOCAL_PROCESSING_FAILED"

# Slightly more specific (but still traceback-free, non-alarming) messages
# for the failure categories a user can act on differently. Every other
# category falls back to UNAVAILABLE_MESSAGE -- the raw yt-dlp exception
# text is NEVER shown, only used internally to pick between these.
_CATEGORY_MESSAGES = {
    NO_AUDIO_STREAM: "No usable audio track was found in this video.",
    JS_RUNTIME_UNAVAILABLE: "Online video extraction is temporarily unavailable on this deployment.",
    YOUTUBE_BOT_CHALLENGE: "YouTube did not permit the application server to retrieve this video's audio.",
    PO_TOKEN_REQUIRED: "YouTube did not permit the application server to retrieve this video's audio.",
    FFMPEG_LOCAL_PROCESSING_FAILED: "The retrieved audio could not be processed for analysis.",
}


def classify_extraction_error(exc: BaseException, *, default: str = EXTRACTION_FAILED) -> str:
    """Best-effort classification of a raw yt-dlp exception into one of the
    internal failure categories above, using substring matching on the
    (never-shown-to-the-user) exception text. `default` is the category
    used when no specific pattern matches -- callers pass a stage-specific
    default (e.g. NATIVE_DOWNLOAD_FAILED for the download stage) so the
    fallback itself still identifies which stage failed. Used only for
    diagnostics and for choosing a slightly more specific safe message --
    classification mistakes are harmless since every category still maps
    to a safe, traceback-free message."""
    text = str(exc).lower()

    if "sign in to confirm" in text or "not a bot" in text or "confirm you" in text:
        return YOUTUBE_BOT_CHALLENGE
    if "po token" in text or "potoken" in text:
        return PO_TOKEN_REQUIRED
    if "no supported javascript runtime" in text or "js runtime" in text or "jsc" in text and "unavailable" in text:
        return JS_RUNTIME_UNAVAILABLE
    if "requested format is not available" in text or "format is not available" in text or "no video formats" in text:
        return FORMAT_UNAVAILABLE
    if "timed out" in text or "timeout" in text:
        return NETWORK_TIMEOUT
    if (
        "video is unavailable" in text
        or "video unavailable" in text
        or "private video" in text
        or "has been removed" in text
        or "does not exist" in text
        or "this video is not available" in text
        or ("age" in text and "restrict" in text)
    ):
        return SOURCE_UNAVAILABLE
    if "ffmpeg" in text and ("error" in text or "failed" in text):
        return FFMPEG_FAILED
    return default


def _safe_message(category: str) -> str:
    return _CATEGORY_MESSAGES.get(category, UNAVAILABLE_MESSAGE)


class VideoURLError(ValueError):
    """Raised for any URL-ingestion failure. The message is always safe to
    show directly to a user -- never an extractor stack trace. `category`
    is one of the internal failure-category constants above, for
    diagnostics/logging only."""

    def __init__(self, message: str, category: str = EXTRACTION_FAILED) -> None:
        super().__init__(message)
        self.category = category


# --------------------------------------------------------------------------
# Server-side diagnostic logging. Everything here is SERVER-LOG-ONLY: it is
# never rendered in the Streamlit UI, which continues to show only the safe
# messages above. See docs/input_sources.md, "Cloud diagnostics logging".
# --------------------------------------------------------------------------

DEBUG_ENV_VAR = "ADF_VIDEO_URL_DEBUG"


def is_debug_enabled() -> bool:
    """Gate for verbose yt-dlp diagnostic logging. Off unless an operator
    sets ADF_VIDEO_URL_DEBUG=1 in the deployment environment -- keeps
    per-request log volume small by default while allowing it to be
    switched on temporarily to diagnose a production failure."""
    return os.environ.get(DEBUG_ENV_VAR, "").strip().lower() in ("1", "true", "yes")


# Signed googlevideo.com media URLs, and any query parameter that looks
# like a token/signature/cookie/auth header, are redacted before anything
# reaches the log -- never cookies, auth headers, tokens, full signed
# media URLs, or Streamlit secrets. Full yt-dlp info dictionaries are never
# logged either; only short, explicitly-built diagnostic lines are.
_SENSITIVE_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_SENSITIVE_PARAM_RE = re.compile(r"(?i)\b(sig|signature|token|po_?token|auth\w*|cookie)=[^\s&\"'<>]+")


def _sanitize_log_text(text: str, *, max_length: int = 600) -> str:
    """Redact anything that looks like a signed media URL or an
    auth/token/cookie parameter from a piece of text before it is logged."""
    if not text:
        return text

    def _redact_url(match: "re.Match[str]") -> str:
        url = match.group(0)
        hostname = (urlparse(url).hostname or "").lower()
        if "googlevideo" in hostname or "youtube.com" in hostname and ("sig=" in url.lower() or "token" in url.lower()):
            return f"https://{hostname}/[signed-media-url-redacted]"
        return url  # plain, unsigned URLs (e.g. the public watch/shorts page URL) are fine to keep

    sanitized = _SENSITIVE_URL_RE.sub(_redact_url, text)
    sanitized = _SENSITIVE_PARAM_RE.sub(r"\1=[redacted]", sanitized)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "...[truncated]"
    return sanitized


def _sanitized_traceback(exc: BaseException, *, max_length: int = 2000) -> str:
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return _sanitize_log_text(formatted, max_length=max_length)


_VIDEO_ID_RE = re.compile(r"(?:[?&]v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{6,})")


def _video_id_for_logging(url: str) -> str:
    """Best-effort, no-network video-ID extraction for log lines -- lets
    server logs identify which video failed using platform + video ID
    instead of ever needing to log a full signed stream URL."""
    match = _VIDEO_ID_RE.search(url or "")
    return match.group(1) if match else "unknown"


def _log_extraction_failure(stage: str, exc: BaseException, category: str) -> None:
    """The key diagnostic line this module exists to produce: what failed,
    classified how, with the real (sanitized) exception -- server-log-only,
    never shown in the Streamlit UI. If `exc` carries ffmpeg stderr (see
    MediaDecodeError.stderr), it is included, sanitized, so a local ffmpeg
    failure is never just an opaque "ffmpeg exited with code N" with no
    context."""
    stderr = getattr(exc, "stderr", "") or ""
    logger.error(
        "VIDEO_URL_FAILURE stage=%s category=%s exception_type=%s sanitized_error=%s%s\n%s",
        stage,
        category,
        type(exc).__name__,
        _sanitize_log_text(str(exc)),
        f" ffmpeg_stderr={_sanitize_log_text(stderr)}" if stderr else "",
        _sanitized_traceback(exc),
    )


def _log_pre_extraction_diagnostic(url: str, start_seconds: float, end_seconds: float) -> None:
    """One concise, server-log-only diagnostic line emitted immediately
    before attempting real audio retrieval -- lets Streamlit Cloud logs
    show the exact runtime (yt-dlp/deno/ejs/ffmpeg versions) an extraction
    attempt ran under, without waiting for a failure to happen."""
    snapshot = diagnostics_snapshot()
    logger.info(
        "VIDEO_URL_DIAGNOSTIC: yt_dlp=%s deno=%s ejs=%s ffmpeg=%s platform=youtube video_id=%s requested_interval=%.0f-%.0f",
        snapshot["yt_dlp_version"],
        snapshot["js_runtime"] or "unavailable",
        snapshot["yt_dlp_ejs_version"] or "unavailable",
        "available" if snapshot["ffmpeg_path"] else "unavailable",
        _video_id_for_logging(url),
        start_seconds,
        end_seconds,
    )


class _SilentYDLLogger:
    """Routes yt-dlp's own internal logging to Python's logging system,
    sanitized.

    Previously `warning`/`error` were both routed to `.debug()`, which is
    why a real production failure (403, bot challenge, PO-token, missing
    JS runtime, format failure) never produced anything in Streamlit
    Cloud's server logs: DEBUG-level records are dropped by the default
    log level, silently swallowing legitimate WARNING/ERROR-level yt-dlp
    diagnostics along with the noisy ones. `warning`/`error` now use the
    matching real log level. Verbose DEBUG-level chatter (including
    yt-dlp's own internal JS-runtime/player-client detection lines) is
    only elevated to INFO when ADF_VIDEO_URL_DEBUG is enabled, so log
    volume stays small by default."""

    def debug(self, msg: str) -> None:
        sanitized = _sanitize_log_text(msg)
        if is_debug_enabled():
            logger.info(sanitized)
        else:
            logger.debug(sanitized)

    def warning(self, msg: str) -> None:
        logger.warning(_sanitize_log_text(msg))

    def error(self, msg: str) -> None:
        logger.error(_sanitize_log_text(msg))


def _hostname_allowed(hostname: str) -> bool:
    hostname = hostname.lower()
    return any(hostname == suffix or hostname.endswith("." + suffix) for suffix in ALLOWED_HOST_SUFFIXES)


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_video_url(url: str) -> str:
    """Scheme allow-list + hostname allow-list + DNS/IP SSRF protection.
    Returns the validated hostname. Raises VideoURLError otherwise. Called
    before ANY network request is made for a user-supplied URL."""
    url = (url or "").strip()
    if not url:
        raise VideoURLError("Please paste a video URL.")

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise VideoURLError("That doesn't look like a valid URL.") from exc

    if parsed.scheme not in ("http", "https"):
        raise VideoURLError("Only public http/https video links are supported.")

    hostname = parsed.hostname
    if not hostname:
        raise VideoURLError("That doesn't look like a valid URL.")

    if not _hostname_allowed(hostname):
        raise VideoURLError(
            "Only YouTube links (youtube.com, youtu.be, YouTube Shorts) are supported for "
            "online video analysis right now."
        )

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except OSError as exc:
        raise VideoURLError("This video's host could not be resolved.") from exc

    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if not _is_public_ip(ip):
            raise VideoURLError("This URL points to a non-public network address and cannot be processed.")

    return hostname


@dataclass
class VideoURLMetadata:
    platform: str
    title: str
    duration_seconds: float | None
    uploader: str | None
    video_id: str
    webpage_url: str
    thumbnail_url: str | None


def fetch_metadata(url: str) -> VideoURLMetadata:
    """Retrieve real title/duration/uploader/thumbnail metadata without
    downloading any media. Raises VideoURLError (never a raw yt-dlp
    exception/traceback) on any failure: unavailable, deleted, private,
    age-restricted, geo-restricted, anti-bot-blocked, or missing-audio
    videos are all reported with the same professional message."""
    validate_public_video_url(url)

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "socket_timeout": YTDLP_METADATA_TIMEOUT_SECONDS,
        "extractor_retries": 1,
        "logger": _SilentYDLLogger(),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001 -- never leak an extractor traceback to the UI
        category = classify_extraction_error(exc)
        _log_extraction_failure("metadata", exc, category)
        raise VideoURLError(_safe_message(category), category=category) from exc

    if not info:
        raise VideoURLError(UNAVAILABLE_MESSAGE, category=SOURCE_UNAVAILABLE)

    extractor = (info.get("extractor_key") or info.get("extractor") or "").lower()
    if "youtube" not in extractor:
        raise VideoURLError(
            "Only YouTube links are supported for online video analysis right now.",
            category=URL_INVALID,
        )

    duration = info.get("duration")
    if duration is not None and duration > MAX_SOURCE_DURATION_SECONDS:
        raise VideoURLError(
            "This video is too long to analyze via a public link. Please download the "
            "portion you need and use the file-analysis option instead.",
            category=SOURCE_UNAVAILABLE,
        )

    formats = info.get("formats") or []
    has_audio = any(f.get("acodec") not in (None, "none") for f in formats) or info.get("acodec") not in (None, "none")
    if not has_audio:
        raise VideoURLError(_safe_message(NO_AUDIO_STREAM), category=NO_AUDIO_STREAM)

    return VideoURLMetadata(
        platform="YouTube",
        title=info.get("title") or "Untitled video",
        duration_seconds=float(duration) if duration is not None else None,
        uploader=info.get("uploader"),
        video_id=info.get("id") or "",
        webpage_url=info.get("webpage_url") or url,
        thumbnail_url=info.get("thumbnail"),
    )


def extract_audio_interval(
    url: str,
    start_seconds: float,
    window_seconds: float,
    *,
    known_duration_seconds: float | None = None,
) -> tuple[AudioSample, bytes, str]:
    """Retrieve and normalize ONLY the selected [start, start+window)
    interval of a public video's audio track.

    Two-stage architecture (see docs/input_sources.md, "Native download /
    local interval extraction split" for why):

    STAGE A -- NATIVE DOWNLOAD: yt-dlp's own native downloader retrieves
    the full audio-only stream to a local temporary file. `download_ranges`
    / `force_keyframes_at_cuts` are deliberately NOT used here -- those
    options make yt-dlp invoke ffmpeg directly against the remote SIGNED
    media URL as an external downloader for the ranged retrieval, which
    worked locally but failed on Streamlit Cloud ("ffmpeg exited with code
    8"). Audio-only (`bestaudio/best`); the video stream is never
    downloaded.

    STAGE B -- LOCAL INTERVAL EXTRACTION: once the full audio track is a
    LOCAL file, `app.analysis.media_ffmpeg.extract_video_audio_window` (the
    same shared, unmodified local-file interval-extraction/normalization
    utility the Video File source already uses) cuts the selected
    [start, start+window) interval and normalizes it -- ffmpeg always
    receives a local filesystem path here, never a remote URL.

    `known_duration_seconds`, when the caller already has it from
    `fetch_metadata`, is checked before Stage A starts so an unexpectedly
    long source is rejected before downloading its full audio track.

    Returns the normalized AudioSample, the raw downloaded source-audio
    bytes (hashed by the caller as the analyzed content's SHA-256 -- the
    URL itself is never used as a proxy for content identity, consistent
    with how the Video File source hashes its full source file), and the
    downloaded audio container's file extension (for display only)."""
    validate_public_video_url(url)

    if known_duration_seconds is not None and known_duration_seconds > MAX_SOURCE_DURATION_SECONDS:
        raise VideoURLError(
            "This video is too long to analyze via a public link. Please download the "
            "portion you need and use the file-analysis option instead.",
            category=SOURCE_UNAVAILABLE,
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="adf_url_"))
    outtmpl = str(tmp_dir / "source_audio.%(ext)s")
    end_seconds = start_seconds + window_seconds
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Robust audio-only selection with a combined-stream fallback --
        # see docs/input_sources.md, "Format selection robustness". Never
        # forces a specific container (m4a/mp3/etc.); ffmpeg normalizes
        # afterward. Deliberately NO download_ranges/force_keyframes_at_cuts
        # -- see the docstring above and Stage B below.
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "socket_timeout": YTDLP_DOWNLOAD_TIMEOUT_SECONDS,
        "max_filesize": MAX_URL_DOWNLOAD_BYTES,
        "logger": _SilentYDLLogger(),
    }
    _log_pre_extraction_diagnostic(url, start_seconds, end_seconds)
    try:
        # --- Stage A: native download of the full audio-only stream ---
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as exc:  # noqa: BLE001 -- never leak an extractor traceback to the UI
            category = classify_extraction_error(exc, default=NATIVE_DOWNLOAD_FAILED)
            _log_extraction_failure("native_audio_download", exc, category)
            raise VideoURLError(_safe_message(category), category=category) from exc

        downloaded = sorted(tmp_dir.glob("source_audio.*"))
        if not downloaded:
            raise VideoURLError(_safe_message(NO_AUDIO_STREAM), category=NO_AUDIO_STREAM)

        source_path = downloaded[0]
        raw_bytes = source_path.read_bytes()
        if not raw_bytes:
            raise VideoURLError(_safe_message(NO_AUDIO_STREAM), category=NO_AUDIO_STREAM)

        source_extension = source_path.suffix

        # --- Stage B: local interval extraction + normalization ---
        # extract_video_audio_window writes `raw_bytes` back out to its OWN
        # local temp file internally and runs ffmpeg against that local
        # path -- ffmpeg never sees a remote URL in this function.
        try:
            audio_sample, _probed = extract_video_audio_window(
                raw_bytes, "online_video_audio", source_extension, start_seconds, window_seconds
            )
        except MediaDecodeError as exc:
            _log_extraction_failure("local_interval_extract", exc, FFMPEG_LOCAL_PROCESSING_FAILED)
            raise VideoURLError(
                _safe_message(FFMPEG_LOCAL_PROCESSING_FAILED), category=FFMPEG_LOCAL_PROCESSING_FAILED
            ) from exc

        if audio_sample.duration_seconds <= 0:
            raise VideoURLError(_safe_message(NO_AUDIO_STREAM), category=NO_AUDIO_STREAM)

        return audio_sample, raw_bytes, source_extension
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Developer-only diagnostics (never shown to normal users)
# --------------------------------------------------------------------------


def diagnostics_snapshot() -> dict:
    """Reports the current URL-ingestion environment: yt-dlp version,
    yt-dlp-ejs availability, ffmpeg availability, detected JS challenge
    runtime, and PO-token provider availability. Intended for an
    operator-only debug panel or deployment log line, never end-user UI."""
    snapshot: dict = {"yt_dlp_version": getattr(yt_dlp.version, "__version__", "unknown")}

    try:
        import yt_dlp_ejs

        snapshot["yt_dlp_ejs_version"] = getattr(yt_dlp_ejs, "__version__", "installed")
    except ImportError:
        snapshot["yt_dlp_ejs_version"] = None

    snapshot["ffmpeg_path"] = shutil.which("ffmpeg")

    js_runtime = None
    try:
        import deno as _deno_pkg

        deno_bin = _deno_pkg.find_deno_bin()
        if deno_bin:
            js_runtime = f"deno ({deno_bin})"
    except Exception:  # noqa: BLE001 -- diagnostics must never raise
        pass
    if js_runtime is None:
        js_runtime = shutil.which("deno") or shutil.which("node")
    snapshot["js_runtime"] = js_runtime

    try:
        from yt_dlp.extractor.youtube.jsc._registry import _jsc_providers  # type: ignore[attr-defined]

        snapshot["jsc_providers"] = sorted(_jsc_providers.value.keys())
    except Exception:  # noqa: BLE001 -- best-effort only, internal yt-dlp layout may change
        snapshot["jsc_providers"] = "unavailable (internal yt-dlp API not present in this version)"

    try:
        import bgutil_ytdlp_pot_provider  # noqa: F401

        snapshot["po_token_provider"] = "bgutil-ytdlp-pot-provider (installed)"
    except ImportError:
        snapshot["po_token_provider"] = None

    return snapshot
