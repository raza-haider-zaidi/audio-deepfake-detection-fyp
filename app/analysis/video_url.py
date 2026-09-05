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
import shutil
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from app.analysis.media_ffmpeg import MediaDecodeError, decode_bytes_to_audio_sample
from audio_deepfake_detector.utils.datatypes import AudioSample

# Network safety limits. These bound worst-case resource usage on
# Streamlit Community Cloud -- see docs/input_sources.md, "Minimizing
# media transfer".
YTDLP_METADATA_TIMEOUT_SECONDS = 20
YTDLP_DOWNLOAD_TIMEOUT_SECONDS = 90
MAX_URL_DOWNLOAD_BYTES = 100 * 1024 * 1024  # safety ceiling, matches the video-file upload cap
MAX_SOURCE_DURATION_SECONDS = 4 * 60 * 60  # reject absurdly long sources outright

# Only these hosts are accepted. Other public video platforms are
# intentionally NOT claimed as supported in this phase -- see the module
# docstring and docs/input_sources.md for why.
ALLOWED_HOST_SUFFIXES = ("youtube.com", "youtube-nocookie.com", "youtu.be")

UNAVAILABLE_MESSAGE = (
    "Unable to access this video. The source could not be retrieved from the supplied "
    "public URL. You can download or export the audio/video yourself and use the "
    "file-analysis option instead."
)


class VideoURLError(ValueError):
    """Raised for any URL-ingestion failure. The message is always safe to
    show directly to a user -- never an extractor stack trace."""


class _SilentYDLLogger:
    """Routes yt-dlp's own internal logging to Python logging at debug
    level instead of stdout/stderr -- keeps Streamlit Cloud logs clean and
    avoids printing raw extractor error text (e.g. anti-bot messages)
    where it could be mistaken for an unhandled application error."""

    def debug(self, msg: str) -> None:
        logging.getLogger(__name__).debug(msg)

    def warning(self, msg: str) -> None:
        logging.getLogger(__name__).debug(msg)

    def error(self, msg: str) -> None:
        logging.getLogger(__name__).debug(msg)


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
        raise VideoURLError(UNAVAILABLE_MESSAGE) from exc

    if not info:
        raise VideoURLError(UNAVAILABLE_MESSAGE)

    extractor = (info.get("extractor_key") or info.get("extractor") or "").lower()
    if "youtube" not in extractor:
        raise VideoURLError(
            "Only YouTube links are supported for online video analysis right now."
        )

    duration = info.get("duration")
    if duration is not None and duration > MAX_SOURCE_DURATION_SECONDS:
        raise VideoURLError(
            "This video is too long to analyze via a public link. Please download the "
            "portion you need and use the file-analysis option instead."
        )

    formats = info.get("formats") or []
    has_audio = any(f.get("acodec") not in (None, "none") for f in formats) or info.get("acodec") not in (None, "none")
    if not has_audio:
        raise VideoURLError("This video does not appear to contain an audio track.")

    return VideoURLMetadata(
        platform="YouTube",
        title=info.get("title") or "Untitled video",
        duration_seconds=float(duration) if duration is not None else None,
        uploader=info.get("uploader"),
        video_id=info.get("id") or "",
        webpage_url=info.get("webpage_url") or url,
        thumbnail_url=info.get("thumbnail"),
    )


def extract_audio_interval(url: str, start_seconds: float, window_seconds: float) -> tuple[AudioSample, bytes, str]:
    """Retrieve and normalize ONLY the selected [start, start+window)
    interval of a public video's audio track -- never the whole source,
    and never an arbitrary/undisclosed section.

    Uses yt-dlp's `download_ranges` + `force_keyframes_at_cuts` with an
    audio-only format selection so, where the resolved format supports it
    (typically YouTube's DASH audio streams), only approximately the
    requested interval is actually transferred rather than the full
    source -- see docs/input_sources.md for the documented limitation on
    formats that do not support ranged/partial retrieval.

    Returns the normalized AudioSample, the raw extracted-audio bytes
    (hashed by the caller as the analyzed content's SHA-256 -- the URL
    itself is never used as a proxy for content identity), and the
    extracted audio container's file extension (for display only)."""
    validate_public_video_url(url)

    tmp_dir = Path(tempfile.mkdtemp(prefix="adf_url_"))
    outtmpl = str(tmp_dir / "audio.%(ext)s")
    end_seconds = start_seconds + window_seconds
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "download_ranges": yt_dlp.utils.download_range_func(None, [(max(0.0, start_seconds), end_seconds)]),
        "force_keyframes_at_cuts": True,
        "socket_timeout": YTDLP_METADATA_TIMEOUT_SECONDS,
        "max_filesize": MAX_URL_DOWNLOAD_BYTES,
        "logger": _SilentYDLLogger(),
    }
    try:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as exc:  # noqa: BLE001 -- never leak an extractor traceback to the UI
            raise VideoURLError(UNAVAILABLE_MESSAGE) from exc

        downloaded = sorted(tmp_dir.glob("audio.*"))
        if not downloaded:
            raise VideoURLError("No audio could be retrieved from this video.")

        raw_bytes = downloaded[0].read_bytes()
        if not raw_bytes:
            raise VideoURLError("No audio could be retrieved from this video.")

        source_extension = downloaded[0].suffix

        try:
            audio_sample = decode_bytes_to_audio_sample(raw_bytes, "online_video_audio", source_extension)
        except MediaDecodeError as exc:
            raise VideoURLError("The retrieved audio could not be decoded.") from exc

        if audio_sample.duration_seconds <= 0:
            raise VideoURLError("The selected interval produced no audio.")

        return audio_sample, raw_bytes, source_extension
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
