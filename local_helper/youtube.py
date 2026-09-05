"""Thin wrapper around the ALREADY TESTED public-video retrieval logic in
`app.analysis.video_url` -- the local helper does not reimplement yt-dlp
handling, SSRF-safe URL validation, or failure classification. It only
adds the helper's own (stricter or equal) resource limits on top.

Both `app.analysis.video_url` and `app.analysis.media_ffmpeg` have no
dependency on Streamlit, ONNX Runtime, torch, or any model code (verified
by inspection -- their only third-party imports are `yt_dlp`, `numpy`,
and `soundfile`), so importing them here does not pull ML dependencies
into the helper's PyInstaller bundle.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.analysis.video_url import (
    MAX_SOURCE_DURATION_SECONDS as _APP_MAX_SOURCE_DURATION_SECONDS,
)
from app.analysis.video_url import (
    VideoURLError,
    VideoURLMetadata,
)
from app.analysis.video_url import (
    extract_audio_interval as _extract_audio_interval,
)
from app.analysis.video_url import (
    fetch_metadata as _fetch_metadata,
)
from app.analysis.video_url import (
    validate_public_video_url as _validate_public_video_url,
)
from local_helper.config import MAX_INTERVAL_SECONDS, MAX_SOURCE_DURATION_SECONDS

# The helper never loosens the app's own SSRF-safe host allow-list
# (youtube.com/www.youtube.com/m.youtube.com/youtu.be/shorts) -- it is
# reused exactly as-is via _validate_public_video_url. The helper's
# duration cap below is the SAME as the app's (both 30 minutes); kept as
# two names so either can be tightened independently without surprising
# the other.
assert MAX_SOURCE_DURATION_SECONDS == _APP_MAX_SOURCE_DURATION_SECONDS


class HelperInputError(ValueError):
    """A validation failure caused by the caller's request (bad URL,
    interval out of range) -- distinct from VideoURLError (a retrieval
    failure), so the API layer can map each to the right HTTP status."""


@dataclass
class PreparedAudio:
    wav_bytes: bytes
    sample_rate: int
    duration_seconds: float
    source_extension: str
    metadata: VideoURLMetadata


def validate_and_fetch_metadata(url: str) -> VideoURLMetadata:
    """Raises HelperInputError for a rejected URL (bad scheme/host/SSRF --
    the caller's mistake, HTTP 422), VideoURLError for a genuine retrieval
    failure (HTTP 502) -- see app.analysis.video_url for what the latter
    covers. `validate_public_video_url`'s own VideoURLError does not
    distinguish these with a category (it predates this helper), so the
    distinction is made structurally here instead: anything raised by
    validation itself is a client input problem, never an upstream one."""
    try:
        _validate_public_video_url(url)
    except VideoURLError as exc:
        raise HelperInputError(str(exc)) from exc
    return _fetch_metadata(url)


def prepare_interval(url: str, interval_start: float, interval_duration: float) -> PreparedAudio:
    """Validates the requested interval against the helper's own limits,
    then delegates the actual retrieval + local ffmpeg extraction to
    app.analysis.video_url.extract_audio_interval (Stage A native
    download, Stage B local-file-only ffmpeg -- unchanged, see that
    module's docstring)."""
    if interval_start < 0:
        raise HelperInputError("interval_start must not be negative.")
    if not (0 < interval_duration <= MAX_INTERVAL_SECONDS):
        raise HelperInputError(f"interval_duration must be between 0 and {MAX_INTERVAL_SECONDS:.0f} seconds.")

    metadata = validate_and_fetch_metadata(url)
    if metadata.duration_seconds and metadata.duration_seconds > MAX_SOURCE_DURATION_SECONDS:
        raise HelperInputError("This video exceeds the helper's maximum supported duration.")

    audio_sample, raw_bytes, source_extension = _extract_audio_interval(
        url, interval_start, interval_duration, known_duration_seconds=metadata.duration_seconds
    )

    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio_sample.waveform, audio_sample.sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = buffer.getvalue()

    return PreparedAudio(
        wav_bytes=wav_bytes,
        sample_rate=audio_sample.sample_rate,
        duration_seconds=audio_sample.duration_seconds,
        source_extension=source_extension,
        metadata=metadata,
    )


__all__ = [
    "HelperInputError",
    "PreparedAudio",
    "VideoURLError",
    "VideoURLMetadata",
    "prepare_interval",
    "validate_and_fetch_metadata",
]
