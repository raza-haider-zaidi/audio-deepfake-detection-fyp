"""Unified input-adapter architecture.

Every supported input source (uploaded audio file, microphone capture,
voice note, video audio track) is normalized here into ONE
`NormalizedAudioInput`, carrying an `AudioSample` produced by the SAME
frozen decode/normalize pipeline
(`audio_deepfake_detector.preprocessing.audio_loader`) regardless of
source. Everything downstream -- inference, segment evidence, audio
diagnostics, session history, reporting -- operates on this one type and
never needs to know which adapter produced it, so there is exactly one
analysis code path, not four.

No adapter here performs classification, denoising, enhancement, speaker
ID, or any other model inference -- decoding/resampling/interval
extraction only. See app/analysis/media_ffmpeg.py for the ffmpeg-backed
decode step used by the voice-note and video adapters.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.analysis.media_ffmpeg import MediaDecodeError, ProbedMedia, extract_video_audio_window, probe_file
from app.errors import UserFacingError
from app.validation import (
    MAX_VIDEO_ANALYSIS_WINDOW_SECONDS,
    validate_and_load_microphone,
    validate_and_load_upload,
    validate_and_load_voice_note,
    validate_video_extension,
    validate_video_upload_size,
)
from audio_deepfake_detector.preprocessing.audio_loader import load_audio_file
from audio_deepfake_detector.utils.datatypes import AudioSample

if TYPE_CHECKING:
    # app.analysis.video_url imports yt_dlp, which is a HELPER-only
    # dependency (see requirements-helper.txt / docs/local_url_helper.md)
    # -- NOT installed in the deployed Streamlit environment. The
    # functions below that need it import it lazily, at call time, so
    # importing this module never requires yt_dlp to be installed. This
    # TYPE_CHECKING import is erased at runtime and only satisfies type
    # checkers/annotations (postponed by `from __future__ import
    # annotations` above).
    from app.analysis.video_url import VideoURLMetadata

SOURCE_AUDIO_FILE = "audio_file"
SOURCE_MICROPHONE = "microphone"
SOURCE_VOICE_NOTE = "voice_note"
SOURCE_VIDEO_AUDIO = "video_audio"
SOURCE_VIDEO_URL = "video_url"

SOURCE_LABELS = {
    SOURCE_AUDIO_FILE: "Audio File",
    SOURCE_MICROPHONE: "Microphone Capture",
    SOURCE_VOICE_NOTE: "Voice Note",
    SOURCE_VIDEO_AUDIO: "Video Audio",
    SOURCE_VIDEO_URL: "Online Video",
}


@dataclass
class NormalizedAudioInput:
    """The single representation every downstream analysis step consumes,
    regardless of which adapter produced it."""

    source_type: str
    display_filename: str
    audio_sample: AudioSample
    sha256: str
    file_size_bytes: int
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source_type, self.source_type)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def from_uploaded_file(data: bytes, filename: str) -> NormalizedAudioInput:
    audio_sample = validate_and_load_upload(data, filename)
    return NormalizedAudioInput(
        source_type=SOURCE_AUDIO_FILE,
        display_filename=filename,
        audio_sample=audio_sample,
        sha256=_sha256(data),
        file_size_bytes=len(data),
        source_metadata={"format": Path(filename).suffix.lstrip(".").upper()},
    )


def from_microphone(data: bytes) -> NormalizedAudioInput:
    audio_sample = validate_and_load_microphone(data)
    return NormalizedAudioInput(
        source_type=SOURCE_MICROPHONE,
        display_filename="Microphone recording.wav",
        audio_sample=audio_sample,
        sha256=_sha256(data),
        file_size_bytes=len(data),
        source_metadata={"capture_type": "Live microphone recording", "requested_sample_rate_hz": 16000},
    )


def from_voice_note(data: bytes, filename: str) -> NormalizedAudioInput:
    audio_sample = validate_and_load_voice_note(data, filename)
    return NormalizedAudioInput(
        source_type=SOURCE_VOICE_NOTE,
        display_filename=filename,
        audio_sample=audio_sample,
        sha256=_sha256(data),
        file_size_bytes=len(data),
        source_metadata={"format": Path(filename).suffix.lstrip(".").upper(), "normalized_to": "Mono · 16 kHz"},
    )


def probe_video(data: bytes, filename: str) -> ProbedMedia:
    """Validate size/extension and probe a video's real container/codec/
    duration via ffprobe -- used before the user selects an analysis
    interval. Raises UserFacingError on any failure (not the raw
    MediaDecodeError), since this is called directly from the view."""
    validate_video_upload_size(data)
    validate_video_extension(filename)

    import tempfile

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        probed = probe_file(tmp_path)
    except MediaDecodeError as exc:
        raise UserFacingError(
            "This video could not be inspected. It may be corrupted or use an unsupported container.",
            technical_detail=str(exc),
        ) from exc
    finally:
        import os

        os.unlink(tmp_path)

    if not probed.has_audio_stream:
        raise UserFacingError("This video does not contain an audio track, so it cannot be analyzed.")
    return probed


def from_video(
    data: bytes,
    filename: str,
    *,
    start_seconds: float,
    window_seconds: float = MAX_VIDEO_ANALYSIS_WINDOW_SECONDS,
) -> NormalizedAudioInput:
    """Extract and normalize ONLY the selected [start, start+window)
    interval of a video's audio track -- never the whole file, and never
    an arbitrary/undisclosed section. Visual frames are never decoded or
    analyzed anywhere in this path."""
    validate_video_upload_size(data)
    validate_video_extension(filename)

    try:
        audio_sample, probed = extract_video_audio_window(data, filename, Path(filename).suffix, start_seconds, window_seconds)
    except MediaDecodeError as exc:
        raise UserFacingError(
            "This video's audio track could not be extracted for the selected interval.", technical_detail=str(exc)
        ) from exc

    end_seconds = start_seconds + audio_sample.duration_seconds
    return NormalizedAudioInput(
        source_type=SOURCE_VIDEO_AUDIO,
        display_filename=filename,
        audio_sample=audio_sample,
        sha256=_sha256(data),
        file_size_bytes=len(data),
        source_metadata={
            "video_filename": filename,
            "container": probed.container,
            "video_codec": probed.video_codec or "Not available",
            "audio_codec": probed.audio_codec,
            "video_duration_seconds": probed.duration_seconds,
            "selected_interval": f"{start_seconds:.0f}s–{end_seconds:.0f}s",
        },
    )


def fetch_video_url_metadata(url: str) -> "VideoURLMetadata":
    """LOCAL-DEVELOPMENT / TESTING ONLY -- direct yt-dlp metadata fetch.
    The deployed Streamlit application no longer calls this from the UI
    (see app/views/analyze.py and app/analysis/url_helper_client.py,
    which talk to the local URL ingestion helper instead); this remains
    for local dev/testing and is what the helper itself calls internally
    (via app.analysis.video_url directly, not through this wrapper).
    Imports app.analysis.video_url lazily so importing this module never
    requires yt-dlp (a helper-only dependency -- see
    requirements-helper.txt) to be installed.

    Validates a public video URL and retrieves its real title/duration/
    uploader/thumbnail metadata -- no media is downloaded. Raises
    UserFacingError (never a raw extractor exception) on any failure:
    invalid/unsupported URL, non-public network address, unavailable/
    private/age-restricted/geo-restricted video, extraction failure, or
    missing audio track."""
    from app.analysis.video_url import VideoURLError
    from app.analysis.video_url import fetch_metadata as _fetch_url_metadata

    try:
        return _fetch_url_metadata(url)
    except VideoURLError as exc:
        raise UserFacingError(str(exc)) from exc


def from_video_url(
    url: str,
    metadata: "VideoURLMetadata",
    *,
    start_seconds: float,
    window_seconds: float = MAX_VIDEO_ANALYSIS_WINDOW_SECONDS,
) -> NormalizedAudioInput:
    """LOCAL-DEVELOPMENT / TESTING ONLY -- see fetch_video_url_metadata's
    docstring above; the deployed application uses
    from_helper_prepared_audio (below) instead.

    Retrieve and normalize ONLY the selected [start, start+window)
    interval of a public video's audio track -- never the whole source.
    Visual frames are never retrieved or analyzed. The SHA-256 recorded on
    the returned input is computed from the actual extracted audio bytes,
    never from the URL itself."""
    from app.analysis.video_url import VideoURLError
    from app.analysis.video_url import extract_audio_interval as _extract_url_audio_interval

    try:
        audio_sample, raw_bytes, source_extension = _extract_url_audio_interval(
            url, start_seconds, window_seconds, known_duration_seconds=metadata.duration_seconds
        )
    except VideoURLError as exc:
        raise UserFacingError(str(exc)) from exc

    end_seconds = start_seconds + audio_sample.duration_seconds
    return NormalizedAudioInput(
        source_type=SOURCE_VIDEO_URL,
        display_filename=metadata.title or "Online video",
        audio_sample=audio_sample,
        sha256=_sha256(raw_bytes),
        file_size_bytes=len(raw_bytes),
        source_metadata={
            "format": source_extension.lstrip(".").upper() or "Not available",
            "platform": metadata.platform,
            "source_title": metadata.title,
            "source_url": metadata.webpage_url,
            "source_uploader": metadata.uploader or "Not available",
            "video_duration_seconds": metadata.duration_seconds,
            "selected_interval": f"{start_seconds:.0f}s–{end_seconds:.0f}s",
        },
    )


def from_helper_prepared_audio(
    wav_bytes: bytes,
    *,
    display_filename: str,
    start_seconds: float,
    window_seconds: float,
    platform: str,
    source_title: str,
    source_url: str,
    source_uploader: str | None,
    video_duration_seconds: float | None,
    source_format: str,
) -> NormalizedAudioInput:
    """Build a NormalizedAudioInput from audio already retrieved and
    normalized by the LOCAL URL INGESTION HELPER (see
    app/analysis/url_helper_client.py and docs/local_url_helper.md) --
    this is what the deployed Streamlit application actually uses for the
    Video URL source; it never runs yt-dlp itself.

    `wav_bytes` is the exact normalized 16 kHz mono WAV the helper
    produced via the same `extract_video_audio_window`/media_ffmpeg
    pipeline every other video-audio source already uses -- this function
    only decodes it through the same frozen `load_audio_file` path every
    other adapter uses, never re-touching ffmpeg or yt-dlp. The SHA-256
    recorded is over these exact bytes -- the actual content that was
    analyzed and that crossed the network from the helper to this
    application (see docs/local_url_helper.md, "Privacy" for why this is
    described accurately as data transfer, not "never leaves your
    device")."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = Path(tmp.name)
    try:
        sample = load_audio_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    audio_sample = AudioSample(
        waveform=sample.waveform, sample_rate=sample.sample_rate, duration_seconds=sample.duration_seconds, source_name=display_filename
    )
    end_seconds = start_seconds + audio_sample.duration_seconds
    return NormalizedAudioInput(
        source_type=SOURCE_VIDEO_URL,
        display_filename=display_filename,
        audio_sample=audio_sample,
        sha256=_sha256(wav_bytes),
        file_size_bytes=len(wav_bytes),
        source_metadata={
            "format": source_format.lstrip(".").upper() or "Not available",
            "platform": platform,
            "source_title": source_title,
            "source_url": source_url,
            "source_uploader": source_uploader or "Not available",
            "video_duration_seconds": video_duration_seconds,
            "selected_interval": f"{start_seconds:.0f}s–{end_seconds:.0f}s",
        },
    )
