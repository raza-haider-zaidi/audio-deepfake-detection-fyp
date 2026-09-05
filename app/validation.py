"""Upload validation for the Streamlit layer.

Delegates actual audio decoding to the shared preprocessing pipeline in
src/audio_deepfake_detector — this module only adds Streamlit-facing
constraints (max duration, max file size) and turns failures into
UserFacingError with a friendly message.
"""

from __future__ import annotations

from pathlib import Path

from app.analysis.media_ffmpeg import MediaDecodeError, decode_bytes_to_audio_sample
from app.errors import UserFacingError
from audio_deepfake_detector.preprocessing.audio_loader import (
    SUPPORTED_EXTENSIONS,
    AudioLoadError,
    load_audio_bytes,
)
from audio_deepfake_detector.utils.datatypes import AudioSample

MAX_DURATION_SECONDS = 30.0
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB -- audio files, voice notes, microphone captures

# Voice notes: container formats ffmpeg reliably decodes in this project's
# deployment (verified against the installed ffmpeg build), on top of the
# natively-supported WAV/MP3/FLAC. AMR is intentionally NOT listed --
# decoding reliability was not verified in this environment, and this
# project does not label unsupported/unverified formats as supported.
SUPPORTED_VOICE_NOTE_FFMPEG_EXTENSIONS = {".m4a", ".aac", ".ogg", ".opus", ".webm"}
SUPPORTED_VOICE_NOTE_EXTENSIONS = SUPPORTED_EXTENSIONS | SUPPORTED_VOICE_NOTE_FFMPEG_EXTENSIONS

# Video: containers ffmpeg reliably demuxes for this project's use case
# (audio-track extraction only -- no video frame analysis, see
# docs/input_sources.md).
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
MAX_VIDEO_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB -- higher than audio, video containers are larger
MAX_VIDEO_ANALYSIS_WINDOW_SECONDS = MAX_DURATION_SECONDS  # same 30s cap the detector is evaluated against


def validate_and_load_upload(
    data: bytes,
    filename: str,
    max_duration_seconds: float = MAX_DURATION_SECONDS,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> AudioSample:
    """Validate an uploaded file's size/format/duration and return a decoded
    AudioSample, or raise UserFacingError with a short, friendly message."""
    if len(data) > max_file_size_bytes:
        max_mb = max_file_size_bytes / (1024 * 1024)
        raise UserFacingError(
            f"That file is too large ({len(data) / (1024 * 1024):.1f} MB). "
            f"Please upload a file under {max_mb:.0f} MB."
        )

    try:
        audio_sample = load_audio_bytes(data, filename)
    except AudioLoadError as exc:
        raise UserFacingError(
            "This file couldn't be read as audio. Please upload a valid "
            f"{', '.join(sorted(ext.lstrip('.').upper() for ext in SUPPORTED_EXTENSIONS))} file.",
            technical_detail=str(exc),
        ) from exc

    if audio_sample.duration_seconds > max_duration_seconds:
        raise UserFacingError(
            f"This clip is {audio_sample.duration_seconds:.1f} seconds long. "
            f"Please upload a clip of {max_duration_seconds:.0f} seconds or less."
        )

    return audio_sample


def validate_and_load_voice_note(
    data: bytes,
    filename: str,
    max_duration_seconds: float = MAX_DURATION_SECONDS,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> AudioSample:
    """Validate and decode a voice note. WAV/MP3/FLAC go through the exact
    same native loader as a regular audio upload; M4A/AAC/OGG/OPUS/WEBM go
    through the ffmpeg input adapter, then converge on the SAME finalize
    path (see app/analysis/media_ffmpeg.py). Duration/size limits are
    identical to a regular audio upload."""
    if len(data) > max_file_size_bytes:
        max_mb = max_file_size_bytes / (1024 * 1024)
        raise UserFacingError(
            f"That file is too large ({len(data) / (1024 * 1024):.1f} MB). Please upload a file under {max_mb:.0f} MB."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VOICE_NOTE_EXTENSIONS:
        raise UserFacingError(
            "This voice note format isn't supported. Please upload "
            f"{', '.join(sorted(ext.lstrip('.').upper() for ext in SUPPORTED_VOICE_NOTE_EXTENSIONS))}."
        )

    try:
        if suffix in SUPPORTED_EXTENSIONS:
            audio_sample = load_audio_bytes(data, filename)
        else:
            audio_sample = decode_bytes_to_audio_sample(data, filename, suffix)
    except (AudioLoadError, MediaDecodeError) as exc:
        raise UserFacingError(
            "This voice note couldn't be decoded. It may be corrupted, use an unsupported codec, "
            "or contain no audio stream.",
            technical_detail=str(exc),
        ) from exc

    if audio_sample.duration_seconds > max_duration_seconds:
        raise UserFacingError(
            f"This voice note is {audio_sample.duration_seconds:.1f} seconds long. "
            f"Please use a clip of {max_duration_seconds:.0f} seconds or less."
        )

    return audio_sample


def validate_and_load_microphone(data: bytes, max_duration_seconds: float = MAX_DURATION_SECONDS) -> AudioSample:
    """Validate and decode a microphone recording. `st.audio_input`
    already produces WAV bytes, so this uses the SAME native loader as a
    regular WAV upload -- no separate decode path, no bypass of
    normalization just because the capture was already close to the
    target sample rate."""
    if not data:
        raise UserFacingError("No audio was captured. Please record again.")

    try:
        audio_sample = load_audio_bytes(data, "microphone_capture.wav")
    except AudioLoadError as exc:
        raise UserFacingError(
            "This recording couldn't be decoded. Please try recording again.", technical_detail=str(exc)
        ) from exc

    if audio_sample.duration_seconds < 0.5:
        raise UserFacingError("That recording is too short to analyze. Please record at least half a second of speech.")
    if audio_sample.duration_seconds > max_duration_seconds:
        raise UserFacingError(
            f"That recording is {audio_sample.duration_seconds:.1f} seconds long. "
            f"Please record {max_duration_seconds:.0f} seconds or less."
        )

    return audio_sample


def validate_video_upload_size(data: bytes, max_file_size_bytes: int = MAX_VIDEO_FILE_SIZE_BYTES) -> None:
    if len(data) > max_file_size_bytes:
        max_mb = max_file_size_bytes / (1024 * 1024)
        raise UserFacingError(
            f"That video is too large ({len(data) / (1024 * 1024):.1f} MB). Please upload a video under {max_mb:.0f} MB."
        )


def validate_video_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise UserFacingError(
            "This video format isn't supported. Please upload "
            f"{', '.join(sorted(ext.lstrip('.').upper() for ext in SUPPORTED_VIDEO_EXTENSIONS))}."
        )
