"""Upload validation for the Streamlit layer.

Delegates actual audio decoding to the shared preprocessing pipeline in
src/audio_deepfake_detector — this module only adds Streamlit-facing
constraints (max duration, max file size) and turns failures into
UserFacingError with a friendly message.
"""

from __future__ import annotations

from app.errors import UserFacingError
from audio_deepfake_detector.preprocessing.audio_loader import (
    SUPPORTED_EXTENSIONS,
    AudioLoadError,
    load_audio_bytes,
)
from audio_deepfake_detector.utils.datatypes import AudioSample

MAX_DURATION_SECONDS = 30.0
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


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
