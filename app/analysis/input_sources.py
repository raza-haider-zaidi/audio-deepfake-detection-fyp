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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from audio_deepfake_detector.utils.datatypes import AudioSample

SOURCE_AUDIO_FILE = "audio_file"
SOURCE_MICROPHONE = "microphone"
SOURCE_VOICE_NOTE = "voice_note"
SOURCE_VIDEO_AUDIO = "video_audio"

SOURCE_LABELS = {
    SOURCE_AUDIO_FILE: "Audio File",
    SOURCE_MICROPHONE: "Microphone Capture",
    SOURCE_VOICE_NOTE: "Voice Note",
    SOURCE_VIDEO_AUDIO: "Video Audio",
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
