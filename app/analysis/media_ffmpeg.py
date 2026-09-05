"""FFmpeg-backed decoding for input formats the shared soundfile-based
loader (audio_deepfake_detector.preprocessing.audio_loader) cannot read
directly -- voice-note container formats (M4A/AAC/OGG/OPUS/WEBM) and
video containers (MP4/MOV/MKV/WEBM).

This module is an INPUT ADAPTER ONLY. It never touches model logic: every
function here ends by handing a plain 16 kHz mono WAV file to the
existing, unmodified `audio_deepfake_detector.preprocessing.audio_loader
.load_audio_file`, so decoded audio from any source converges on the
exact same finalize/resample/mono/finite-check path already used for
ordinary WAV/MP3/FLAC uploads. No preprocessing behavior is duplicated or
reimplemented here.

All ffmpeg/ffprobe invocations use explicit argument lists (never
`shell=True`, never a string-concatenated command), so a hostile filename
cannot be interpreted as shell syntax. All intermediate files are created
with `tempfile` in the OS temp directory and deleted in a `finally` block
-- nothing is written under the project directory, and nothing survives
past the current analysis.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from audio_deepfake_detector.preprocessing.audio_loader import load_audio_file
from audio_deepfake_detector.utils.datatypes import AudioSample

FFMPEG_PROBE_TIMEOUT_SECONDS = 20
FFMPEG_DECODE_TIMEOUT_SECONDS = 60


class MediaDecodeError(ValueError):
    """Raised when ffmpeg/ffprobe is unavailable, or the media cannot be
    probed/decoded/extracted. Always carries a message safe to show to a
    friendly-error wrapper upstream. `stderr` (if any) is the raw ffmpeg
    stderr output -- never shown to a user, but available for a caller to
    log server-side for diagnostics (see app/analysis/video_url.py)."""

    def __init__(self, message: str, *, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


@dataclass
class ProbedMedia:
    format_name: str
    container: str
    duration_seconds: float | None
    has_audio_stream: bool
    audio_codec: str
    video_codec: str | None  # None if no video stream (i.e. this is audio-only media)
    sample_rate: str
    channels: str
    size_bytes: int


def probe_file(path: str) -> ProbedMedia:
    """Real ffprobe metadata, authoritative for whether the file actually
    decodes -- never guessed. Raises MediaDecodeError if ffprobe is
    unavailable or the file cannot be probed at all."""
    ffprobe = ffprobe_path()
    if ffprobe is None:
        raise MediaDecodeError("Media inspection is unavailable in this deployment (ffprobe not found).")

    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
            capture_output=True,
            text=True,
            timeout=FFMPEG_PROBE_TIMEOUT_SECONDS,
            check=True,
        )
        info = json.loads(proc.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        raise MediaDecodeError("This file could not be read as media.", stderr=getattr(exc, "stderr", "") or "") from exc

    fmt = info.get("format", {})
    streams = info.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio = audio_streams[0] if audio_streams else {}

    duration = fmt.get("duration")
    return ProbedMedia(
        format_name=fmt.get("format_name", "") or "",
        container=fmt.get("format_long_name") or fmt.get("format_name") or "Not available",
        duration_seconds=float(duration) if duration else None,
        has_audio_stream=bool(audio_streams),
        audio_codec=audio.get("codec_long_name") or audio.get("codec_name") or "Not available",
        video_codec=(video_streams[0].get("codec_long_name") or video_streams[0].get("codec_name")) if video_streams else None,
        sample_rate=f"{audio['sample_rate']} Hz" if audio.get("sample_rate") else "Not available",
        channels=str(audio["channels"]) if audio.get("channels") else "Not available",
        size_bytes=int(fmt.get("size", 0) or 0),
    )


def _run_ffmpeg_decode(input_path: str, output_wav_path: str, *, start_seconds: float | None = None, duration_seconds: float | None = None) -> None:
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise MediaDecodeError("Audio decoding is unavailable in this deployment (ffmpeg not found).")

    cmd = [ffmpeg, "-y", "-v", "error"]
    if start_seconds is not None:
        cmd += ["-ss", f"{max(0.0, start_seconds):.3f}"]
    cmd += ["-i", input_path]
    if duration_seconds is not None:
        cmd += ["-t", f"{max(0.0, duration_seconds):.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-f", "wav", output_wav_path]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_DECODE_TIMEOUT_SECONDS)
    except subprocess.SubprocessError as exc:
        raise MediaDecodeError("Audio decoding timed out or failed.") from exc

    if proc.returncode != 0:
        raise MediaDecodeError(
            "This file's audio track could not be decoded (unsupported or corrupted media).",
            stderr=proc.stderr or "",
        )


def decode_bytes_to_audio_sample(data: bytes, source_name: str, suffix: str) -> AudioSample:
    """Decode arbitrary media bytes (voice note or video) to a normalized
    AudioSample by round-tripping through ffmpeg -> a temporary 16 kHz
    mono WAV -> the shared, unmodified `load_audio_file` pipeline."""
    if not data:
        raise MediaDecodeError(f"'{source_name}' is empty (0 bytes).")

    tmp_dir = Path(tempfile.mkdtemp(prefix="adf_media_"))
    input_path = tmp_dir / f"input{suffix}"
    output_path = tmp_dir / "decoded.wav"
    try:
        input_path.write_bytes(data)
        probed = probe_file(str(input_path))
        if not probed.has_audio_stream:
            raise MediaDecodeError(f"'{source_name}' does not contain an audio stream.")
        _run_ffmpeg_decode(str(input_path), str(output_path))
        sample = load_audio_file(output_path)
        return AudioSample(
            waveform=sample.waveform, sample_rate=sample.sample_rate, duration_seconds=sample.duration_seconds, source_name=source_name
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_video_audio_window(data: bytes, source_name: str, suffix: str, start_seconds: float, window_seconds: float) -> tuple[AudioSample, ProbedMedia]:
    """Probe a video file, then extract and normalize only the selected
    [start_seconds, start_seconds + window_seconds) audio window -- never
    the whole (potentially long) track. Returns the normalized audio plus
    the probed source-video metadata for reporting."""
    if not data:
        raise MediaDecodeError(f"'{source_name}' is empty (0 bytes).")

    tmp_dir = Path(tempfile.mkdtemp(prefix="adf_video_"))
    input_path = tmp_dir / f"input{suffix}"
    output_path = tmp_dir / "decoded.wav"
    try:
        input_path.write_bytes(data)
        probed = probe_file(str(input_path))
        if not probed.has_audio_stream:
            raise MediaDecodeError(f"'{source_name}' does not contain an audio track.")
        _run_ffmpeg_decode(str(input_path), str(output_path), start_seconds=start_seconds, duration_seconds=window_seconds)
        sample = load_audio_file(output_path)
        if sample.duration_seconds <= 0:
            raise MediaDecodeError("The selected interval produced no audio.")
        return (
            AudioSample(waveform=sample.waveform, sample_rate=sample.sample_rate, duration_seconds=sample.duration_seconds, source_name=source_name),
            probed,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
