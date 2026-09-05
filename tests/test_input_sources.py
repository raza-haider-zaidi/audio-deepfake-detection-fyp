"""Tests for app/analysis/input_sources.py -- the unified input-adapter
architecture. Confirms every source (uploaded audio file, microphone,
voice note, video audio) converges on the same NormalizedAudioInput
shape, that source_type is preserved, and that error paths (empty
recording, unsupported extension, missing audio stream) surface a
UserFacingError rather than a raw exception."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from app.analysis import input_sources as ins
from app.errors import UserFacingError
from audio_deepfake_detector.utils.datatypes import AudioSample

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available in this environment")


def _wav_bytes(duration_seconds=2.0, sample_rate=16000):
    import io

    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV")
    return buf.getvalue()


def test_from_uploaded_file_returns_normalized_input_with_correct_source_type():
    ni = ins.from_uploaded_file(_wav_bytes(), "clip.wav")
    assert ni.source_type == ins.SOURCE_AUDIO_FILE
    assert isinstance(ni.audio_sample, AudioSample)
    assert ni.sha256 and len(ni.sha256) == 64
    assert ni.source_label == "Audio File"


def test_from_microphone_returns_normalized_input():
    ni = ins.from_microphone(_wav_bytes())
    assert ni.source_type == ins.SOURCE_MICROPHONE
    assert ni.source_metadata["requested_sample_rate_hz"] == 16000
    assert ni.source_label == "Microphone Capture"


def test_from_microphone_empty_bytes_raises_user_facing_error():
    with pytest.raises(UserFacingError):
        ins.from_microphone(b"")


def test_from_microphone_too_short_recording_raises_user_facing_error():
    with pytest.raises(UserFacingError):
        ins.from_microphone(_wav_bytes(duration_seconds=0.1))


def test_from_voice_note_native_format_reuses_native_loader():
    ni = ins.from_voice_note(_wav_bytes(), "note.wav")
    assert ni.source_type == ins.SOURCE_VOICE_NOTE
    assert ni.source_metadata["normalized_to"] == "Mono · 16 kHz"


def test_from_voice_note_unsupported_extension_raises_user_facing_error():
    with pytest.raises(UserFacingError):
        ins.from_voice_note(_wav_bytes(), "note.amr")


def test_from_voice_note_corrupted_file_raises_user_facing_error():
    with pytest.raises(UserFacingError):
        ins.from_voice_note(b"not a real wav file" * 10, "note.wav")


@requires_ffmpeg
def test_from_voice_note_ogg_converges_to_same_audio_sample_type(tmp_path):
    wav = tmp_path / "tone.wav"
    wav.write_bytes(_wav_bytes(duration_seconds=3.0))
    ogg = tmp_path / "tone.ogg"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(wav), "-c:a", "libvorbis", str(ogg)], check=True, timeout=30)
    ni = ins.from_voice_note(ogg.read_bytes(), "tone.ogg")
    assert isinstance(ni.audio_sample, AudioSample)
    assert ni.audio_sample.sample_rate == 16000


def test_validate_video_upload_size_rejects_oversized_file():
    from app.validation import validate_video_upload_size

    with pytest.raises(UserFacingError):
        validate_video_upload_size(b"x" * 100, max_file_size_bytes=10)


def test_probe_video_rejects_unsupported_extension():
    with pytest.raises(UserFacingError):
        ins.probe_video(b"irrelevant", "clip.avi")


@requires_ffmpeg
def test_from_video_extracts_selected_window_and_converges_to_normalized_input(tmp_path):
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            FFMPEG, "-y", "-v", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=8:sample_rate=16000",
            "-f", "lavfi", "-i", "color=size=160x120:rate=5:duration=8",
            "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video),
        ],
        check=True,
        timeout=60,
    )
    ni = ins.from_video(video.read_bytes(), "clip.mp4", start_seconds=1.0, window_seconds=5.0)
    assert ni.source_type == ins.SOURCE_VIDEO_AUDIO
    assert isinstance(ni.audio_sample, AudioSample)
    assert 4.0 <= ni.audio_sample.duration_seconds <= 5.5
    assert "selected_interval" in ni.source_metadata
    assert ni.source_label == "Video Audio"


def test_all_source_types_produce_the_same_normalized_shape():
    """Unified pipeline guarantee: every adapter returns the SAME
    NormalizedAudioInput type, carrying an AudioSample built by the SAME
    frozen decode/normalize pipeline -- no source-specific prediction
    logic exists anywhere downstream."""
    audio_ni = ins.from_uploaded_file(_wav_bytes(), "a.wav")
    mic_ni = ins.from_microphone(_wav_bytes())
    voice_ni = ins.from_voice_note(_wav_bytes(), "v.wav")
    for ni in (audio_ni, mic_ni, voice_ni):
        assert isinstance(ni, ins.NormalizedAudioInput)
        assert isinstance(ni.audio_sample, AudioSample)
        assert ni.audio_sample.sample_rate == 16000
        assert ni.audio_sample.waveform.dtype == np.float32
