"""Tests for app/analysis/media_ffmpeg.py -- the ffmpeg-backed input
adapter used for voice-note container formats and video audio-track
extraction. All tests skip gracefully if ffmpeg/ffprobe are not present
in the environment (mirrors the existing robustness-test pattern of
app/analysis/robustness.py::ffmpeg_available), so the default suite never
hard-fails on a machine without ffmpeg -- it just gets less coverage.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from app.analysis.media_ffmpeg import MediaDecodeError, decode_bytes_to_audio_sample, extract_video_audio_window, probe_file

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not available in this environment")


def _make_tone_wav(path, duration=3.0, sample_rate=16000):
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}:sample_rate={sample_rate}", str(path)],
        check=True,
        timeout=30,
    )


def _transcode(src, dst, codec_args):
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src)] + codec_args + [str(dst)], check=True, timeout=30)


def _make_video(path, duration=6.0, with_audio=True):
    if with_audio:
        subprocess.run(
            [
                FFMPEG, "-y", "-v", "error",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}:sample_rate=16000",
                "-f", "lavfi", "-i", f"color=size=160x120:rate=5:duration={duration}",
                "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path),
            ],
            check=True,
            timeout=60,
        )
    else:
        subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=size=160x120:rate=5:duration={duration}", "-an", "-c:v", "libx264", str(path)],
            check=True,
            timeout=60,
        )


@requires_ffmpeg
def test_probe_file_reports_real_metadata(tmp_path):
    wav = tmp_path / "tone.wav"
    _make_tone_wav(wav)
    probed = probe_file(str(wav))
    assert probed.has_audio_stream is True
    assert probed.duration_seconds is not None and probed.duration_seconds > 2.5


@requires_ffmpeg
def test_decode_bytes_to_audio_sample_handles_ogg(tmp_path):
    wav = tmp_path / "tone.wav"
    ogg = tmp_path / "tone.ogg"
    _make_tone_wav(wav)
    _transcode(wav, ogg, ["-c:a", "libvorbis"])
    sample = decode_bytes_to_audio_sample(ogg.read_bytes(), "tone.ogg", ".ogg")
    assert sample.sample_rate == 16000
    assert sample.duration_seconds > 2.5


@requires_ffmpeg
def test_decode_bytes_to_audio_sample_handles_opus(tmp_path):
    wav = tmp_path / "tone.wav"
    opus = tmp_path / "tone.opus"
    _make_tone_wav(wav)
    _transcode(wav, opus, ["-c:a", "libopus"])
    sample = decode_bytes_to_audio_sample(opus.read_bytes(), "tone.opus", ".opus")
    assert sample.sample_rate == 16000
    assert sample.duration_seconds > 2.5


@requires_ffmpeg
def test_decode_bytes_to_audio_sample_empty_bytes_raises():
    with pytest.raises(MediaDecodeError):
        decode_bytes_to_audio_sample(b"", "empty.ogg", ".ogg")


@requires_ffmpeg
def test_decode_bytes_to_audio_sample_corrupt_data_raises():
    with pytest.raises(MediaDecodeError):
        decode_bytes_to_audio_sample(b"not actually audio data" * 50, "corrupt.ogg", ".ogg")


@requires_ffmpeg
def test_decode_leaves_no_temp_files_behind(tmp_path):
    import os
    import tempfile

    wav = tmp_path / "tone.wav"
    ogg = tmp_path / "tone.ogg"
    _make_tone_wav(wav)
    _transcode(wav, ogg, ["-c:a", "libvorbis"])
    before = set(os.listdir(tempfile.gettempdir()))
    decode_bytes_to_audio_sample(ogg.read_bytes(), "tone.ogg", ".ogg")
    after = set(os.listdir(tempfile.gettempdir()))
    assert after == before


@requires_ffmpeg
def test_extract_video_audio_window_extracts_only_selected_range(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video, duration=10.0, with_audio=True)
    sample, probed = extract_video_audio_window(video.read_bytes(), "clip.mp4", ".mp4", start_seconds=2.0, window_seconds=4.0)
    assert probed.has_audio_stream is True
    assert 3.5 <= sample.duration_seconds <= 4.5  # ~4s window, allow small encoder slack


@requires_ffmpeg
def test_extract_video_audio_window_no_audio_stream_raises(tmp_path):
    video = tmp_path / "silent.mp4"
    _make_video(video, duration=3.0, with_audio=False)
    with pytest.raises(MediaDecodeError):
        extract_video_audio_window(video.read_bytes(), "silent.mp4", ".mp4", start_seconds=0.0, window_seconds=3.0)


@requires_ffmpeg
def test_probe_file_video_reports_video_codec(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video, duration=3.0, with_audio=True)
    probed = probe_file(str(video))
    assert probed.video_codec is not None
    assert probed.audio_codec != "Not available"
