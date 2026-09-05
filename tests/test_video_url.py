"""Tests for app/analysis/video_url.py (public YouTube video URL audio
ingestion) and its integration into app/analysis/input_sources.py.

URL parsing, scheme/hostname allow-listing, and SSRF (DNS/IP) protection
are tested with NO network access -- DNS resolution is monkeypatched.
Metadata/extraction adapter behavior is tested against a fake `yt_dlp`
client, so the default unit suite never depends on YouTube being online.
Anything that genuinely hits the network is marked
`@pytest.mark.integration` and skipped by default.
"""

from __future__ import annotations

import hashlib
import shutil
import socket
import subprocess

import pytest

from app.analysis import input_sources as ins
from app.analysis import video_url
from app.analysis.video_url import VideoURLError, VideoURLMetadata, validate_public_video_url
from app.errors import UserFacingError
from app.reporting.pdf_report import build_pdf_report
from app.reporting.report import build_report_data
from audio_deepfake_detector.utils.datatypes import AudioSample

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available in this environment")


# --------------------------------------------------------------------------
# URL parsing / scheme / hostname allow-list (no network)
# --------------------------------------------------------------------------


def _fake_addrinfo_public(hostname, *args, **kwargs):
    # A public IP (Google DNS) -- exercises the "resolves, and is public" path
    # without making a real DNS query depend on network availability.
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch):
    monkeypatch.setattr(video_url.socket, "getaddrinfo", _fake_addrinfo_public)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    ],
)
def test_validate_public_video_url_accepts_supported_youtube_variants(url):
    assert validate_public_video_url(url)


def test_validate_public_video_url_rejects_empty():
    with pytest.raises(VideoURLError):
        validate_public_video_url("")


def test_validate_public_video_url_rejects_invalid_url():
    with pytest.raises(VideoURLError):
        validate_public_video_url("not a url at all")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://www.youtube.com/watch?v=abc",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/plain;base64,aGVsbG8=",
    ],
)
def test_validate_public_video_url_rejects_unsupported_scheme(url):
    with pytest.raises(VideoURLError):
        validate_public_video_url(url)


def test_validate_public_video_url_rejects_localhost():
    with pytest.raises(VideoURLError):
        validate_public_video_url("http://localhost/watch?v=abc")


def test_validate_public_video_url_rejects_non_youtube_domain():
    with pytest.raises(VideoURLError):
        validate_public_video_url("https://example.com/watch?v=abc")


def test_validate_public_video_url_rejects_private_ip_resolution(monkeypatch):
    """Even an allow-listed hostname must be rejected if it resolves to a
    non-public address -- defense in depth against DNS-based SSRF."""

    def _private_addrinfo(hostname, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))]

    monkeypatch.setattr(video_url.socket, "getaddrinfo", _private_addrinfo)
    with pytest.raises(VideoURLError):
        validate_public_video_url("https://www.youtube.com/watch?v=abc")


def test_validate_public_video_url_rejects_loopback_ip_resolution(monkeypatch):
    def _loopback_addrinfo(hostname, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(video_url.socket, "getaddrinfo", _loopback_addrinfo)
    with pytest.raises(VideoURLError):
        validate_public_video_url("https://www.youtube.com/watch?v=abc")


def test_validate_public_video_url_rejects_link_local_ip_resolution(monkeypatch):
    def _link_local_addrinfo(hostname, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))]

    monkeypatch.setattr(video_url.socket, "getaddrinfo", _link_local_addrinfo)
    with pytest.raises(VideoURLError):
        validate_public_video_url("https://www.youtube.com/watch?v=abc")


def test_validate_public_video_url_rejects_unresolvable_host(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("name resolution failed")

    monkeypatch.setattr(video_url.socket, "getaddrinfo", _raise)
    with pytest.raises(VideoURLError):
        validate_public_video_url("https://www.youtube.com/watch?v=abc")


# --------------------------------------------------------------------------
# Metadata retrieval (fake yt-dlp client -- no network)
# --------------------------------------------------------------------------

FAKE_INFO = {
    "extractor_key": "Youtube",
    "title": "Sample Public Video",
    "duration": 245.0,
    "uploader": "Sample Channel",
    "id": "abc123",
    "webpage_url": "https://www.youtube.com/watch?v=abc123",
    "thumbnail": "https://i.ytimg.com/vi/abc123/hqdefault.jpg",
    "formats": [{"acodec": "mp4a.40.2", "vcodec": "none"}],
}


class _FakeYDL:
    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        return FAKE_INFO


class _RaisingYDL(_FakeYDL):
    def extract_info(self, url, download=False):
        raise RuntimeError("The page needs to be reloaded.")


def test_fetch_metadata_returns_parsed_fields(monkeypatch):
    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _FakeYDL)
    metadata = video_url.fetch_metadata("https://www.youtube.com/watch?v=abc123")
    assert metadata.platform == "YouTube"
    assert metadata.title == "Sample Public Video"
    assert metadata.duration_seconds == 245.0
    assert metadata.uploader == "Sample Channel"
    assert metadata.video_id == "abc123"


def test_fetch_metadata_wraps_extractor_failure_without_traceback(monkeypatch):
    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _RaisingYDL)
    with pytest.raises(VideoURLError) as excinfo:
        video_url.fetch_metadata("https://www.youtube.com/watch?v=abc123")
    assert "reloaded" not in str(excinfo.value)
    assert "Unable to access this video" in str(excinfo.value)


def test_fetch_metadata_rejects_missing_audio_track(monkeypatch):
    class _NoAudioYDL(_FakeYDL):
        def extract_info(self, url, download=False):
            return {**FAKE_INFO, "formats": [{"acodec": "none", "vcodec": "avc1"}]}

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _NoAudioYDL)
    with pytest.raises(VideoURLError):
        video_url.fetch_metadata("https://www.youtube.com/watch?v=abc123")


def test_fetch_metadata_rejects_non_youtube_extractor(monkeypatch):
    class _OtherYDL(_FakeYDL):
        def extract_info(self, url, download=False):
            return {**FAKE_INFO, "extractor_key": "Vimeo"}

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _OtherYDL)
    with pytest.raises(VideoURLError):
        video_url.fetch_metadata("https://www.youtube.com/watch?v=abc123")


def test_fetch_metadata_rejects_excessively_long_source(monkeypatch):
    class _LongYDL(_FakeYDL):
        def extract_info(self, url, download=False):
            return {**FAKE_INFO, "duration": video_url.MAX_SOURCE_DURATION_SECONDS + 1}

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _LongYDL)
    with pytest.raises(VideoURLError):
        video_url.fetch_metadata("https://www.youtube.com/watch?v=abc123")


def test_input_sources_fetch_video_url_metadata_wraps_user_facing_error(monkeypatch):
    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _RaisingYDL)
    with pytest.raises(UserFacingError):
        ins.fetch_video_url_metadata("https://www.youtube.com/watch?v=abc123")


# --------------------------------------------------------------------------
# Extraction (real ffmpeg decode step, fake yt-dlp download)
# --------------------------------------------------------------------------


def _make_tone_wav(path, duration=6.0, sample_rate=16000):
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}:sample_rate={sample_rate}", str(path)],
        check=True,
        timeout=30,
    )


class _DownloadingYDL:
    """Fake yt-dlp client whose .download() writes a real, ffmpeg-decodable
    WAV file to the configured outtmpl -- exercises the real
    extract->decode->normalize path without any network access."""

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def download(self, urls):
        path = self.opts["outtmpl"].replace("%(ext)s", "wav")
        _make_tone_wav(path, duration=6.0)


@requires_ffmpeg
def test_extract_audio_interval_hashes_actual_extracted_bytes_not_url(monkeypatch):
    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _DownloadingYDL)
    audio_sample, raw_bytes, ext = video_url.extract_audio_interval(
        "https://www.youtube.com/watch?v=abc123", start_seconds=0.0, window_seconds=5.0
    )
    assert isinstance(audio_sample, AudioSample)
    assert raw_bytes  # non-empty
    assert hashlib.sha256(raw_bytes).hexdigest() != hashlib.sha256(b"https://www.youtube.com/watch?v=abc123").hexdigest()
    assert ext == ".wav"


@requires_ffmpeg
def test_extract_audio_interval_leaves_no_temp_directory_behind(monkeypatch, tmp_path):
    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _DownloadingYDL)
    captured_dirs = []
    real_mkdtemp = video_url.tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        captured_dirs.append(d)
        return d

    monkeypatch.setattr(video_url.tempfile, "mkdtemp", _tracking_mkdtemp)
    video_url.extract_audio_interval("https://www.youtube.com/watch?v=abc123", start_seconds=0.0, window_seconds=5.0)
    assert captured_dirs, "mkdtemp should have been called"
    import os

    assert not os.path.exists(captured_dirs[0])


@requires_ffmpeg
def test_from_video_url_produces_normalized_input_with_correct_source_type(monkeypatch):
    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _DownloadingYDL)
    metadata = VideoURLMetadata(
        platform="YouTube",
        title="Sample Public Video",
        duration_seconds=245.0,
        uploader="Sample Channel",
        video_id="abc123",
        webpage_url="https://www.youtube.com/watch?v=abc123",
        thumbnail_url=None,
    )
    normalized = ins.from_video_url(
        "https://www.youtube.com/watch?v=abc123", metadata, start_seconds=10.0, window_seconds=5.0
    )
    assert normalized.source_type == ins.SOURCE_VIDEO_URL
    assert normalized.source_label == "Online Video"
    assert normalized.source_metadata["platform"] == "YouTube"
    assert normalized.source_metadata["source_title"] == "Sample Public Video"
    assert normalized.source_metadata["source_url"] == "https://www.youtube.com/watch?v=abc123"
    assert normalized.source_metadata["selected_interval"].startswith("10s")
    assert normalized.audio_sample.sample_rate == 16000
    assert normalized.audio_sample.waveform.dtype.name == "float32"
    assert len(normalized.sha256) == 64


@requires_ffmpeg
def test_extract_audio_interval_wraps_download_failure(monkeypatch):
    class _FailingYDL(_DownloadingYDL):
        def download(self, urls):
            raise RuntimeError("Sign in to confirm you're not a bot")

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _FailingYDL)
    with pytest.raises(VideoURLError) as excinfo:
        video_url.extract_audio_interval("https://www.youtube.com/watch?v=abc123", start_seconds=0.0, window_seconds=5.0)
    assert "bot" not in str(excinfo.value)
    assert "Unable to access this video" in str(excinfo.value)


# --------------------------------------------------------------------------
# Report generation for the video_url source (no network, no ffmpeg)
# --------------------------------------------------------------------------


def _url_report_data(**overrides):
    from datetime import datetime

    kwargs = dict(
        generated_at=datetime(2026, 9, 5, 12, 0, 0),
        filename="Sample Public Video",
        file_sha256="feedface" * 8,
        duration_seconds=30.0,
        sample_rate=16000,
        audio_format="WEBM",
        channels=1,
        presentation_state="BONAFIDE",
        prediction_label="Likely Real / Bonafide",
        bonafide_probability=0.88,
        spoof_probability=0.12,
        calibrated_threshold=0.9397,
        binary_model_decision="BONAFIDE",
        n_segments=1,
        segment_rows=[],
        segment_agreement=None,
        audio_quality={"peak_amplitude": 0.5, "rms_level": 0.08, "silence_ratio": 0.03, "clipping_ratio": 0.0, "approximate_bitrate_kbps": None},
        suitability_level="Good",
        model_info={
            "display_name": "Spectra-AASIST3 INT8",
            "architecture_short": "XLS-R-300M + KAN-AASIST",
            "runtime": "ONNX Runtime CPU",
            "repository": "Limitless-8/spectra-aasist3-int8-audio-deepfake",
            "revision": "abcdef1234",
            "threshold_description": "Calibrated 93.97% spoof probability",
        },
        inference_time_ms=310.0,
        source_type="video_url",
        source_metadata={
            "format": "WEBM",
            "platform": "YouTube",
            "source_title": "Sample Public Video",
            "source_url": "https://www.youtube.com/watch?v=" + "x" * 80,
            "source_uploader": "Sample Channel",
            "video_duration_seconds": 522.0,
            "selected_interval": "10s–40s",
        },
    )
    kwargs.update(overrides)
    return build_report_data(**kwargs)


def test_report_data_preserves_video_url_source_metadata():
    data = _url_report_data()
    assert data["analysis_information"]["source_type"] == "video_url"
    assert data["analysis_information"]["source_metadata"]["platform"] == "YouTube"


def test_pdf_report_renders_for_video_url_source():
    import io

    from pypdf import PdfReader

    data = _url_report_data()
    pdf_bytes = build_pdf_report(data)
    assert pdf_bytes[:5] == b"%PDF-"
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_bytes)).pages)
    assert "Online Video" in text
    assert "Sample Public Video" in text
    assert "YouTube" in text


def test_pdf_report_never_exposes_personal_hosting_account_for_model_artifact():
    """The frozen model artifact's internal repository identifier
    (Limitless-8/...) must never appear in a user-facing report -- only
    the neutral display label."""
    import io

    from pypdf import PdfReader

    data = _url_report_data()
    pdf_bytes = build_pdf_report(data)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_bytes)).pages)
    assert "Limitless" not in text
    assert "Spectra-AASIST3 INT8 deployment artifact" in text


def test_html_report_wraps_long_url_and_hides_personal_hosting_account():
    from app.reporting.report import render_html_report

    data = _url_report_data()
    html_report = render_html_report(data)
    assert "word-break" in html_report
    assert "Limitless" not in html_report
