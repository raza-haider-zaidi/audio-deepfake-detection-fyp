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
# Developer-only diagnostics (Step 13) -- never shown to end users by default
# --------------------------------------------------------------------------


def test_diagnostics_snapshot_never_raises_and_reports_yt_dlp_version():
    snapshot = video_url.diagnostics_snapshot()
    assert snapshot["yt_dlp_version"]
    assert set(snapshot.keys()) >= {
        "yt_dlp_version",
        "yt_dlp_ejs_version",
        "ffmpeg_path",
        "js_runtime",
        "jsc_providers",
        "po_token_provider",
    }


def test_debug_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv(video_url.DEBUG_ENV_VAR, raising=False)
    assert video_url.is_debug_enabled() is False


def test_debug_flag_can_be_enabled_via_env(monkeypatch):
    monkeypatch.setenv(video_url.DEBUG_ENV_VAR, "1")
    assert video_url.is_debug_enabled() is True


# --------------------------------------------------------------------------
# Cloud diagnostics logging -- the real yt-dlp failure must reach the
# server-side log (Streamlit Cloud logs) even though the UI only ever
# shows the safe, generic message. No network required: yt_dlp.YoutubeDL
# is replaced with a fake client that raises like the real library would.
# --------------------------------------------------------------------------


def test_sanitize_log_text_redacts_signed_media_url_and_tokens():
    raw = (
        "Failed to download https://rr2---sn-abc.googlevideo.com/videoplayback?"
        "expire=123&sig=AABBCCDD&token=xyz and cookie=super-secret-session"
    )
    sanitized = video_url._sanitize_log_text(raw)
    assert "sig=AABBCCDD" not in sanitized
    assert "super-secret-session" not in sanitized
    assert "[signed-media-url-redacted]" in sanitized


def test_sanitize_log_text_keeps_plain_public_urls():
    raw = "extracting https://www.youtube.com/watch?v=abc123"
    sanitized = video_url._sanitize_log_text(raw)
    assert "https://www.youtube.com/watch?v=abc123" in sanitized


def test_sanitize_log_text_truncates_long_text():
    sanitized = video_url._sanitize_log_text("x" * 5000, max_length=100)
    assert len(sanitized) <= 120
    assert sanitized.endswith("...[truncated]")


def test_metadata_failure_logs_category_and_sanitized_error_server_side(monkeypatch, caplog):
    """The exact bug this patch fixes: previously warning/error yt-dlp
    output and the classified failure were both logged at DEBUG level,
    which Streamlit Cloud's default log level drops -- so a real
    production failure produced NOTHING in the server logs. The failure
    line must now be emitted at ERROR level with the category, exception
    type, and a sanitized version of the real error."""

    class _BotChallengeYDL(_FakeYDL):
        def extract_info(self, url, download=False):
            raise RuntimeError(
                "Sign in to confirm you're not a bot. https://accounts.google.com/signin?sig=SECRETVALUE"
            )

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _BotChallengeYDL)
    with caplog.at_level("ERROR", logger="app.analysis.video_url"):
        with pytest.raises(VideoURLError) as excinfo:
            video_url.fetch_metadata("https://www.youtube.com/watch?v=abc123")

    # UI-facing message stays safe and generic/specific -- never the raw text.
    assert "bot" not in str(excinfo.value)
    assert excinfo.value.category == video_url.YOUTUBE_BOT_CHALLENGE

    failure_records = [r for r in caplog.records if "VIDEO_URL_FAILURE" in r.message]
    assert failure_records, "expected a VIDEO_URL_FAILURE record at ERROR level"
    record = failure_records[0]
    assert record.levelname == "ERROR"
    assert "category=YOUTUBE_BOT_CHALLENGE" in record.message
    assert "exception_type=RuntimeError" in record.message
    assert "sig=SECRETVALUE" not in record.message


@requires_ffmpeg
def test_extraction_failure_logs_category_server_side(monkeypatch, caplog):
    class _FailingYDL(_DownloadingYDL):
        def download(self, urls):
            raise RuntimeError("Requested format is not available")

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _FailingYDL)
    with caplog.at_level("ERROR", logger="app.analysis.video_url"):
        with pytest.raises(VideoURLError):
            video_url.extract_audio_interval("https://www.youtube.com/watch?v=abc123", start_seconds=0.0, window_seconds=5.0)

    failure_records = [r for r in caplog.records if "VIDEO_URL_FAILURE" in r.message]
    assert failure_records
    assert "category=FORMAT_UNAVAILABLE" in failure_records[0].message
    assert "stage=native_audio_download" in failure_records[0].message


@requires_ffmpeg
def test_extraction_emits_pre_extraction_diagnostic_line(monkeypatch, caplog):
    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _DownloadingYDL)
    with caplog.at_level("INFO", logger="app.analysis.video_url"):
        video_url.extract_audio_interval("https://www.youtube.com/watch?v=abc123", start_seconds=0.0, window_seconds=5.0)

    diagnostic_records = [r for r in caplog.records if "VIDEO_URL_DIAGNOSTIC" in r.message]
    assert diagnostic_records, "expected a VIDEO_URL_DIAGNOSTIC record before extraction"
    assert "video_id=abc123" in diagnostic_records[0].message


def test_silent_ydl_logger_routes_warning_and_error_to_matching_log_level(caplog):
    """Regression guard for the actual production bug: warning()/error()
    must NOT be downgraded to DEBUG (that is what hid the real cloud
    failure from the server logs in the first place)."""
    ydl_logger = video_url._SilentYDLLogger()
    with caplog.at_level("DEBUG", logger="app.analysis.video_url"):
        ydl_logger.warning("some yt-dlp warning")
        ydl_logger.error("some yt-dlp error")

    levels = {r.levelname for r in caplog.records}
    assert "WARNING" in levels
    assert "ERROR" in levels
    assert not any(r.levelname == "DEBUG" and "yt-dlp warning" in r.message for r in caplog.records)


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
        # Simulates Stage A: the FULL source audio track is downloaded
        # (never pre-cut to the requested interval) -- long enough to
        # cover every start/window combination used across this test file,
        # since Stage B now performs the real local -ss/-t trim against it.
        path = self.opts["outtmpl"].replace("%(ext)s", "wav")
        _make_tone_wav(path, duration=20.0)


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
    assert excinfo.value.category == video_url.YOUTUBE_BOT_CHALLENGE
    assert "did not permit" in str(excinfo.value)


# --------------------------------------------------------------------------
# Failure classification (Step 7 of the hardening pass) -- no network
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_message,expected_category",
    [
        ("ERROR: Sign in to confirm you're not a bot", video_url.YOUTUBE_BOT_CHALLENGE),
        ("PO Token required for this request", video_url.PO_TOKEN_REQUIRED),
        ("No supported JavaScript runtime could be found", video_url.JS_RUNTIME_UNAVAILABLE),
        ("Requested format is not available", video_url.FORMAT_UNAVAILABLE),
        ("HTTPSConnectionPool: Read timed out", video_url.NETWORK_TIMEOUT),
        ("ERROR: [youtube] xyz: This video is unavailable", video_url.SOURCE_UNAVAILABLE),
        ("ffmpeg exited with a non-zero error", video_url.FFMPEG_FAILED),
        ("Some completely unexpected extractor error", video_url.EXTRACTION_FAILED),
    ],
)
def test_classify_extraction_error(raw_message, expected_category):
    assert video_url.classify_extraction_error(RuntimeError(raw_message)) == expected_category


def test_every_failure_category_maps_to_a_traceback_free_message():
    raw = "some internal yt-dlp/ffmpeg detail that must never reach the user"
    for category in (
        video_url.SOURCE_UNAVAILABLE,
        video_url.NO_AUDIO_STREAM,
        video_url.YOUTUBE_BOT_CHALLENGE,
        video_url.JS_RUNTIME_UNAVAILABLE,
        video_url.PO_TOKEN_REQUIRED,
        video_url.FORMAT_UNAVAILABLE,
        video_url.NETWORK_TIMEOUT,
        video_url.EXTRACTION_FAILED,
        video_url.FFMPEG_FAILED,
        video_url.NATIVE_DOWNLOAD_FAILED,
        video_url.FFMPEG_LOCAL_PROCESSING_FAILED,
    ):
        message = video_url._safe_message(category)
        assert raw not in message


@requires_ffmpeg
def test_extract_audio_interval_uses_download_timeout_not_metadata_timeout(monkeypatch):
    """Regression guard: the download call must use the (longer)
    YTDLP_DOWNLOAD_TIMEOUT_SECONDS, not the metadata-only timeout -- using
    the short metadata timeout for real audio downloads risked premature
    timeouts on slower connections."""
    captured_opts = {}

    class _CapturingYDL(_DownloadingYDL):
        def __init__(self, opts):
            captured_opts.update(opts)
            super().__init__(opts)

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _CapturingYDL)
    video_url.extract_audio_interval("https://www.youtube.com/watch?v=abc123", start_seconds=0.0, window_seconds=5.0)
    assert captured_opts["socket_timeout"] == video_url.YTDLP_DOWNLOAD_TIMEOUT_SECONDS


# --------------------------------------------------------------------------
# Architecture regression (Step 12 of the two-stage hardening pass):
# extract_audio_interval() must NEVER configure yt-dlp to invoke ffmpeg as
# a remote/external downloader again -- that is the exact bug that caused
# "ffmpeg exited with code 8" on Streamlit Cloud (download_ranges +
# force_keyframes_at_cuts makes yt-dlp run ffmpeg directly against the
# remote signed media URL). And local ffmpeg processing must always
# receive a local filesystem path.
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_native_download_opts_never_configure_ffmpeg_as_remote_downloader(monkeypatch):
    captured_opts = {}

    class _CapturingYDL(_DownloadingYDL):
        def __init__(self, opts):
            captured_opts.update(opts)
            super().__init__(opts)

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _CapturingYDL)
    video_url.extract_audio_interval("https://www.youtube.com/watch?v=abc123", start_seconds=0.0, window_seconds=5.0)

    # These are what make yt-dlp invoke ffmpeg directly against the remote
    # URL as an external/ranged downloader -- must be absent entirely.
    assert "download_ranges" not in captured_opts
    assert "force_keyframes_at_cuts" not in captured_opts
    assert captured_opts.get("external_downloader") != "ffmpeg"
    assert "external_downloader_args" not in captured_opts
    # Format selection stays a robust audio-only preference, not a single
    # forced container.
    assert captured_opts["format"] == "bestaudio/best"


@requires_ffmpeg
def test_local_interval_extraction_ffmpeg_input_is_a_local_file(monkeypatch):
    """Proves Stage B (local interval extraction) never hands ffmpeg a
    remote URL -- every `-i` argument ffmpeg receives during
    extract_audio_interval() must be an existing local file path."""
    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _DownloadingYDL)

    import os

    import app.analysis.media_ffmpeg as media_ffmpeg

    captured_input_paths = []
    real_subprocess_run = media_ffmpeg.subprocess.run

    def _tracking_run(cmd, *args, **kwargs):
        # media_ffmpeg.subprocess IS the shared stdlib `subprocess` module
        # object, so this also sees the test fixture's own synthetic-tone
        # ffmpeg call (`-f lavfi -i sine=...`) -- skip that one, it is not
        # part of the code path under test.
        if "-i" in cmd and "lavfi" not in cmd:
            input_path = cmd[cmd.index("-i") + 1]
            # Checked HERE, before the real ffmpeg call and before the
            # temp directory is cleaned up -- proves ffmpeg is handed an
            # existing local file, never a remote URL.
            captured_input_paths.append((input_path, os.path.exists(input_path)))
        return real_subprocess_run(cmd, *args, **kwargs)

    monkeypatch.setattr(media_ffmpeg.subprocess, "run", _tracking_run)
    video_url.extract_audio_interval("https://www.youtube.com/watch?v=abc123", start_seconds=0.0, window_seconds=5.0)

    assert captured_input_paths, "expected at least one ffmpeg -i invocation"
    for input_path, existed_at_call_time in captured_input_paths:
        assert not input_path.lower().startswith(("http://", "https://"))
        assert existed_at_call_time, f"ffmpeg -i target {input_path!r} did not exist as a local file at call time"


def test_native_download_failure_uses_native_download_failed_fallback_category():
    """A native-download-stage exception that matches no specific pattern
    must fall back to NATIVE_DOWNLOAD_FAILED (not the shared generic
    EXTRACTION_FAILED), so logs distinguish which stage failed."""
    category = video_url.classify_extraction_error(
        RuntimeError("some completely novel yt-dlp download error"),
        default=video_url.NATIVE_DOWNLOAD_FAILED,
    )
    assert category == video_url.NATIVE_DOWNLOAD_FAILED


@requires_ffmpeg
def test_local_ffmpeg_failure_is_classified_separately_from_native_download_failure(monkeypatch, caplog):
    """Step 8: a native-download success followed by a local ffmpeg
    failure must be classified as FFMPEG_LOCAL_PROCESSING_FAILED, distinct
    from any native-download-stage category."""

    class _BadAudioYDL(_DownloadingYDL):
        def download(self, urls):
            # Native download "succeeds" but writes unusable garbage bytes
            # instead of real audio -- simulates ffmpeg being handed a
            # corrupt/incompatible local file at Stage B.
            path = self.opts["outtmpl"].replace("%(ext)s", "bin")
            with open(path, "wb") as fh:
                fh.write(b"not a real media file")

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _BadAudioYDL)
    with caplog.at_level("ERROR", logger="app.analysis.video_url"):
        with pytest.raises(VideoURLError) as excinfo:
            video_url.extract_audio_interval("https://www.youtube.com/watch?v=abc123", start_seconds=0.0, window_seconds=5.0)

    assert excinfo.value.category == video_url.FFMPEG_LOCAL_PROCESSING_FAILED

    failure_records = [r for r in caplog.records if "VIDEO_URL_FAILURE" in r.message]
    assert failure_records
    assert "stage=local_interval_extract" in failure_records[0].message
    assert "category=FFMPEG_LOCAL_PROCESSING_FAILED" in failure_records[0].message


@requires_ffmpeg
def test_local_ffmpeg_failure_logs_sanitized_stderr(monkeypatch, caplog):
    """Step 9: when local ffmpeg processing fails, its stderr must be
    captured and logged (sanitized), so a failure is never just an opaque
    'ffmpeg exited with code N' with no context."""

    class _BadAudioYDL(_DownloadingYDL):
        def download(self, urls):
            path = self.opts["outtmpl"].replace("%(ext)s", "bin")
            with open(path, "wb") as fh:
                fh.write(b"not a real media file")

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _BadAudioYDL)
    with caplog.at_level("ERROR", logger="app.analysis.video_url"):
        with pytest.raises(VideoURLError):
            video_url.extract_audio_interval("https://www.youtube.com/watch?v=abc123", start_seconds=0.0, window_seconds=5.0)

    failure_records = [r for r in caplog.records if "VIDEO_URL_FAILURE" in r.message]
    assert failure_records
    assert "ffmpeg_stderr=" in failure_records[0].message


def test_media_decode_error_carries_stderr_without_changing_message():
    exc = video_url.MediaDecodeError("safe message", stderr="raw ffmpeg stderr detail")
    assert str(exc) == "safe message"
    assert exc.stderr == "raw ffmpeg stderr detail"


# --------------------------------------------------------------------------
# Duration cap enforced before Stage A (native download) starts
# --------------------------------------------------------------------------


def test_extract_audio_interval_rejects_known_long_duration_before_download(monkeypatch):
    called = {"download": False}

    class _ShouldNotBeCalledYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            called["download"] = True

    monkeypatch.setattr(video_url.yt_dlp, "YoutubeDL", _ShouldNotBeCalledYDL)
    with pytest.raises(VideoURLError) as excinfo:
        video_url.extract_audio_interval(
            "https://www.youtube.com/watch?v=abc123",
            start_seconds=0.0,
            window_seconds=5.0,
            known_duration_seconds=video_url.MAX_SOURCE_DURATION_SECONDS + 1,
        )
    assert excinfo.value.category == video_url.SOURCE_UNAVAILABLE
    assert called["download"] is False


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
