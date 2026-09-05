# Input Sources — Multimodal Audio Input Architecture

Branch: `feat/spectra-streamlit-candidate`. `main` is unmodified.

This document describes the input side of the "multimodal audio inputs
and premium reporting" phase and its "public video URL audio analysis"
follow-up: five supported input sources (Audio File, Microphone, Voice
Note, Video Audio, Video URL) that all converge on the SAME frozen
detector via one unified input-adapter architecture. It complements
`docs/analysis_platform.md` (scientific/architecture baseline) and
`docs/analysis_platform_v2.md` (visual/navigation baseline) rather than
replacing either.

**The detector is frozen.** Nothing in this document changes the
Spectra-AASIST3 model, INT8 artifact, Hugging Face revision, SHA256,
calibrated threshold, binary decision rule, inconclusive presentation
rule, preprocessing, segmentation, or aggregation. Every input source
below is an INPUT ADAPTER ONLY — it ends by handing a decoded, normalized
`AudioSample` to the exact same `detector.predict()` /
`detector.predict_full_clip()` calls already used for a plain WAV upload.

## 1. Input types

| Source | Selector label | Adapter | Extensions accepted |
|---|---|---|---|
| Audio file | Audio File | `app.analysis.input_sources.from_uploaded_file` | WAV, MP3, FLAC |
| Microphone | Microphone | `app.analysis.input_sources.from_microphone` | `st.audio_input` (native Streamlit WAV capture) |
| Voice note | Voice Note | `app.analysis.input_sources.from_voice_note` | WAV, MP3, FLAC (native) + M4A, AAC, OGG, OPUS, WEBM (ffmpeg) |
| Video audio | Video | `app.analysis.input_sources.from_video` | MP4, MOV, MKV, WEBM (audio track only) |
| Video URL | Video URL | `app.analysis.input_sources.from_video_url` | Public YouTube links (watch, Shorts, youtu.be) — audio track only |

All five adapters are chosen from a single segmented-control source
selector on the Analyze page (`app/views/analyze.py`), not five
simultaneous upload boxes.

## 2. Normalization flow

Every adapter returns a `NormalizedAudioInput`
(`app/analysis/input_sources.py`): `source_type`, `display_filename`, an
`AudioSample`, a SHA-256 of the original bytes, file size, and a
source-specific `source_metadata` dict. Downstream code (inference,
segment evidence, audio diagnostics, session history, reporting) consumes
only this one type and never branches on which adapter produced it — one
analysis code path, not four.

```
Audio file upload ─────────────┐
Microphone (st.audio_input) ───┤
Voice note upload ─────┐       │
Video upload ── ffmpeg  │       │
  (interval extraction) │       │
                         ▼       ▼
        ffmpeg → 16kHz mono WAV → audio_deepfake_detector.preprocessing
                                   .audio_loader.load_audio_file()  (UNCHANGED)
                                            │
                                            ▼
                              AudioSample (mono, 16 kHz, float32)
                                            │
                                            ▼
                         detector.predict() / predict_full_clip()  (FROZEN)
```

WAV/MP3/FLAC (audio file, voice note, microphone) decode via the
project's existing `soundfile`-based loader, unchanged. M4A/AAC/OGG/
OPUS/WEBM voice notes and every video's audio track are decoded via
`app/analysis/media_ffmpeg.py`, which invokes `ffmpeg` with an explicit
argument list (never `shell=True`) to produce a temporary 16 kHz mono WAV
file, then hands that WAV to the SAME unmodified `load_audio_file()` —
so ffmpeg-decoded audio goes through identical mono/resample/finite-value
validation as a native upload. No preprocessing logic is duplicated.

## 3. Microphone / acoustic capture

Implemented with Streamlit's native `st.audio_input(label, sample_rate=16000, ...)`
(confirmed present via `inspect.signature` on the installed Streamlit
1.62.0 — no custom WebRTC stack was needed). It requests 16 kHz capture
where the browser supports it, but the returned WAV bytes still go
through the full normalization pipeline above — a 16 kHz capture is never
treated as "already ready" and given a shortcut path.

Flow: record → stop → **playback preview is shown automatically by the
widget** → recording metadata is shown → user clicks **Analyze
Recording** (a microphone-specific button label; the shared audio-file
button reads "Analyze Audio"). Analysis never starts automatically on
recording completion. Microphone access is only requested when the user
opens the Microphone tab and interacts with the widget — never at page
load. An "Acoustic capture advisory" is shown after analysis: room
acoustics, playback equipment, microphone processing and background noise
can affect detector output; this is presentation-only text and never
changes the classifier.

### Limitations

- A recording under 0.5 seconds is rejected with a friendly error before
  reaching the detector.
- "Live Detection" is intentionally not used as a label anywhere — there
  is no streaming inference; capture and analysis are two explicit,
  separate steps.
- Whether the browser actually honors the requested 16 kHz capture rate
  is browser/OS-dependent; the app does not assume it and always
  re-normalizes.

## 4. Video audio analysis

**This is audio-track analysis only.** No face, lip-sync, or visual
deepfake detection is performed anywhere in this project — video frames
are never decoded for analysis, only probed for container metadata and
discarded.

Flow: upload → `ffprobe` reports real container/codec/duration (never
guessed) → the video previews via `st.video()` → if the video is longer
than the detector's supported analysis duration (30 seconds), the user
selects a start time with a slider and a 30-second window is computed →
`ffmpeg` extracts and normalizes **only that window** (`-ss <start> -t
<window> -vn -ac 1 -ar 16000`) → the extracted window goes through the
same `load_audio_file()` path as everything else. A video shorter than
30 seconds is analyzed in full (no slider is shown). No arbitrary or
undisclosed section of a long video is ever silently chosen.

- Max video file size: **100 MB** (`app.validation.MAX_VIDEO_FILE_SIZE_BYTES`) —
  set in both `.streamlit/config.toml` (`maxUploadSize`) and app-level
  validation. This is larger than the 25 MB audio/voice-note/microphone
  cap because video containers are inherently larger, even though only a
  short audio window is ever analyzed.
- Max analyzed audio duration: **30 seconds**
  (`app.validation.MAX_VIDEO_ANALYSIS_WINDOW_SECONDS`), matching the
  detector's existing analysis duration everywhere else in the app.
- Supported containers: MP4, MOV, MKV, WEBM. AVI is intentionally **not**
  listed as supported — its decoding reliability was not verified in this
  environment.
- A video with no audio stream is rejected with a friendly error before
  extraction is attempted (`ffprobe` is authoritative on this, never
  guessed).

## 5. Voice notes

Supported: **WAV · MP3 · FLAC** (native) and **M4A · AAC · OGG · OPUS ·
WEBM** (ffmpeg-decoded) — all verified against the installed ffmpeg
build in this environment (see "Manual media QA" in the phase's final
report). **AMR is intentionally not listed as supported** — its decoding
reliability was not verified here, and this project does not label an
unverified format as supported.

No direct integration with WhatsApp, Telegram, or any other messaging
platform exists or is implied — the user exports/saves the voice note
from wherever it originated and uploads the resulting file, like any
other upload.

Normalization: identical size (25 MB) and duration (30 s) limits as a
regular audio upload. After decoding, the file is shown as "Normalized
for analysis: Mono · 16 kHz" — this description is purely descriptive of
what preprocessing did, and does not imply that normalization improves or
changes authenticity detection.

## 6. Video URL (public online video audio)

Lets a user paste a link to a **publicly accessible YouTube video**
(standard watch links, Shorts, `youtu.be`) and analyze a selected portion
of its audio track — the SAME frozen detector, the SAME normalization
pipeline. **Only YouTube is supported in this phase** — other public
video platforms are intentionally not claimed as supported, since their
reliability against the installed `yt-dlp` build has not been verified
here (the same "never label an unverified format as supported" principle
already applied to voice notes and video containers).

**This is audio-track analysis only.** No frame of video is ever
retrieved or decoded for analysis — `app/analysis/video_url.py` only
resolves and extracts the audio stream.

### Flow

1. The user pastes a URL and clicks **Load Video**. This retrieves real
   title/duration/uploader/thumbnail metadata via `yt-dlp`
   (`skip_download=True` — no media is transferred at this step).
2. If the video is longer than the detector's 30-second analysis window,
   the user selects a start time with a slider; otherwise the full (short)
   video is used.
3. The user clicks **Prepare Selected Audio**. Only then does the app
   retrieve media — and only the selected interval's audio, not the whole
   video (see "Minimizing media transfer" below).
4. The extracted audio converges on the exact same
   `ffmpeg → load_audio_file()` path used by voice notes and video
   uploads, then goes through the unmodified frozen detector.

### Supported URL forms

`https://www.youtube.com/watch?v=...`, `https://youtu.be/...`, and
`https://www.youtube.com/shorts/...` (and the `m.youtube.com` /
`youtube-nocookie.com` variants) are all recognized as the same YouTube
extractor — Shorts are not rejected merely because their route differs
from a standard watch link.

### Reliability: a convenience adapter, not a dependency

Online video ingestion is explicitly a convenience input adapter. Audio
File, Microphone, Voice Note, and Video File analysis do not import from
`app/analysis/video_url.py` and are unaffected if YouTube changes its
delivery behavior or `yt-dlp` needs to be updated. Any retrieval failure —
unavailable, deleted, private, age-restricted, geo-restricted, anti-bot
blocked, or a network timeout — surfaces the same professional message
("Unable to access this video...") with no extractor stack trace ever
shown to the user or left in application logs (a silent `yt-dlp` logger
routes its internal messages to debug-level Python logging instead of
stdout/stderr).

### JavaScript runtime and format-selection robustness

Modern YouTube extraction requires solving a JavaScript-based signature
challenge for most formats. Without a JS runtime available, `yt-dlp` logs
"YouTube extraction without a JS runtime has been deprecated" and falls
back to a more limited player client — one more likely to be rejected or
rate-limited, especially from a datacenter/cloud egress IP (this was the
observed root cause of a real failure: metadata succeeded via the
lightweight client, but the "Prepare Selected Audio" download step failed).
`requirements.txt` therefore pins `yt-dlp[default,deno]` — the `deno`
extra bundles a self-contained Deno binary (current `yt-dlp` guidance's
preferred JS-challenge runtime; see
https://github.com/yt-dlp/yt-dlp/wiki/EJS) via a pure pip wheel, requiring
no system package manager step, so it installs unmodified on Streamlit
Community Cloud; the `default` extra pulls `yt-dlp-ejs` (the EJS
challenge-solver scripts) plus the small pure-Python libraries current
`yt-dlp` expects to have available. Format selection remains
`"bestaudio/best"` (never a brittle single-container selector like m4a-only
or mp3-only) — ffmpeg normalizes whatever container is retrieved.

**Documented limitation:** a PO (proof-of-origin) token requirement was
not observed against the tested public videos in this project's local
environment; if YouTube begins requiring one from a given deployment's
egress IP, extraction will surface the same safe "YouTube did not permit
the application server to retrieve this video's audio" message rather
than a raw error, but audio retrieval for the affected video will fail
until (if ever) a PO-token provider is evaluated and added — no such
provider is installed today, per the explicit "do not add unless actually
necessary" guidance.

### Failure classification

Every URL-ingestion failure is classified internally (never shown
verbatim to the user) into one of: `URL_INVALID`, `SOURCE_UNAVAILABLE`,
`NO_AUDIO_STREAM`, `YOUTUBE_BOT_CHALLENGE`, `JS_RUNTIME_UNAVAILABLE`,
`PO_TOKEN_REQUIRED`, `FORMAT_UNAVAILABLE`, `NETWORK_TIMEOUT`,
`EXTRACTION_FAILED`, `FFMPEG_FAILED` (see
`app/analysis/video_url.py::classify_extraction_error`, a substring-based
heuristic over the raw — never displayed — exception text). Each category
maps to a short, professional, traceback-free message; most fall back to
the generic "Unable to access this video..." message, while a few get a
more specific one (e.g. missing audio track, or a YouTube-side
restriction). Classification errors are harmless — every category still
resolves to a safe message.

### Developer-only diagnostics

`app/analysis/video_url.py::diagnostics_snapshot()` reports the installed
`yt-dlp`/`yt-dlp-ejs` versions, `ffmpeg` availability, the detected JS
challenge runtime, available JS-challenge provider names, and PO-token
provider availability — for operator troubleshooting only. It is never
called from the normal Streamlit UI; `is_debug_enabled()` gates any future
debug surface behind the `ADF_VIDEO_URL_DEBUG` environment variable, off
by default.

### Public content only

No login, cookie import, or account-credential support of any kind exists
for this feature. Only media already accessible to the server as public
content is retrieved — private videos, age-gated content requiring
sign-in, and geo-restricted content are not and cannot be bypassed.

### Minimizing media transfer

Format selection is restricted to audio-only (`bestaudio/best` — the
video stream is never requested), and `yt-dlp`'s `download_ranges` /
`force_keyframes_at_cuts` options are used so that, for formats that
support ranged retrieval (YouTube's DASH audio streams typically do),
only approximately the selected interval is actually transferred rather
than the full source. Verified manually: extracting a 10-second window
from a public video transferred under 100 KB. **Documented limitation:**
for a format that does not support ranged retrieval, `yt-dlp` may need to
retrieve more than the selected interval before extraction; a
`max_filesize` ceiling (100 MB, matching the video-file upload cap) and a
socket timeout bound the worst case.

### URL security / SSRF protection

Because this feature accepts arbitrary user-entered URLs,
`app/analysis/video_url.py::validate_public_video_url` runs BEFORE any
network request:

- Only `http`/`https` schemes are accepted — `file://`, `ftp://`, `data:`,
  and `javascript:` are rejected outright.
- Only a hostname allow-list (`youtube.com`, `youtube-nocookie.com`,
  `youtu.be`, and their subdomains) is accepted — every other domain is
  rejected before any DNS lookup or request.
- The hostname is then resolved and every returned IP address is checked
  against `ipaddress`'s private/loopback/link-local/multicast/reserved/
  unspecified predicates — rejecting `localhost`, `127.0.0.0/8`, `::1`,
  private network ranges, link-local ranges (including the
  `169.254.169.254` cloud metadata address), before any request is made.
  This is defense-in-depth on top of the domain allow-list, not a general
  arbitrary-URL SSRF proxy.
- A `socket_timeout` and a `max_filesize` ceiling bound every request.
- **Documented limitation:** per-redirect-hop IP re-validation is not
  separately implemented — `yt-dlp` manages its own request/redirect
  handling internally; the domain allow-list plus the upfront DNS/IP check
  is the primary mitigation.
- Every `yt-dlp`/`ffmpeg` call uses the Python API / explicit argument
  lists — never `shell=True`, never a string-concatenated command with a
  user-supplied URL.

### Source metadata and identity

`source_type = "video_url"` carries `platform`, `source_title`,
`source_url`, `source_uploader`, `video_duration_seconds`, and
`selected_interval` in `source_metadata`. The recorded SHA-256 is always
computed from the **actual extracted audio bytes**, never from the URL —
the URL is not treated as a proxy for content identity.

### Privacy

For URL analysis, the application retrieves only the public media
required for the selected analysis interval and temporarily processes its
audio. The retrieved media is never permanently archived, and no history
of submitted URLs is stored beyond the current session.

## 7. Unified result model

Regardless of source, `PredictionResult`, segment evidence, audio
diagnostics, Analysis Session entries, and reports all carry a
`source_type` (`audio_file` / `microphone` / `voice_note` / `video_audio` /
`video_url`) and render a "SOURCE" badge/label. See `docs/reporting.md`
for how source type flows into the PDF/HTML/JSON report.

## 8. Security / content handling

- Filenames are never used to construct file-system paths or shell
  commands directly — `tempfile.mkstemp`/`NamedTemporaryFile` generate
  the actual on-disk names; the user's filename is only used for display
  and extension detection.
- Every `ffmpeg`/`ffprobe` invocation uses an explicit Python list of
  arguments (`subprocess.run([...])`), never `shell=True` and never a
  string-concatenated command — a filename cannot be interpreted as shell
  syntax.
- Uploaded/recorded media is never executed.
- Actual decodability (via `ffprobe`/`ffmpeg`'s own success/failure) is
  authoritative — an extension or claimed MIME type alone is never
  trusted as proof a file is valid media.
- All intermediate files (extracted WAVs, transcoded voice notes) are
  written under the OS temp directory and removed in a `finally` block
  immediately after use — never under the project directory, never
  retained across requests.
