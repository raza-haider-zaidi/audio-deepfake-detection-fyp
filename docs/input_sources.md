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

**Deployment note:** the deployed Streamlit application now serves this
feature through a separately-run **local URL ingestion helper** rather
than calling `yt-dlp` from the Streamlit process itself — see
`docs/local_url_helper.md` for why and how. Everything below (`app/
analysis/video_url.py`, the PO-token provider, etc.) remains fully
accurate as the underlying retrieval logic — it is what the helper
(`local_helper/youtube.py`) calls internally, and what local development/
testing still exercises directly — but the Streamlit UI
(`app/views/analyze.py`) no longer calls it directly; it calls
`app/analysis/url_helper_client.py` instead, which talks HTTPS to the
connected helper.

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
("Unable to access this video...") to the user, with no extractor stack
trace ever shown in the UI. The real, sanitized failure IS written to the
server-side application log (never the UI) — see "Cloud diagnostics
logging" below; this is deliberate, so a real production failure is
diagnosable from Streamlit Cloud's server logs.

### Native download / local interval extraction split

A second real Streamlit Cloud production failure occurred after the JS
runtime fix above: metadata retrieval, Deno, `yt-dlp-ejs`, and `ffmpeg`
were all confirmed present and working, yet audio retrieval failed with
`ERROR: ffmpeg exited with code 8` (`VIDEO_URL_FAILURE ... category=
FFMPEG_FAILED`). Root cause, found by inspecting the exact `yt-dlp`
options in `extract_audio_interval`: `download_ranges` combined with
`force_keyframes_at_cuts=True` makes `yt-dlp` invoke `ffmpeg` itself as an
**external downloader**, running it directly against the **remote signed
googlevideo.com media URL** to cut the requested interval during
download — not against a local file. This worked on local Windows
development but failed against Streamlit Cloud's `ffmpeg` build/sandbox
network behavior.

Fixed by splitting retrieval into two independent stages, so `ffmpeg` is
never used as a remote downloader again:

- **Stage A — native download** (`stage=native_audio_download` in logs):
  `yt-dlp`'s own native (non-ffmpeg) downloader retrieves the full
  audio-only stream (`format: "bestaudio/best"`, never the video track) to
  a local temporary file. `download_ranges` and `force_keyframes_at_cuts`
  are deliberately absent from the options.
- **Stage B — local interval extraction** (`stage=local_interval_extract`
  in logs): once the audio track is a **local file**,
  `app.analysis.media_ffmpeg.extract_video_audio_window` — the same
  shared, unmodified local-file interval-extraction/normalization utility
  the Video File source already uses — cuts the selected
  `[start, start+window)` interval and normalizes it. `ffmpeg` only ever
  receives a local filesystem `-i` path in this stage; a regression test
  (`tests/test_video_url.py::test_local_interval_extraction_ffmpeg_input_is_a_local_file`)
  asserts this directly, and another
  (`test_native_download_opts_never_configure_ffmpeg_as_remote_downloader`)
  asserts `download_ranges`/`force_keyframes_at_cuts`/`external_downloader`
  are never present in the Stage A options, so this failure mode cannot
  silently return.

**Trade-off, accepted deliberately:** for a long source video, Stage A now
downloads its full audio-only track before Stage B cuts out the selected
interval, rather than the old (broken-on-Cloud) approach of asking the
remote server to serve only the requested byte range. Correctness and
Streamlit Cloud reliability were prioritized over minimizing transfer for
this phase. To keep worst-case download size and time bounded,
`MAX_SOURCE_DURATION_SECONDS` (the "video too long to analyze via a public
link" cutoff) was tightened from 4 hours to **30 minutes** — comfortably
under `MAX_URL_DOWNLOAD_BYTES` (100 MB) for typical YouTube audio-only
bitrates, and checked both by `fetch_metadata` and again by
`extract_audio_interval` (via a `known_duration_seconds` parameter passed
from the already-fetched metadata) before Stage A starts, so an
unexpectedly long source is rejected before any download begins. A future
phase could revisit partial/ranged native downloading (without invoking
ffmpeg as the downloader) if bandwidth minimization becomes a priority
again.

**Failure boundaries:** `NATIVE_DOWNLOAD_FAILED` is the fallback category
for an unrecognized Stage A failure (a specific failure like
`YOUTUBE_BOT_CHALLENGE`/`PO_TOKEN_REQUIRED`/`JS_RUNTIME_UNAVAILABLE`/
`FORMAT_UNAVAILABLE`/`NETWORK_TIMEOUT`/`SOURCE_UNAVAILABLE` is still
classified more specifically when the raw error text matches).
`FFMPEG_LOCAL_PROCESSING_FAILED` is used only for a Stage B failure — a
native download that succeeded but whose local `ffmpeg` processing then
failed. Keeping these separate means a log line always tells you which
half of the pipeline actually failed. When Stage B fails, the raw
(sanitized) `ffmpeg`/`ffprobe` stderr is included in the
`VIDEO_URL_FAILURE` log line as `ffmpeg_stderr=...` (via
`MediaDecodeError.stderr`, populated in `app/analysis/media_ffmpeg.py`),
so a failure is never just an opaque "ffmpeg exited with code 8" without
context — while the UI-facing message stays the same short, generic text.

### Cloud diagnostics logging

A first deployment of this feature to Streamlit Community Cloud produced
metadata successfully but failed at audio retrieval with only the generic
UI message — and nothing useful in the server logs. Root cause: the
internal yt-dlp logger adapter (`_SilentYDLLogger`) routed `warning()` and
`error()` calls to Python's `logging.debug()`, and the classified-failure
log line was also emitted at debug level — so real WARNING/ERROR-level
yt-dlp diagnostics (JS-runtime status, HTTP errors, bot-challenge/PO-token
messages, format failures) were silently dropped by the default log level
before ever reaching Streamlit Cloud's log output.

Fixed by giving `_SilentYDLLogger.warning()`/`.error()` their matching
real log levels, and by logging two explicit, sanitized, server-log-only
lines from `app/analysis/video_url.py`:

- `VIDEO_URL_DIAGNOSTIC: yt_dlp=<version> deno=<runtime or unavailable> ejs=<version or unavailable> ffmpeg=<available/unavailable> platform=youtube video_id=<id> requested_interval=<start>-<end>` —
  emitted immediately before every real audio-retrieval attempt.
- `VIDEO_URL_FAILURE stage=<metadata|native_audio_download|native_audio_download_pot_retry|local_interval_extract> category=<...> exception_type=<...> sanitized_error=<...> [ffmpeg_stderr=<...>]` plus a sanitized traceback —
  emitted whenever metadata retrieval, the Stage A native download (first
  attempt or the PO-token provider-backed retry), or the Stage B local
  ffmpeg interval extraction fails (see "Native download / local interval
  extraction split" and "Proof-of-Origin token support" below), using the
  same failure categories as the UI-facing classifier.
- `VIDEO_URL_POT_DIAGNOSTIC provider=<name>-<mode> provider_available=<yes/no> player_client=mweb token_generated=<yes/no/not_attempted>` —
  emitted after a provider-backed retry is attempted (or immediately, if
  it could not even start) — see "Proof-of-Origin token support" below.

Verbose yt-dlp DEBUG-level chatter (including yt-dlp's own internal
JS-runtime/player-client detection lines, e.g. "JS runtimes: deno-2.9.6")
is only elevated to INFO when the `ADF_VIDEO_URL_DEBUG=1` environment
variable is set on the deployment — this can be toggled on temporarily on
Streamlit Cloud to see exactly which JS-challenge runtime and player
client yt-dlp resolved for a specific failing video, without leaving
verbose logging on by default.

**Sanitization:** `_sanitize_log_text()` redacts any URL whose hostname
looks like a signed googlevideo.com media URL (kept only as
`https://<host>/[signed-media-url-redacted]`) and any `sig=`/`signature=`/
`token=`/`po_token=`/`auth*=`/`cookie=` query parameter anywhere in the
logged text, plus (per "Proof-of-Origin token support" below)
`visitorData`/`dataSyncId`/`rolloutToken`/`deviceExperimentId`/`poToken`/
`integrityToken` JSON fields, then truncates to a bounded length. Plain, unsigned URLs
(e.g. the public watch/Shorts page URL itself) are left intact so a log
line can still identify which video failed. The full `yt-dlp` info
dictionary is never logged; a video is identified in logs by
platform + video ID (extracted from the URL with a local regex, no
network call), never by dumping a signed stream URL. Streamlit secrets,
cookies, and auth headers are never read or logged by this module at all
— it has no code path that touches them.

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

**Update:** a real Streamlit Cloud deployment DID subsequently hit exactly
this condition (`HTTP Error 403: Forbidden` from the native downloader,
with the two-stage architecture above otherwise confirmed working
correctly) — see "Proof-of-Origin token support" below for the
provider-backed fallback this project added in response, once concrete
evidence (not speculation) showed one was actually needed.

### Proof-of-Origin (PO) token support

**Why:** current yt-dlp guidance
(https://github.com/yt-dlp/yt-dlp-wiki/blob/master/PO%20Token%20Guide.md,
verified against the live upstream page, not memory) documents that
YouTube increasingly requires a GVS (Google Video Server) PO Token for
media requests from the `mweb` client, and that missing one can produce
`HTTP 403`. This project's own Streamlit Cloud deployment hit exactly that
403 on the native download stage, with every other part of the two-stage
architecture (native download, local ffmpeg interval extraction) already
confirmed working — see the failure this section's provider was added to
resolve, above.

**Policy (unchanged, see docs/development_policy.md and the module docstrings):** public
content only. This provider path never uses cookies, a Google account,
OAuth, or a proxy — only the same anonymous PO-token generation path
unauthenticated `mweb` playback already uses. It was added only after a
real, observed 403 proved one was needed — not speculatively.

**Provider chosen:** the maintained
`bgutil-ytdlp-pot-provider` (https://github.com/Brainicism/bgutil-ytdlp-pot-provider),
pinned at release **1.3.2** (GPL-3.0-only), using its **Deno script mode**
(no persistent server process, no Docker sidecar — Streamlit Community
Cloud provides neither; see `app/analysis/pot_provider.py`). Two halves:

- The yt-dlp-side plugin is installed via the pinned PyPI package
  `bgutil-ytdlp-pot-provider==1.3.2` in `requirements.txt`.
- The actual token-generation source (a small Deno/TypeScript program,
  `server/` from the same upstream release) is **vendored** at
  `third_party/bgutil-ytdlp-pot-provider/server/` — see
  `third_party/bgutil-ytdlp-pot-provider/VENDORED.md` for the exact pinned
  commit, license, and attribution. Vendoring (rather than a deploy-time
  `git clone`) is deterministic and matches this project's existing
  "pin a release, never track master" policy — it is also the only option,
  since Streamlit Community Cloud has no arbitrary build/postBuild hook,
  only `requirements.txt` (pip) and `packages.txt` (apt).

**Native-dependency install:** the vendored server's `canvas` npm package
(used by its BotGuard-interfacing library) is a native module. It is
installed lazily, via `deno install --allow-scripts=npm:canvas,npm:@swc/core
--frozen`, **at most once per process** (`pot_provider.ensure_provider_ready`,
memoized) — only the first time a plain (non-provider) retrieval attempt
is declined by YouTube, never at import time, app startup, or on every
request. `canvas` ships prebuilt binaries for common platforms (confirmed
locally on Windows: the install completed in ~20s using a downloaded
prebuilt binary, no compiler invoked); whether a prebuilt binary also
exists for Streamlit Community Cloud's exact Linux container (and so
whether any apt build toolchain is even needed) is **unverified** from
this sandbox — the local proof only shows a real GVS PO token can be
generated and used end-to-end on this machine (see "Local verification"
below).

**`packages.txt` deliberately stays minimal (`ffmpeg` only).** No apt
build-toolchain packages (`build-essential`, `libcairo2-dev`, etc.) are
added preemptively for `canvas` — if the lazy install ever fails on Cloud
for lack of a compiler/library, `ensure_provider_ready` returns
`(False, <reason>)` and the feature degrades gracefully to
`PO_TOKEN_PROVIDER_UNAVAILABLE` (the safe upload-fallback message) rather
than crashing the app, so there is no correctness reason to add packages
speculatively. **Lesson learned the hard way:** an earlier version of this
file added those packages WITH explanatory `#` comments directly in
`packages.txt`; Streamlit Community Cloud's packages.txt installer does
not strip `#` comments the way plain apt does, and treated every word of
the comment text as a literal package name ("E: Unable to locate package
Runtime", "... package `canvas`", etc.), aborting the entire dependency
install and breaking deployment completely (not just PO-token support) --
fixed in commit that follows 53e7845. `packages.txt` must therefore
contain ONLY real package names and nothing else, ever again (enforced by
`tests/test_deployment_dependencies.py::
test_packages_txt_lines_look_like_valid_apt_package_names`); if a real
Cloud deployment log later shows `canvas` failing to compile for lack of a
specific library, add that exact package (verified against the log, not
guessed) as its own commit.

**Player client and extractor args:** `pot_provider.provider_extractor_args()`
returns `{"youtube": {"player_client": ["mweb"]}, "youtubepot-bgutilscript":
{"server_home": [<absolute path>]}}` — `mweb` per current upstream
guidance, and an absolute `server_home` derived from this repository's own
location (`Path(__file__).resolve().parents[2] / "third_party" / ...`),
never a `~`-relative path (Streamlit Cloud's home-directory layout is not
something this project depends on).

**Retry policy:** at most **one** provider-backed retry, and only after a
plain attempt fails with a category that indicates YouTube itself declined
the request (`YOUTUBE_BOT_CHALLENGE`, `PO_TOKEN_REQUIRED`,
`YOUTUBE_MEDIA_FORBIDDEN`) — never for network/format/ffmpeg failures the
provider cannot fix, and never more than once (avoids worsening any
IP-level rate-limiting). Most videos succeed on the first, provider-free
attempt, so the (slow, one-time) native-dependency install is only ever
triggered when actually needed.

**Failure classification** (see `app/analysis/video_url.py`): an `HTTP 403`
is no longer collapsed into the generic `NATIVE_DOWNLOAD_FAILED` fallback:

- `YOUTUBE_MEDIA_FORBIDDEN` — a 403 occurred (pattern-matched from the
  exception text, e.g. `"403"` + `"forbidden"`).
- `PO_TOKEN_PROVIDER_UNAVAILABLE` — the provider-backed retry could not
  even run (vendored server directory, yt-dlp plugin, or Deno missing, or
  the one-time native-dependency install failed).
- `PO_TOKEN_GENERATION_FAILED` — the provider ran but yt-dlp never
  reported generating a token for this video (detected via yt-dlp's own
  `"Retrieved a gvs PO Token for <client> client"` debug line — see
  `_ProviderAttemptLogger`).
- `YOUTUBE_DATACENTER_BLOCK` — used **only** when a token was confirmed
  generated for this exact video and the media request still returned
  403. This is never inferred from a bare 403 alone (per the current
  guidance's own caution against over-claiming datacenter blocking) — it
  requires positive evidence the token was not the problem.

All four map to the same user-facing message: *"Online video audio could
not be retrieved from this hosting environment. You can still analyze the
recording by uploading the audio or video file directly."* — never a claim
of "bot detection" unless yt-dlp's own text explicitly said so
(`YOUTUBE_BOT_CHALLENGE`).

**Diagnostics:** a `VIDEO_URL_POT_DIAGNOSTIC provider=<name>-<mode>
provider_available=<yes/no> player_client=mweb
token_generated=<yes/no/not_attempted>` line is logged server-side after
every provider-backed retry attempt (or immediately, if the retry could
not even start) — like every other log line in this module, it NEVER
includes the token value, visitor ID, data-sync ID, rollout token, or
device experiment ID (`_sanitize_log_text`'s `_SENSITIVE_JSON_FIELD_RE`
redacts these specifically, since yt-dlp's own verbose debug output
embeds them in the innertube client-context JSON passed to the
token-generation subprocess).

**Local verification performed:** the vendored `server/` script was run
directly (`deno run --allow-all src/generate_once.ts --content-binding
<video-id>`) and produced a real GVS PO token; a full `yt_dlp.YoutubeDL`
extraction with `player_client=mweb` + the script-mode provider configured
retrieved real metadata and logged yt-dlp's own `"Retrieved a gvs PO Token
for mweb client"` line; the full `extract_audio_interval` retry path was
exercised (with a monkeypatched first-attempt failure) end-to-end,
confirming exactly one retry, the correct extractor args on that retry,
and correct sanitized logging.

**What is NOT yet verified (requires an actual Streamlit Cloud
deployment):**

1. Whether `canvas`'s native install succeeds via a prebuilt binary on
   Streamlit Community Cloud's specific container image with only
   `packages.txt`'s `ffmpeg` present (no build toolchain) -- if not, the
   exact missing package must be read from that failure's log line and
   added deliberately, never guessed in advance (see "Native-dependency
   install" above for why no build-toolchain packages are pre-added).
2. Whether the one-time `deno install` completes within Streamlit Cloud's
   free-tier CPU/memory/time limits on a cold container.
3. Whether a PO token generated this way is actually accepted by YouTube
   from Streamlit Cloud's egress IP (the local test above proves token
   *generation* works; it cannot prove YouTube's *server-side acceptance*
   of a Cloud-originated request, which is exactly what the original 403
   was about).
4. If a token is confirmed generated on Cloud and 403 still persists —
   per this project's own escalation policy, that is the trigger to stop
   and report Streamlit Cloud's network/IP as likely unsuitable for
   reliable YouTube media retrieval, rather than reaching for a proxy,
   cookies, or a Google account.

### Failure classification

Every URL-ingestion failure is classified internally (never shown
verbatim to the user) into one of: `URL_INVALID`, `SOURCE_UNAVAILABLE`,
`NO_AUDIO_STREAM`, `YOUTUBE_BOT_CHALLENGE`, `JS_RUNTIME_UNAVAILABLE`,
`PO_TOKEN_REQUIRED`, `FORMAT_UNAVAILABLE`, `NETWORK_TIMEOUT`,
`EXTRACTION_FAILED`, `FFMPEG_FAILED`, `NATIVE_DOWNLOAD_FAILED`,
`FFMPEG_LOCAL_PROCESSING_FAILED`, `YOUTUBE_MEDIA_FORBIDDEN`,
`PO_TOKEN_PROVIDER_UNAVAILABLE`, `PO_TOKEN_GENERATION_FAILED`,
`YOUTUBE_DATACENTER_BLOCK` (the last four are described in detail under
"Proof-of-Origin token support" above; see
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
video stream is never requested). **Superseded by the Cloud-reliability
fix below:** an earlier version of this module used `yt-dlp`'s
`download_ranges` / `force_keyframes_at_cuts` options so that only
approximately the selected interval was transferred; that approach made
`yt-dlp` invoke `ffmpeg` as an external downloader directly against the
remote signed media URL, which failed on Streamlit Cloud (see "Native
download / local interval extraction split"). The full audio-only track
is now downloaded natively before the selected interval is cut locally —
**documented trade-off:** for a long source video this means more data is
transferred than the selected interval alone would require. A
`max_filesize` ceiling (100 MB, matching the video-file upload cap), a
socket timeout, and a tightened 30-minute maximum source duration
(`MAX_SOURCE_DURATION_SECONDS`) bound the worst case.

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
