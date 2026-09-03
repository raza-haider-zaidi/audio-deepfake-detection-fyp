# Input Sources — Multimodal Audio Input Architecture

Branch: `feat/spectra-streamlit-candidate`. `main` is unmodified.

This document describes the input side of the "multimodal audio inputs
and premium reporting" phase: four supported input sources (Audio File,
Microphone, Voice Note, Video Audio) that all converge on the SAME frozen
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

All four adapters are chosen from a single segmented-control source
selector on the Analyze page (`app/views/analyze.py`), not four
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

## 6. Unified result model

Regardless of source, `PredictionResult`, segment evidence, audio
diagnostics, Analysis Session entries, and reports all carry a
`source_type` (`audio_file` / `microphone` / `voice_note` / `video_audio`)
and render a "SOURCE" badge/label. See `docs/reporting.md` for how source
type flows into the PDF/HTML/JSON report.

## 7. Security / content handling

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
