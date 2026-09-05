# Audio Deepfake Detection

**Detecting AI-Generated and Cloned Voices: A Deep Learning System for Robust Audio Deepfake Detection**

Final-year Computer Science project. Author: **Syed Raza Haider Zaidi**.

Branch: `feat/spectra-streamlit-candidate` — the deployed candidate described
in this README. (`main` tracks an earlier phase; see its own README.)

## Purpose

This project builds a system for detecting AI-generated and voice-cloned
audio (audio deepfakes). It evaluates how well a pretrained deepfake
detection approach generalises across datasets and generation methods, how
robust it is to real-world signal degradation (compression, noise,
re-recording), and packages it as an interactive, browser-based analysis
tool.

## Features

- Binary bonafide/spoof classification with a calibrated decision threshold,
  per-segment evidence, and a plain-language result summary.
- Five input sources feeding the **same** frozen detection pipeline: Audio
  File, Microphone, Voice Note, Video (audio track only), and Video URL
  (public YouTube links, via the Local URL Helper — see below).
- Segment-level evidence table, audio-quality diagnostics, waveform/mel-
  spectrogram visualizations, and an optional robustness analysis
  (compression/noise/resampling degradations applied on demand).
- Downloadable PDF/HTML/JSON analysis reports.
- CPU-only inference — no GPU, no CUDA dependency, no dedicated inference
  server.

## Model

- **Architecture:** `spectra_aasist3_onnx_int8` — a dynamic INT8
  weight-quantized (ONNX Runtime `quantize_dynamic`, MatMul-only) export of
  an XLS-R-300M (`facebook/wav2vec2-xls-r-300m`) + KAN-enhanced AASIST
  back-end anti-spoofing model, quantized from a verified upstream FP32
  checkpoint (`lab260/Spectra-AASIST3`, Apache-2.0) with no retraining.
- **Runtime:** ONNX Runtime, CPU execution provider only.
- **Input:** 16 kHz mono audio, analyzed in a native ~4.04 s window
  (64,600 samples) per segment; up to 30 seconds of audio is analyzed per
  request across all input sources.
- **Decision threshold:** calibrated on a held-out split
  (`INT8_DYNAMIC_CALIBRATED_THRESHOLD = 0.939693808555603`); see
  `src/audio_deepfake_detector/models/candidate_e.py`.
- **License:** Apache-2.0 (see `configs/models.yaml` for the exact upstream
  and derived-artifact provenance, revision, and SHA-256).
- The upstream model is **pre-release/unpublished** — no peer-reviewed
  paper exists for it. Any EER/accuracy figures quoted from the model
  publisher are clearly labeled as author-reported, never presented as
  this project's own measured result. This project's own measured
  evaluation is in
  [`docs/spectra_aasist3_evaluation.md`](docs/spectra_aasist3_evaluation.md).

## Deployment architecture

The application is a **Streamlit app running entirely on CPU**:

- No CUDA/GPU dependency in the deployed path; `torch`/`torchaudio` (used
  only by research-only candidate adapters, not the deployed model) are
  installed from the official CPU wheel index when needed.
- No database, no user accounts, no persistent server-side storage of
  submitted media.
- Model weights are downloaded from Hugging Face at runtime (via
  `huggingface_hub`), verified by SHA-256 before use, and cached
  (`st.cache_resource`) — not committed to this repository.

### Local URL Ingestion Helper

Streamlit Community Cloud's datacenter network is not reliably accepted by
YouTube's media servers, even with a Proof-of-Origin token provider (see
[`docs/input_sources.md`](docs/input_sources.md)). Video URL analysis is
therefore served by a small, separately-run **local helper**
(`local_helper/`) on a machine where retrieval works normally, reached over
a temporary, authenticated Cloudflare Quick Tunnel. Full architecture,
privacy implications, and usage in
[`docs/local_url_helper.md`](docs/local_url_helper.md).

**To use Video URL analysis:**
1. Run `Raza Audio URL Helper.exe` (built via
   `scripts/build_local_helper.ps1`) on a machine with working internet
   access to YouTube.
2. Click **Copy Connection** once it shows **Ready**.
3. In the deployed app's Video URL source, paste the connection code and
   click **Connect**.
4. Use **Load Video** → select an interval → **Prepare Selected Audio** as
   normal.

Every other input source (Audio File, Microphone, Voice Note, Video) works
without the helper.

## Installation / local use

### Prerequisites
- Windows with Python 3.12
- Git
- FFmpeg on `PATH`
- No GPU required — everything here runs and is benchmarked on CPU.

### Set up the environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For local development of the research adapters (not needed to run the
deployed app), install the full package instead:
`.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`.

### Run the Streamlit app locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The page renders immediately; the detection model is downloaded/loaded
lazily the first time you click **Analyze Audio**, and cached for the rest
of the session.

### Run tests

```powershell
# Default suite — no real model downloads, safe/fast:
.\.venv\Scripts\python.exe -m pytest -v -m "not integration and not slow"

# Real model download/load/inference tests (network + large downloads):
.\.venv\Scripts\python.exe -m pytest -v -m "integration"
```

### Build the Local URL Helper

```powershell
.\scripts\build_local_helper.ps1
```

Produces `dist\Raza Audio URL Helper\Raza Audio URL Helper.exe` — see
[`docs/local_url_helper.md`](docs/local_url_helper.md).

## Repository structure

```
streamlit_app.py                    Streamlit entrypoint (Community Cloud runs this directly)
requirements.txt                    Root-level pinned dependencies for the deployed Streamlit app
requirements-helper.txt             Separate, smaller dependency set for local_helper/ only
packages.txt                        Debian apt packages for Streamlit Community Cloud (ffmpeg)
.streamlit/config.toml              Streamlit server/theme configuration (no secrets)
app/                                 Streamlit presentation-layer helpers (no model logic)
    analysis/                       Input adapters (audio file/mic/voice note/video/video URL), reporting inputs
    reporting/                      PDF/HTML/JSON report generation
    views/                          One render() function per page (app/nav.py registers them)
local_helper/                       Local URL Ingestion Helper (separate app, own dependency set)
src/audio_deepfake_detector/
    config/                         configs/models.yaml loader
    preprocessing/                  Audio loading (audio_loader.py), fixed-window framing (windowing.py)
    inference/                      Device-agnostic inference service
    models/                         BaseDeepfakeDetector interface, registry/factory, per-candidate adapters
    evaluation/                     Cross-dataset/robustness evaluation harnesses
    utils/                          Typed data structures (datatypes.py), device resolution (device.py)
tests/                              Unit tests (default suite) + integration/slow-marked real-model tests
scripts/                            Benchmark/evaluation harnesses, build_local_helper.ps1
configs/                            models.yaml — single source of truth for model repos/revisions/labels
third_party/                        Vendored PO-token provider source (pinned release, own LICENSE)
docs/                                Architecture, evaluation, and methodology documentation
```

## Evaluation summary

This project's own measured evaluation (not the model publisher's
author-reported figures) is documented in
[`docs/spectra_aasist3_evaluation.md`](docs/spectra_aasist3_evaluation.md)
(generalization) and
[`docs/spectra_production_optimization.md`](docs/spectra_production_optimization.md)
(INT8 quantization: 71.5% smaller artifact, ~1.9x faster cold load, no
measurable accuracy degradation vs. the FP32 baseline on the same
evaluation set). Every reported figure states its exact dataset, split
size, and model revision — no figure is presented without that context.

## Limitations

Detection performance can be affected by: synthesis methods not
represented in the evaluation data, audio compression and re-encoding,
background noise, very short clips, limited or non-speech audio content,
multiple overlapping speakers, language or domain shift relative to the
evaluation dataset, recording equipment characteristics, deliberate
adversarial manipulation, and general distribution shift between the
evaluation set and real-world audio. The upstream model is pre-release and
has no peer-reviewed publication. Video URL analysis additionally depends
on a connected Local URL Helper and is best-effort against a third-party
platform's own delivery behavior.

## Responsible use

This system is a **research prototype**. Its output should not be used as
the sole basis for legal decisions, forensic conclusions, disciplinary
actions, identity verification, or security decisions.

## Privacy

No account or sign-in is required. Uploaded/recorded media is processed
in memory by the hosted application's server-side process to produce the
current analysis and is not intentionally retained afterward. For Video
URL analysis, the public video is retrieved by the user's own connected
Local URL Helper, and only the selected audio interval is transferred to
the hosted application — never the full video, never account credentials
or cookies. Full details in
[`docs/local_url_helper.md`](docs/local_url_helper.md) and the in-app
**About** page.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system architecture
- [`docs/implementation_plan.md`](docs/implementation_plan.md) — phased implementation plan
- [`docs/model_candidate_analysis.md`](docs/model_candidate_analysis.md) — verified, cited candidate model research
- [`docs/cpu_model_benchmark.md`](docs/cpu_model_benchmark.md) — measured CPU engineering benchmarks
- [`docs/streamlit_mvp.md`](docs/streamlit_mvp.md) — Streamlit MVP architecture and behavior
- [`docs/deployment.md`](docs/deployment.md) — Streamlit Community Cloud deployment notes
- [`docs/spectra_aasist3_evaluation.md`](docs/spectra_aasist3_evaluation.md) — Spectra-AASIST3 generalization evaluation vs. the deployed model
- [`docs/spectra_production_optimization.md`](docs/spectra_production_optimization.md) — INT8 quantization of Spectra-AASIST3 for CPU deployment
- [`docs/spectra_streamlit_candidate.md`](docs/spectra_streamlit_candidate.md) — this deployment candidate's architecture and history
- [`docs/input_sources.md`](docs/input_sources.md) — the five input sources, normalization pipeline, and Proof-of-Origin token support
- [`docs/local_url_helper.md`](docs/local_url_helper.md) — Local URL Ingestion Helper architecture
- [`docs/development_policy.md`](docs/development_policy.md) — engineering conventions (environment, data/model provenance, code organization)

## License and third-party attribution

This project's own code is declared MIT-licensed in `pyproject.toml`; a
standalone `LICENSE` file has not yet been added at the repository root.
Third-party components retain their own licenses and are not relicensed
by inclusion here:

- Model weights: Apache-2.0 (see `configs/models.yaml` for exact upstream
  attribution and revision).
- `third_party/bgutil-ytdlp-pot-provider/` — vendored at a pinned release,
  GPL-3.0-only, upstream `LICENSE`/`README.md`/attribution preserved
  unmodified (see `third_party/bgutil-ytdlp-pot-provider/VENDORED.md`).
- FFmpeg, yt-dlp, Cloudflare `cloudflared`, and every other third-party
  dependency retain their own upstream licenses; none are redistributed
  from this repository (installed via `pip`/`packages.txt`, or downloaded
  by `scripts/build_local_helper.ps1` from the official source at build
  time).
