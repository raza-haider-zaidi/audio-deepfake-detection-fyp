# Audio Deepfake Detection

**Detecting AI-Generated and Cloned Voices: A Deep Learning System for Robust Audio Deepfake Detection**

Final-year Computer Science project.

## Purpose

This project builds a system for detecting AI-generated and voice-cloned audio
(audio deepfakes). It aims to evaluate how well deepfake detection approaches
generalise across datasets and generation methods, and how robust they are to
real-world signal degradation (compression, noise, re-recording).

## Current Status: Phase 3 — CPU Streamlit MVP

Completed:
- **Phase 1:** Project-local Python 3.12 `.venv`, FFmpeg, Git repository, and
  baseline repository structure.
- **Phase 2:** CPU-only pretrained Wav2Vec2 anti-spoofing model candidates
  researched, verified against their documented sources, integrated behind
  a model-independent adapter interface, and benchmarked for CPU inference
  latency and memory usage. See
  [`docs/model_candidate_analysis.md`](docs/model_candidate_analysis.md) and
  [`docs/cpu_model_benchmark.md`](docs/cpu_model_benchmark.md).
- **Phase 3 (this phase):** A polished, CPU-only Streamlit web application
  (`streamlit_app.py`) built around the Phase 2 inference backend, using
  `sara_wav2vec2` (`Sara1708/deepfake-audio-wav2vec2`, pinned revision
  `6c43629c953d6ff008501bf5f3eb983ac2321ad6`) as the deployment model. See
  [`docs/streamlit_mvp.md`](docs/streamlit_mvp.md) and
  [`docs/deployment.md`](docs/deployment.md).

Not yet started:
- Deploying to GitHub / Streamlit Community Cloud
- Dataset acquisition and real accuracy evaluation (Phase 4)
- Any model training or fine-tuning (out of scope for this project's MVP)

**Important:** No dataset has been used to evaluate real-world detection
accuracy at any phase so far. Any EER/accuracy figures referenced in this
project's documentation are author-reported model-card claims from the
model publishers, not results produced by this project.

## Deployment Target: CPU-Only, Browser-Based

The final application is a **Streamlit app running entirely on CPU**,
accessible from an ordinary browser on almost any device. It is explicitly
**not** designed around the development machine's NVIDIA GPU:

- No CUDA / NVIDIA GPU dependency anywhere in the production path
- No dedicated inference server, no other machine needing to be online, no
  database, no user accounts
- `torch`/`torchaudio` are installed from the official **CPU wheel index**
  (`https://download.pytorch.org/whl/cpu`), both locally and in
  `requirements.txt` for Streamlit Community Cloud
- All model adapters and the inference service accept `device="cpu"` or
  `device="auto"`, and `"auto"` always safely resolves to CPU
- The end user's device needs only a browser — no Python, PyTorch, CUDA, or
  local backend server

## Implementation Strategy

The project follows an **MVP-first approach**:

1. **Phase 1:** Environment and repository setup.
2. **Phase 2:** Select and CPU-benchmark a pretrained Wav2Vec2 audio
   deepfake detector (no training from scratch). See
   [`docs/model_candidate_analysis.md`](docs/model_candidate_analysis.md).
3. **Phase 3 (this phase):** Build a polished **Streamlit** app around the
   chosen deployment candidate for interactive, browser-based
   demonstration. See [`docs/streamlit_mvp.md`](docs/streamlit_mvp.md).
4. **Phase 4 (research extension):** Acquire an evaluation dataset, compare
   multiple detection approaches, evaluate **cross-dataset generalisation**,
   and test **robustness** under audio degradations (compression codecs,
   additive noise, resampling, re-recording/replay conditions).

Training a model from scratch is explicitly out of scope; the system relies
on existing pretrained detectors.

## Setup Instructions

### Prerequisites
- Windows with Python 3.12 available (installed via winget if not already present)
- Git
- FFmpeg (installed via winget)
- No GPU required. (The development machine has an NVIDIA RTX 5060 Ti, but
  it is intentionally not used or optimized for — everything here runs and
  is benchmarked on CPU.)

### Create and activate the virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Always use the project-local `.venv` — do not use a global or unrelated
project's Python environment.

### Install dependencies

```powershell
# CPU-only PyTorch/torchaudio — install from the official CPU wheel index,
# NOT plain PyPI (which may resolve a CUDA build):
.\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Remaining project dependencies (includes Streamlit + matplotlib):
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### Run the Streamlit app locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The page renders immediately without loading the ~468 MiB detection model —
the model is downloaded/loaded lazily the first time you click
**Analyze Audio**, and is cached (`st.cache_resource`) for the rest of the
running session.

### Run tests

```powershell
# Default suite — no real model downloads, safe/fast:
.\.venv\Scripts\python.exe -m pytest -v -m "not integration and not slow"

# Real model download/load/inference tests (network + large downloads):
.\.venv\Scripts\python.exe -m pytest -v -m "integration"
```

### Generate smoke-test audio and run a CLI prediction

```powershell
.\.venv\Scripts\python.exe scripts\generate_smoke_audio.py
.\.venv\Scripts\python.exe scripts\predict_audio.py data\samples\smoke_multitone_4s.wav --model sara_wav2vec2
```

Smoke audio is synthetic (sine/multi-tone waveforms) and used only to
validate the decode -> resample -> inference pipeline. Its predicted labels
have no scientific or detection-quality meaning.

### Run the CPU benchmark harnesses

```powershell
# Model-only CPU benchmark (Phase 2):
.\.venv\Scripts\python.exe scripts\benchmark_models.py

# Full-process resource diagnostic for the Streamlit app path (Phase 3, dev-only):
.\.venv\Scripts\python.exe scripts\benchmark_streamlit_resources.py
```

## Repository Structure

```
streamlit_app.py                    Streamlit entrypoint (Community Cloud runs this directly)
requirements.txt                    Root-level pinned dependencies for Streamlit Community Cloud
packages.txt                        Debian apt packages for Streamlit Community Cloud (ffmpeg)
.streamlit/config.toml              Streamlit server/theme configuration (no secrets)
app/                                 Streamlit presentation-layer helpers (no model logic)
    errors.py                       UserFacingError — friendly message + technical detail
    validation.py                   Upload constraints on top of shared preprocessing
    formatting.py                   Pure label/probability/window-table formatting
    visualizations.py               Waveform + mel-spectrogram matplotlib figures
    model_loader.py                 The only st.cache_resource-decorated model loader
src/audio_deepfake_detector/
    config/                         configs/models.yaml loader
    preprocessing/                  Audio loading (audio_loader.py), fixed-window framing (windowing.py)
    inference/                      Device-agnostic inference service
    models/                         BaseDeepfakeDetector interface, registry/factory, per-candidate adapters
    evaluation/                     Reserved for Phase 4 cross-dataset/robustness metrics
    utils/                          Typed data structures (datatypes.py), device resolution (device.py)
tests/                              Unit tests (default suite) + integration/slow-marked real-model tests
scripts/                            generate_smoke_audio.py, benchmark_models.py, predict_audio.py,
                                     benchmark_streamlit_resources.py
configs/                            models.yaml — single source of truth for model repos/revisions/labels
data/
    raw/                            Raw downloaded datasets (gitignored)
    processed/                      Processed/derived data (gitignored)
    samples/                        Generated smoke-test audio (gitignored)
models/
    checkpoints/                    Downloaded model weights (gitignored)
    cache/                          Hugging Face cache (gitignored)
results/
    metrics/                        cpu_model_benchmark.json and other measured metrics (gitignored)
    figures/                        Generated plots (gitignored)
    predictions/                    Model prediction outputs (gitignored)
    model_manifest.json             Metadata (no weights) for successfully tested models — committed
docs/                                Design, planning, and model-candidate documentation
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system architecture
- [`docs/implementation_plan.md`](docs/implementation_plan.md) — phased implementation plan
- [`docs/model_candidate_analysis.md`](docs/model_candidate_analysis.md) — verified, cited candidate model research
- [`docs/cpu_model_benchmark.md`](docs/cpu_model_benchmark.md) — measured CPU engineering benchmarks
- [`docs/streamlit_mvp.md`](docs/streamlit_mvp.md) — Streamlit MVP architecture and behavior
- [`docs/deployment.md`](docs/deployment.md) — Streamlit Community Cloud deployment notes
- [`docs/spectra_aasist3_evaluation.md`](docs/spectra_aasist3_evaluation.md) — Spectra-AASIST3 generalization evaluation vs. the deployed model
- [`docs/spectra_production_optimization.md`](docs/spectra_production_optimization.md) — INT8 quantization of Spectra-AASIST3 for CPU deployment
- [`docs/spectra_streamlit_candidate.md`](docs/spectra_streamlit_candidate.md) — experimental second Streamlit candidate using the INT8 model (branch `feat/spectra-streamlit-candidate`; not yet deployed, blocked on Hugging Face publishing)
- [`CLAUDE.md`](CLAUDE.md) — development rules for AI-assisted work in this repo
