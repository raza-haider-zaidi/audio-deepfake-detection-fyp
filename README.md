# Audio Deepfake Detection

**Detecting AI-Generated and Cloned Voices: A Deep Learning System for Robust Audio Deepfake Detection**

Final-year Computer Science project.

## Purpose

This project builds a system for detecting AI-generated and voice-cloned audio
(audio deepfakes). It aims to evaluate how well deepfake detection approaches
generalise across datasets and generation methods, and how robust they are to
real-world signal degradation (compression, noise, re-recording).

## Current Status: Phase 1 — Environment Setup

This repository is currently in **Phase 1**: project scaffolding and
environment setup only. No model has been selected, downloaded, or
implemented yet, and no datasets have been downloaded.

Completed in this phase:
- Project-local Python 3.12 virtual environment (`.venv`)
- FFmpeg installed for audio I/O support
- Git repository initialised
- Repository structure and baseline configuration files

Not yet started:
- Pretrained model selection / integration
- Streamlit demonstration interface
- Dataset acquisition
- Any model training or fine-tuning

## Implementation Strategy

The project follows an **MVP-first approach**:

1. **Phase 1 (this phase):** Environment and repository setup.
2. **Phase 2 (MVP):** Integrate a single **pretrained** audio deepfake
   detection model (no training from scratch) behind a simple inference
   interface, with a **Streamlit** app for interactive demonstration.
3. **Phase 3 (research extension):** Compare multiple detection approaches,
   evaluate **cross-dataset generalisation**, and test **robustness** under
   audio degradations (compression codecs, additive noise, resampling,
   re-recording/replay conditions).

Training a model from scratch is explicitly out of scope for the MVP; the
initial system relies on an existing pretrained detector.

## Setup Instructions

### Prerequisites
- Windows with Python 3.12 available (installed via winget if not already present)
- Git
- FFmpeg (installed via winget)
- NVIDIA GPU with CUDA support recommended (developed against an RTX 5060 Ti, 16 GB VRAM)

### Create and activate the virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Always use the project-local `.venv` — do not use a global or unrelated
project's Python environment.

### Install dependencies

Dependencies are not yet pinned (Phase 1 has no ML dependencies). Once the
MVP model integration begins:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Repository Structure

```
app/                                Streamlit / application entry point (UI layer, added in Phase 2)
src/audio_deepfake_detector/
    config/                         Configuration loading/schemas
    preprocessing/                  Audio loading, resampling, feature extraction
    inference/                      Model inference wrappers
    models/                         Model adapters (pretrained model integrations)
    evaluation/                     Metrics, cross-dataset & robustness evaluation
    utils/                          Shared utilities
tests/                              Unit and integration tests
scripts/                            One-off / CLI utility scripts
configs/                            YAML/JSON configuration files
data/
    raw/                            Raw downloaded datasets (gitignored)
    processed/                      Processed/derived data (gitignored)
    samples/                        Small example audio clips (gitignored)
models/
    checkpoints/                    Downloaded/fine-tuned model weights (gitignored)
    cache/                          Hugging Face / torch hub cache (gitignored)
results/
    metrics/                        Evaluation metric outputs (gitignored)
    figures/                        Generated plots (gitignored)
    predictions/                    Model prediction outputs (gitignored)
docs/                                Design and planning documentation
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — planned system architecture
- [`docs/implementation_plan.md`](docs/implementation_plan.md) — phased implementation plan
- [`CLAUDE.md`](CLAUDE.md) — development rules for AI-assisted work in this repo
