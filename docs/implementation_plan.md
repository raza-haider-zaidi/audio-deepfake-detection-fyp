# Implementation Plan

## Phase 1 — Environment & Repository Setup (this phase)

- [x] Inspect working directory for existing files
- [x] Inspect Windows environment (Python, Git, GPU, FFmpeg, VS Code)
- [x] Install Python 3.12 (via winget)
- [x] Install FFmpeg (via winget)
- [x] Create project-local `.venv`
- [x] Initialise Git repository
- [x] Create repository directory structure
- [x] Create `.gitignore` with `.gitkeep` placeholders
- [x] Create baseline documentation and config files
- [x] Add a minimal import smoke test
- [x] Validate environment (Python version, import test, pytest, git status)
- [x] First commit

No ML model, dataset, or UI code is implemented in this phase.

## Phase 2 — MVP (Pretrained Model Integration)

- [ ] Select a pretrained audio deepfake / spoof detection model (document
      exact model name + revision in this file once chosen)
- [ ] Add ML/audio dependencies to `pyproject.toml` (torch, torchaudio,
      transformers, etc. — pinned versions)
- [ ] Implement a model adapter under
      `src/audio_deepfake_detector/models/`
- [ ] Implement preprocessing (audio loading/resampling via FFmpeg/torchaudio)
      under `src/audio_deepfake_detector/preprocessing/`
- [ ] Implement an inference wrapper under
      `src/audio_deepfake_detector/inference/`
- [ ] Add unit tests for preprocessing and inference
- [ ] Build a minimal Streamlit demonstration app under `app/`
- [ ] Manually verify end-to-end: upload audio → prediction displayed

## Phase 3 — Research Extension

- [ ] Acquire evaluation dataset(s) (document exact dataset name + version;
      do not commit raw data)
- [ ] Implement evaluation metrics under
      `src/audio_deepfake_detector/evaluation/`
- [ ] Run cross-dataset generalisation experiments
- [ ] Implement audio robustness perturbations (compression, noise,
      resampling, replay) and evaluate detector robustness under each
- [ ] Compare multiple detection approaches under identical evaluation
      protocol
- [ ] Summarise findings in `results/` and `docs/`

## Notes

- Evaluation results must always reflect actual runs — never fabricated or
  estimated numbers (see [`CLAUDE.md`](../CLAUDE.md)).
- Each phase should have accompanying tests before being considered complete.
