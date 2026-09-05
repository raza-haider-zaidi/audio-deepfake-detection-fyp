# Implementation Plan

## Phase 1 — Environment & Repository Setup (complete)

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

## Phase 2 — CPU-Only Wav2Vec2 Model Selection & Inference Benchmarking (complete)

Purpose: determine which pretrained Wav2Vec2-based audio deepfake detector
gives the best practical balance of documentation quality, CPU inference
speed, RAM usage, checkpoint size, reproducibility, licensing, and Streamlit
suitability. This phase does **not** implement the Streamlit frontend.

- [x] Verify repository state clean; confirm `.venv` interpreter
- [x] Install CPU-only `torch`/`torchaudio` (from
      `https://download.pytorch.org/whl/cpu`) and remaining ML/audio
      dependencies; update `pyproject.toml`
- [x] Verify both candidate models' Hugging Face repositories and cited
      GitHub source repositories directly (not by trusting model-card prose
      alone) — see `docs/model_candidate_analysis.md`
- [x] Write `docs/model_candidate_analysis.md` — architecture, license,
      training data, checkpoint size, label mapping, published metrics
      (explicitly labeled author-reported), documented limitations
- [x] Create `configs/models.yaml` as the single source of truth for model
      IDs, repositories, revisions, checkpoint filenames, label mappings
- [x] Implement typed data structures (`AudioSample`, `PredictionResult`,
      `ModelLoadMetadata`, `ModelBenchmarkResult`) decoupled from Streamlit
- [x] Implement shared audio preprocessing (`preprocessing/audio_loader.py`)
      — WAV/MP3/FLAC decode, mono conversion, 16 kHz resampling, validation
      errors; does not crop to a fixed length
- [x] Implement `BaseDeepfakeDetector` interface and a registry/factory
      (`models/registry.py`) that resolves configured model IDs to adapters
- [x] Implement the Candidate B adapter
      (`Sara1708/deepfake-audio-wav2vec2`) from verified working source code
- [x] Attempt Candidate A adapter
      (`caa-speech-detection-asvspoof2019/wav2vec2-v2-unfrozen`) — its cited
      source repository does not exist (verified 404); documented as a
      critical finding rather than reconstructed by guesswork
- [x] Implement `scripts/generate_smoke_audio.py` — deterministic synthetic
      audio for pipeline validation only (explicitly not speech; no
      scientific meaning)
- [x] Implement `scripts/benchmark_models.py` — sequential, lazy-loaded CPU
      benchmarking (RAM before/after load, cold load time, warm inference
      mean/median/min/max)
- [x] Implement `scripts/predict_audio.py` — CLI prediction interface
- [x] Expand pytest coverage (config loading, audio loading, windowing,
      registry, device resolution, datatypes) with `integration`/`slow`
      markers separating real-model tests from the default suite
- [x] Download and attempt to load each candidate on CPU; record load
      success/failure honestly — both succeeded (Candidate A required an
      explicitly allow-listed, documented-as-expected missing key)
- [x] Write `results/model_manifest.json` (metadata only, no weights) and
      `results/metrics/cpu_model_benchmark.json` / `docs/cpu_model_benchmark.md`
- [x] Produce a preliminary (not final) deployment recommendation,
      distinguishing "best research candidate" from "best deployment
      candidate" — see the Phase 2 completion report

## Phase 3 — CPU Streamlit MVP (complete)

Purpose: build a polished, functional, CPU-only Streamlit web application
around the Phase 2 inference backend, deployable to Streamlit Community
Cloud, using `sara_wav2vec2` as the deployment model. This phase does
**not** create a GitHub repository or deploy to Streamlit Cloud.

- [x] Verify repository state clean; confirm `.venv` interpreter
- [x] Install Streamlit + matplotlib; update `pyproject.toml`
- [x] Create root-level `requirements.txt` (CPU PyTorch via
      `--extra-index-url`, pinned versions) and `packages.txt` (`ffmpeg`)
      for Streamlit Community Cloud
- [x] Create `streamlit_app.py` entrypoint + `app/` presentation-layer
      helpers (`errors.py`, `validation.py`, `formatting.py`,
      `visualizations.py`, `model_loader.py`) — no model logic duplicated
      outside `src/audio_deepfake_detector/`
- [x] Implement lazy, `st.cache_resource`-cached model loading pinned to
      revision `6c43629c953d6ff008501bf5f3eb983ac2321ad6`; verified the
      ~468 MiB model is not loaded on page visit, only on first "Analyze
      Audio" click, and is reused (not reloaded) on repeated analysis
- [x] Implement upload validation (30 s max duration, 25 MB max size,
      WAV/MP3/FLAC, friendly errors, no permanent storage)
- [x] Implement full-clip windowed inference reusing the existing
      `sara_wav2vec2` adapter (no silent first-4-seconds-only truncation),
      result card, window-level table, waveform plot, mel spectrogram,
      technical details expander, "Why Wav2Vec2?" explainer, research
      disclaimer — author-reported metrics intentionally kept off the main
      result view
- [x] Create `.streamlit/config.toml` (safe server/theme config, no
      secrets)
- [x] Create `scripts/benchmark_streamlit_resources.py` (dev-only process
      RSS diagnostic: startup / after load / after analysis)
- [x] Portability audit: no hardcoded Windows paths, no unnecessary
      CUDA/NVIDIA references in production code (found and fixed one
      documentation inconsistency in `.env.example`)
- [x] Expand pytest coverage: app formatting/validation/visualization/
      model-loader unit tests plus a Streamlit `AppTest`-based render test
      confirming no model load on initial page render — all in the default
      (non-integration) suite
- [x] Manually verify end-to-end on CPU: launched the real app locally,
      uploaded a smoke WAV through the browser, clicked Analyze, confirmed
      real inference, result card, waveform, spectrogram, and technical
      details all render correctly, and repeated analysis reuses the
      cached model
- [x] Write `docs/streamlit_mvp.md` and `docs/deployment.md`

## Phase 4 — Research Extension (not started)

- [ ] Acquire evaluation dataset(s) (document exact dataset name + version;
      do not commit raw data)
- [ ] Implement evaluation metrics (EER, ROC-AUC, F1, accuracy) under
      `src/audio_deepfake_detector/evaluation/`
- [ ] Run cross-dataset generalisation experiments
- [ ] Implement audio robustness perturbations (compression, noise,
      resampling, replay) and evaluate detector robustness under each
- [ ] Compare multiple detection approaches under identical evaluation
      protocol
- [ ] Summarise findings in `results/` and `docs/`

## Notes

- Evaluation results must always reflect actual runs — never fabricated or
  estimated numbers (see [`docs/development_policy.md`](development_policy.md)).
- Phase 2 established a working, CPU-only, benchmarked integration path. It
  did **not** establish real-world detection accuracy: there is still no
  evaluation dataset in this project. Any accuracy/EER figures quoted in
  Phase 2 documentation are author-reported model-card claims, not results
  produced by this project.
- Each phase should have accompanying tests before being considered complete.
