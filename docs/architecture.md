# Architecture

This document describes the system architecture as of **Phase 3 (CPU
Streamlit MVP)**. The Streamlit presentation layer is now built and calls
into the inference backend selected and benchmarked in Phase 2.

## Production data flow (implemented — no GPU dependency)

```
Browser (any device)
   |
   v
Streamlit                     (streamlit_app.py + app/ — presentation layer only)
   |
   v
shared audio preprocessing    (src/audio_deepfake_detector/preprocessing/)
   |
   v
model adapter                 (src/audio_deepfake_detector/models/candidate_b.py — sara_wav2vec2)
   |
   v
CPU Wav2Vec2 inference         (torch, CPU-only wheel; no CUDA dependency anywhere in this path)
   |
   v
prediction + visualization    (Streamlit — result card, waveform, mel spectrogram)
```

The entire production path runs with `torch` installed from the CPU-only
wheel index (`https://download.pytorch.org/whl/cpu`). No layer in this path
requires CUDA, an NVIDIA GPU, a dedicated inference server, an API key, or
another machine being online. The RTX 5060 Ti present on the development
machine is intentionally not used or optimized for. See
`docs/deployment.md` for the Streamlit Community Cloud deployment plan.

## Layers

```
+-------------------------------+
|  streamlit_app.py + app/       |  presentation layer only — no model logic
|  |-- errors.py                  |  UserFacingError
|  |-- validation.py              |  upload duration/size constraints
|  |-- formatting.py              |  pure label/probability/table formatting
|  |-- visualizations.py          |  waveform + mel-spectrogram figures
|  `-- model_loader.py            |  the ONLY st.cache_resource model loader
+-------------------------------+
|  src/audio_deepfake_detector   |
|  |-- inference/                 |  device-agnostic service: create_detector -> load -> predict -> unload
|  |-- models/                    |  BaseDeepfakeDetector interface + registry/factory + one adapter per candidate
|  |-- preprocessing/             |  audio_loader.py (decode/mono/resample), windowing.py (fixed-window framing)
|  |-- config/                    |  models_config.py loads configs/models.yaml
|  |-- evaluation/                |  reserved for Phase 4 cross-dataset/robustness metrics (empty)
|  `-- utils/                     |  datatypes.py (AudioSample/PredictionResult/...), device.py
+-------------------------------+
```

## Design Principles

- **UI/inference separation.** `app/` and `streamlit_app.py` only call into
  `src/audio_deepfake_detector/inference/service.py` and the model
  registry — they contain no model logic. `app/model_loader.py` is the
  single point where a `BaseDeepfakeDetector` is instantiated and loaded
  for the UI.
- **Model adapters, not a monolith.** `BaseDeepfakeDetector`
  (`models/base.py`) defines `load()`, `predict()`, `unload()`,
  `model_info()`. The generic inference service, CLI, and Streamlit app
  never import a candidate adapter directly — they resolve model IDs
  through `models/registry.py`, which reads `configs/models.yaml`. No
  repository name or checkpoint filename is hardcoded outside that config
  file. `app/model_loader.DEPLOYMENT_MODEL_ID` is the one place the
  Streamlit app pins its model choice (`sara_wav2vec2`).
- **Shared preprocessing stays generic.** `preprocessing/audio_loader.py`
  decodes WAV/MP3/FLAC, converts to mono float32, and resamples to 16 kHz —
  it does **not** crop or pad to a fixed length. Padding, truncation, and
  sliding-window framing are model-specific and live inside each adapter
  (`preprocessing/windowing.py` provides the reusable primitive; adapters
  decide their own window/hop sizes based on what their source
  documentation specifies).
- **CPU-only, always.** `utils/device.py` only accepts `"cpu"` or `"auto"`
  as inputs, and `"auto"` always resolves to `"cpu"` in this project — there
  is no code path that requests a CUDA device.
- **No training.** Both candidate models are pretrained checkpoints
  downloaded via `huggingface_hub`; nothing in this repository trains or
  fine-tunes a model.
- **Lazy, cached model loading.** `app/model_loader.get_detector()` is
  wrapped in `st.cache_resource` and is only called from inside the
  "Analyze Audio" button handler — never merely because a user visits the
  page. Once loaded, the same detector instance is reused for every
  subsequent analysis in that running app process (verified: repeated
  analysis in the same session completes in ~180 ms with no reload
  spinner). The standalone benchmark harness
  (`scripts/benchmark_models.py`) similarly loads one model, benchmarks it,
  unloads it, then moves to the next — never two large models resident in
  memory simultaneously.
- **Portable paths.** All filesystem paths throughout `src/`, `app/`, and
  `scripts/` use `pathlib`; nothing hardcodes a Windows path or drive
  letter, so the same code runs unmodified on Streamlit Community Cloud's
  Debian Linux runtime.

## Phase 2 deliverables

- `docs/model_candidate_analysis.md` — verified, cited model-card research
  for both candidates plus the optional reference model
- `configs/models.yaml` — single source of truth for repository IDs,
  revisions, checkpoint filenames, and label mappings
- `src/audio_deepfake_detector/models/candidate_b.py` — adapter for
  `Sara1708/deepfake-audio-wav2vec2`, reproduced from verified working
  source code
- Candidate A (`caa-speech-detection-asvspoof2019/wav2vec2-v2-unfrozen`) —
  its cited source repository does not exist (verified 404); see
  `docs/model_candidate_analysis.md` for the documented failure and how it
  was handled (checkpoint state_dict inspection instead of invented
  architecture). Retained for research purposes only; not used by the
  Streamlit app.
- `scripts/generate_smoke_audio.py`, `scripts/benchmark_models.py`,
  `scripts/predict_audio.py` — deterministic synthetic audio, CPU
  engineering benchmarks, and a CLI prediction interface
- `results/model_manifest.json`, `results/metrics/cpu_model_benchmark.json`,
  `docs/cpu_model_benchmark.md` — measured engineering results, clearly
  separated from author-reported detection metrics

## Phase 3 deliverables (this phase)

- `streamlit_app.py` — Streamlit entrypoint; upload, preview, analyze,
  result card, waveform, mel spectrogram, technical details, disclaimer
- `app/` — presentation-layer helpers (validation, formatting,
  visualization, cached model loading), all decoupled from Streamlit where
  practical for testability
- `requirements.txt`, `packages.txt`, `.streamlit/config.toml` — Streamlit
  Community Cloud deployment files (see `docs/deployment.md`)
- `scripts/benchmark_streamlit_resources.py` — development-only process
  RSS diagnostic for the full app path (startup / after load / after
  analysis)
- `docs/streamlit_mvp.md`, `docs/deployment.md` — MVP behavior and
  deployment documentation

## Planned Research Extension (Phase 4 — not started)

- `evaluation/` will support running a given model adapter against multiple
  datasets to measure cross-dataset generalisation, and against
  perturbed/degraded audio (compression, noise, resampling) to measure
  robustness. This requires acquiring an evaluation dataset, which has not
  happened yet.
- Results will be written to `results/metrics/`, `results/figures/`, and
  `results/predictions/` (all gitignored).

## Open Decisions (not yet made)

- Whether/when to create the GitHub repository and deploy to Streamlit
  Community Cloud (explicitly deferred past Phase 3)
- Which dataset(s) to use for Phase 4 evaluation (e.g. ASVspoof family, or
  others)
- Exact robustness perturbation set

These will be decided in later phases and documented here once decided.
