# Architecture

This document describes the system architecture as of **Phase 2 (CPU-only
Wav2Vec2 model selection and inference benchmarking)**. The Streamlit
frontend itself has not been built yet — Phase 2 built and benchmarked the
inference layer it will call into.

## Production data flow (target — no GPU dependency)

```
Browser
   |
   v
Streamlit  (not yet implemented — Phase 3)
   |
   v
shared audio preprocessing   (src/audio_deepfake_detector/preprocessing/)
   |
   v
model adapter                 (src/audio_deepfake_detector/models/)
   |
   v
CPU Wav2Vec2 inference         (torch, CPU-only wheel; no CUDA dependency anywhere in this path)
   |
   v
prediction + visualization    (Streamlit — Phase 3)
```

The entire production path is designed to run with `torch` installed from
the CPU-only wheel index (`https://download.pytorch.org/whl/cpu`). No layer
in this path requires CUDA, an NVIDIA GPU, a dedicated inference server, an
API key, or another machine being online. The RTX 5060 Ti present on the
development machine is intentionally not used or optimized for.

## Layers

```
+-------------------------------+
|  app/  (Streamlit UI)          |  presentation layer — Phase 3, not yet built
+-------------------------------+
|  src/audio_deepfake_detector   |
|  |-- inference/                 |  device-agnostic service: create_detector -> load -> predict -> unload
|  |-- models/                    |  BaseDeepfakeDetector interface + registry/factory + one adapter per candidate
|  |-- preprocessing/             |  audio_loader.py (decode/mono/resample), windowing.py (fixed-window framing)
|  |-- config/                    |  models_config.py loads configs/models.yaml
|  |-- evaluation/                |  reserved for Phase 3 cross-dataset/robustness metrics (empty in Phase 2)
|  `-- utils/                     |  datatypes.py (AudioSample/PredictionResult/...), device.py
+-------------------------------+
```

## Design Principles

- **UI/inference separation.** `app/` (Streamlit, Phase 3) will only call
  into `src/audio_deepfake_detector/inference/service.py`; it must not
  contain model logic.
- **Model adapters, not a monolith.** `BaseDeepfakeDetector`
  (`models/base.py`) defines `load()`, `predict()`, `unload()`,
  `model_info()`. The generic inference service and CLI never import a
  candidate adapter directly — they resolve model IDs through
  `models/registry.py`, which reads `configs/models.yaml`. No repository
  name or checkpoint filename is hardcoded outside that config file.
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
- **No training in Phase 2.** Both candidate models are pretrained
  checkpoints downloaded via `huggingface_hub`; nothing in this repository
  trains or fine-tunes a model.
- **Lazy, sequential model loading.** The benchmark harness
  (`scripts/benchmark_models.py`) loads one model, benchmarks it, unloads it
  (`gc.collect()`), then moves to the next — never two large models resident
  in memory simultaneously. This matters because the eventual Streamlit
  deployment target has limited RAM.

## Phase 2 deliverables (this phase)

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
  architecture)
- `scripts/generate_smoke_audio.py`, `scripts/benchmark_models.py`,
  `scripts/predict_audio.py` — deterministic synthetic audio, CPU
  engineering benchmarks, and a CLI prediction interface
- `results/model_manifest.json`, `results/metrics/cpu_model_benchmark.json`,
  `docs/cpu_model_benchmark.md` — measured engineering results, clearly
  separated from author-reported detection metrics

## Planned Research Extension (Phase 4 — not started)

- `evaluation/` will support running a given model adapter against multiple
  datasets to measure cross-dataset generalisation, and against
  perturbed/degraded audio (compression, noise, resampling) to measure
  robustness. This requires acquiring an evaluation dataset, which has not
  happened yet — see "Do not yet claim robust results" in
  `docs/implementation_plan.md`.
- Results will be written to `results/metrics/`, `results/figures/`, and
  `results/predictions/` (all gitignored).

## Open Decisions (not yet made)

- Final model choice for the Streamlit MVP (pending completed CPU
  benchmarking — see `docs/cpu_model_benchmark.md`)
- Which dataset(s) to use for Phase 4 evaluation (e.g. ASVspoof family, or
  others)
- Exact robustness perturbation set

These will be decided in later phases and documented here once decided.
