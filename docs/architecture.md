# Architecture (Planned)

This document describes the intended system architecture. It reflects the
plan as of Phase 1 (environment setup) and will be revised as implementation
proceeds — nothing described below beyond directory layout has been built
yet.

## Layers

```
┌─────────────────────────────┐
│  app/  (Streamlit UI)        │  presentation layer — Phase 2
├─────────────────────────────┤
│  src/audio_deepfake_detector │
│  ├── inference/               │  loads a model adapter, runs prediction
│  ├── models/                  │  one adapter per detection approach
│  ├── preprocessing/           │  audio loading, resampling, feature extraction
│  ├── evaluation/               │  metrics, cross-dataset & robustness tests
│  ├── config/                   │  configuration schemas/loading
│  └── utils/                    │  shared helpers
└─────────────────────────────┘
```

## Design Principles

- **UI/inference separation.** `app/` (Streamlit) only calls into
  `src/audio_deepfake_detector/inference/`; it must not contain model logic.
- **Model adapters.** Each detection approach (e.g. a pretrained
  spectrogram-based classifier, a self-supervised-embedding-based detector)
  is implemented as an adapter with a common interface, so approaches can be
  swapped or compared without changing inference/evaluation code.
- **No training in the MVP.** Phase 2 wraps an existing pretrained model.
  Training/fine-tuning code, if added later, is a separate concern from
  inference.

## Planned Data Flow (Phase 2 MVP)

1. User uploads/records an audio clip via Streamlit (`app/`).
2. `preprocessing/` loads and normalises the audio (resampling, mono
   conversion, FFmpeg-backed decoding as needed).
3. `inference/` passes the processed audio to the selected model adapter in
   `models/`.
4. The adapter returns a real/fake prediction with a confidence score.
5. `app/` renders the result to the user.

## Planned Research Extension (Phase 3)

- `evaluation/` will support running a given model adapter against multiple
  datasets to measure cross-dataset generalisation, and against
  perturbed/degraded audio (compression, noise, resampling) to measure
  robustness.
- Results are written to `results/metrics/`, `results/figures/`, and
  `results/predictions/` (all gitignored — see [`implementation_plan.md`](implementation_plan.md)).

## Open Decisions (not yet made)

- Which pretrained model(s) to use for the MVP.
- Which dataset(s) to use for evaluation (e.g. ASVspoof family, or others).
- Exact robustness perturbation set.

These will be decided in later phases and documented here once decided.
