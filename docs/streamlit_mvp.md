# Streamlit MVP (Phase 3)

## Purpose

Phase 3 builds a Streamlit presentation layer around the CPU inference
backend selected and benchmarked in Phase 2. Streamlit is *only* the
presentation layer — no preprocessing, model, or inference logic lives in
`streamlit_app.py` or `app/`; everything ML-related is imported from
`src/audio_deepfake_detector/`.

## Deployment model

| | |
|---|---|
| Model ID | `sara_wav2vec2` |
| Repository | `Sara1708/deepfake-audio-wav2vec2` |
| Pinned revision | `6c43629c953d6ff008501bf5f3eb983ac2321ad6` |
| Checkpoint size | 491,044,441 bytes (~468 MiB) |
| Label mapping | `0 = bonafide`, `1 = spoof` (confirmed from source code — see `docs/model_candidate_analysis.md`) |

Candidate A (`caa_wav2vec2`) remains in `configs/models.yaml` and
`src/audio_deepfake_detector/models/candidate_a.py` for research purposes,
but is **not** used by the Streamlit app. Its adapter is never imported by
`streamlit_app.py` or any module under `app/`.

The model revision is pinned explicitly in `configs/models.yaml` and read
through the existing `models_config` loader — the app never fetches "latest"
from Hugging Face.

## Entry point

```
streamlit_app.py                (repository root — Streamlit Cloud entry point)
app/
    errors.py                    UserFacingError — friendly message + technical detail
    validation.py                Upload constraints (duration, size) on top of shared preprocessing
    formatting.py                Pure label/probability/window-table formatting (no Streamlit import)
    visualizations.py            Waveform + mel-spectrogram matplotlib figures (no Streamlit import)
    model_loader.py              The ONLY st.cache_resource-decorated loader in the app
```

Run locally:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Lazy model loading

`app/model_loader.get_detector()` is decorated with `st.cache_resource` and
is called **only** from inside the "Analyze Audio" button handler in
`streamlit_app.py`. Visiting the page, uploading a file, or previewing audio
never triggers a model load. Once loaded, the detector object is cached for
the lifetime of the running app process — a rerun (e.g. from a widget
interaction) does not reload the ~468 MiB checkpoint.

## Long-audio windowing and aggregation

Uploads are capped at 30 seconds (`app/validation.py`). The existing,
tested `sara_wav2vec2` adapter (`src/audio_deepfake_detector/models/candidate_b.py`)
handles all windowing:

- Clips ≤ 4 seconds are zero-padded to one window (documented short-audio
  behavior from the verified source implementation).
- Longer clips are split into overlapping 4-second windows (50% overlap,
  32,000-sample hop) covering the **entire** clip — never just the first
  4 seconds.
- The clip-level result is the **mean spoof probability across all
  windows** — a project-level aggregation choice for this MVP, not a method
  the original model authors scientifically validated. This is stated
  explicitly in the app's "Technical details" expander.
- Window-level probabilities are retained and shown in an optional
  "Window-level analysis" table when more than one window was analyzed.

## Result presentation

- Normalized labels are shown as user-friendly phrases ("Likely Real /
  Bonafide" / "Likely AI-Generated / Spoofed"), never as absolute claims.
- The class probability is labeled "Model confidence" / "probability",
  never "certainty" — it is not claimed to be a calibrated confidence score.
- Author-reported model-card metrics (e.g. the 5.55% eval EER) are
  intentionally **not** shown on the main MVP page, to avoid any appearance
  that this project measured them.

## Memory management

- `st.cache_resource` ensures a single detector instance per model ID per
  running process.
- No uploaded audio is written to a permanent data folder; validation and
  decoding happen from the in-memory upload buffer via the existing
  `load_audio_bytes` preprocessing function.
- matplotlib figures are rendered with `st.pyplot(fig, clear_figure=True)`
  so figures are closed after rendering rather than accumulating in memory
  across analyses.
- `scripts/benchmark_streamlit_resources.py` is a development-only tool
  (not part of the public UI) that measures process RSS at startup, after
  model load, and after one analysis — see `docs/deployment.md` for
  measured figures.

## Testing

- `tests/test_app_formatting.py`, `tests/test_app_validation.py`,
  `tests/test_app_visualizations.py`, `tests/test_app_model_loader.py` —
  pure-function unit tests, no Streamlit runtime or model download needed.
- `tests/test_streamlit_app.py` — uses Streamlit's `AppTest` to render the
  app and assert it does not error and does not load the model on initial
  render. Included in the default (`not integration and not slow`) suite
  because it never downloads a checkpoint.
- Real end-to-end inference through the Streamlit code path is exercised
  manually (see `docs/deployment.md` for the local functional test run)
  and via the existing `tests/test_integration_models.py` for the
  underlying adapter, which is the same adapter the app calls.
