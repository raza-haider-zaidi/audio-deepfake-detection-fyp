# Deployment (Streamlit Community Cloud)

**Status: not yet deployed.** This document describes the architecture and
files prepared for deployment; no GitHub repository or Streamlit Cloud app
has been created yet (that is a deliberately separate, later step).

## Final application architecture

```
Browser (any device)
   |
   v
Streamlit Community Cloud (Debian Linux)
   |
   v
streamlit_app.py  +  app/  (presentation layer only)
   |
   v
src/audio_deepfake_detector/  (shared preprocessing, model registry, adapters)
   |
   v
CPU Wav2Vec2 inference (sara_wav2vec2, pinned revision 6c43629c953d6ff008501bf5f3eb983ac2321ad6)
   |
   v
prediction + waveform + mel spectrogram (rendered back in the browser)
```

The end user's device needs only a browser. No Python, PyTorch, CUDA,
NVIDIA hardware, or local backend server is required on the client side —
all inference happens in the Streamlit Cloud process itself.

## CPU-only inference

- `requirements.txt` pins `torch==2.13.0+cpu` and `torchaudio==2.11.0+cpu`
  via `--extra-index-url https://download.pytorch.org/whl/cpu`, so
  Community Cloud never installs a CUDA build.
- `src/audio_deepfake_detector/utils/device.py` only accepts `"cpu"` and
  `"auto"`, and `"auto"` always resolves to `"cpu"` — there is no code path
  anywhere in the application that requests a CUDA device.
- Verified locally: `torch.cuda.is_available() == False` throughout, and
  the full upload -> analyze -> result flow was exercised in a real
  browser session against a locally running instance of the app with no
  GPU involved.

## Runtime environment

- Python 3.12 (`requires-python = ">=3.12,<3.13"` in `pyproject.toml`)
- `requirements.txt` (repository root) — the file Streamlit Community Cloud
  installs directly with pip. Kept logically consistent with
  `pyproject.toml`'s `dependencies` list.
- `packages.txt` (repository root) — Debian `apt-get` packages Community
  Cloud installs before Python dependencies. Currently just `ffmpeg`, as a
  robust decoding fallback for compressed audio formats.

## Model download and caching at runtime

- The `sara_wav2vec2` checkpoint (and the `facebook/wav2vec2-base` backbone
  it's built on) is **not** stored in this Git repository. It is downloaded
  from Hugging Face Hub at runtime via `huggingface_hub.hf_hub_download`,
  the same mechanism used throughout this project (see
  `src/audio_deepfake_detector/models/candidate_b.py`).
- The model revision is **pinned** to
  `6c43629c953d6ff008501bf5f3eb983ac2321ad6` in `configs/models.yaml` — the
  app never silently picks up the newest repository revision.
- `HF_HOME` is not hardcoded to a Windows path anywhere; when unset,
  `huggingface_hub` uses its own portable default cache location
  (`~/.cache/huggingface` on Linux, the Community Cloud runtime's home
  directory), which works identically on Windows and Debian without any
  code change.
- No checkpoint file, Hugging Face cache directory, or generated audio file
  is committed to Git — see `.gitignore` and the Git safety review below.

## Expected first-load behavior

- Visiting the deployed app does **not** download or load the model. The
  page renders immediately with just the upload widget, "Why Wav2Vec2?"
  expander, and disclaimer.
- The model (~468 MiB checkpoint plus the `facebook/wav2vec2-base`
  backbone) is downloaded and loaded lazily, the first time a user clicks
  "Analyze Audio" — wrapped in a spinner reading "Loading the detection
  model. The first analysis may take longer while the model is prepared."
  No exact loading time is promised, since Streamlit Cloud's actual
  network/CPU characteristics may differ from local measurements below.
- `app/model_loader.get_detector()` is decorated with `st.cache_resource`,
  so the model is loaded at most once per running app process/container —
  subsequent analyses (by the same or different users hitting the same
  running instance) reuse the cached detector instead of reloading.

## No permanent user audio storage

Uploaded audio is processed entirely in memory
(`app/validation.py` -> `audio_deepfake_detector.preprocessing.audio_loader.load_audio_bytes`,
which operates on the in-memory upload buffer, `io.BytesIO`, never a
temp/data file). Nothing under `data/` is written to during a normal
Streamlit session; `data/samples/` is only populated by the developer-run
`scripts/generate_smoke_audio.py` script, and is gitignored regardless.

## Measured local resource usage (development machine, not Streamlit Cloud)

Measured 2026-08-23 via two independent methods — the standalone
`scripts/benchmark_streamlit_resources.py` diagnostic, and direct OS-level
process RSS inspection of a real, browser-driven `streamlit run` session:

| Stage | Standalone script RSS | Real `streamlit run` process RSS |
|---|---|---|
| Startup (before any model load) | 32.6 MB | ~63 MB (Streamlit server overhead) |
| After model load | 1218.3 MB (+1185.7 MB) | ~930 MB after load + one analysis (combined) |
| After one analysis | 1235.8 MB (+17.5 MB) | (see above — measured together) |

First real-browser model load + inference (cold): completed in a few
seconds end-to-end, including the spinner. Second (cached) inference on the
same running server: 186 ms measured inference time, no reload spinner —
confirming `st.cache_resource` avoids repeat loads within a process.

**These are single-machine, Windows-development-environment measurements.
Streamlit Community Cloud's actual CPU and memory characteristics may
differ** — this is stated explicitly rather than assumed. Community Cloud's
free tier historically provides roughly 1 GB of RAM per app; a ~0.9-1.2 GB
peak footprint for this model is a real constraint worth monitoring at
actual deployment time, not before.

## Streamlit resource considerations

- `st.cache_resource` is the only caching mechanism used for the model —
  no duplicate model instances, and Candidate A (`caa_wav2vec2`) is never
  imported or loaded by the app.
- matplotlib figures are rendered with `st.pyplot(fig, clear_figure=True)`
  so they are closed immediately after rendering rather than accumulating.
- `.streamlit/config.toml` sets a 50 MB server-level upload cap (tighter
  than Streamlit's 200 MB default) on top of the stricter 25 MB / 30-second
  application-level validation in `app/validation.py`.

## Not yet done (explicitly out of scope for this phase)

- No GitHub repository has been created.
- The app has not been deployed to Streamlit Community Cloud.
- Actual Streamlit Cloud CPU/RAM/cold-start behavior has not been measured
  and should not be assumed to match the local figures above.
