# Spectra-AASIST3 INT8 Streamlit Candidate

Branch: `feat/spectra-streamlit-candidate` (child of
`feat/spectra-production-optimization`). `main` is unmodified and the
currently live Streamlit deployment (still `sara_wav2vec2`) is untouched.
This branch prepares a **second, temporary** Streamlit app for validation
only — it does not replace anything live.

## Status: BLOCKED on Hugging Face authentication (manual action required)

This phase's primary goal — publishing the derived INT8 ONNX artifact to a
public Hugging Face model repository — **could not be completed**, because
it requires a Hugging Face account/token this project cannot obtain or
guess on its own, and the instructions for this phase are explicit: do not
guess a username, do not ask for a token in chat, and stop for manual
browser/account action when authentication is required.

```
$ hf auth whoami
Error: Not logged in
```

**What you need to do, exactly:**

1. Open a terminal with this project's `.venv` activated (or run
   `.venv\Scripts\hf auth login` directly).
2. Run `hf auth login` (or, in this environment, `.venv\Scripts\hf.exe
   auth login`).
3. It will prompt for a Hugging Face access token. Create one, if you
   don't already have a suitable one, at
   https://huggingface.co/settings/tokens — a token with **write** access
   is required to create a repository and upload files. Paste it into the
   terminal prompt yourself; **do not paste it into this chat.**
4. Once logged in, tell me to continue, and I will:
   - confirm your authenticated username (`hf auth whoami`, printing only
     the username, never the token),
   - check whether `<your-username>/spectra-aasist3-int8-audio-deepfake`
     already exists (and stop without overwriting if it contains
     unrelated work),
   - create the public model repository,
   - upload the model card, `quantization_metadata.json`, and the INT8
     ONNX artifact,
   - record the resulting commit SHA,
   - pin that exact repository/revision/SHA256 into `configs/models.yaml`
     for `spectra_aasist3_onnx_int8` (replacing the current
     `PENDING_HF_PUBLISH` placeholders),
   - re-run the resource benchmark against the real hosted artifact
     (Step 19), and
   - finish Steps 21–22 (commit, push, second-deployment coordinates).

Everything in this document that does **not** depend on the actual upload
has already been completed and is described below.

## 1. INT8 artifact re-verification (Step 2)

Re-verified from the previous phase's local artifact
(`models/cache/quantized/spectra-aasist3-int8-dynamic.onnx`, gitignored,
not committed):

- **Filename**: `spectra-aasist3-int8-dynamic.onnx`
- **Size**: 364,036,647 bytes
- **SHA256**: `444f832d306a2be4f823119f84e698e8821db6a1aab248593d4b05b7a9a48108`
- **Input**: `wav` float32 `[batch, 64600]`
- **Output**: `logits` float32 `[batch, 2]`
- Loads with `providers == ['CPUExecutionProvider']` only
- Deterministic on repeated smoke inference (bit-identical output)

Base FP32 model this was derived from: `lab260/Spectra-AASIST3`, revision
`bc0ded888080ddad493177bb53aa6f5b95219d7c`, Apache-2.0.

## 2. License / redistribution material (Step 3)

Both `lab260/Spectra-AASIST3` and its upstream backbone
`facebook/wav2vec2-xls-r-300m` are Apache-2.0 (verified directly via the
HF API in the prior phase — `docs/spectra_production_optimization.md`
Section 14). No separate `NOTICE`/`LICENSE` file exists in the source
repository to carry forward (checked directly via the HF API file
listing — none present). The model card prepared for the derived
repository (Section 4 below) explicitly states, at minimum: the original
model name and repository, the original revision, the Apache-2.0
reference, that the weights were **not retrained**, and that this is a
dynamic INT8 ONNX derivative produced for an academic FYP. This is a
factual statement of what was done, not a legal opinion, and no
redistribution guarantee beyond the license's own text is claimed.

## 3. Bug fixes made while preparing the production path

Two real defects in `candidate_e.py` were found and fixed while wiring
this branch, independent of the HF-publish blocker:

1. **`predict()` ignored the calibrated threshold.** It compared
   `spoof_prob >= bonafide_prob` (an implicit 0.5 softmax split) instead
   of the actual calibrated operating threshold. Since the calibrated
   thresholds are ~0.93 (FP32) / ~0.94 (INT8), this would have made every
   real classification decision wrong relative to the calibration
   experiment. Fixed: `predict()` now compares against `self.threshold`.
2. **`load()` had a hard, unconditional `import torch`** used only to
   report a version string — this would have crashed on a genuine
   Streamlit Cloud environment once torch is (correctly) removed from
   `requirements.txt`. Fixed: the import is now wrapped in `try/except
   ImportError`, falling back to a descriptive string.

Both are covered by new/updated tests (Section 7).

## 4. Threshold wiring (Step 14)

`CandidateSpectraAasist3OnnxDetector.__init__` now selects the threshold
automatically from `model_config.id`, so callers can never accidentally
apply the wrong one:

- `spectra_aasist3_onnx` (FP32) → `FP32_CALIBRATED_THRESHOLD = 0.9299831390380859`
- `spectra_aasist3_onnx_int8` (INT8) → `INT8_DYNAMIC_CALIBRATED_THRESHOLD = 0.939693808555603`

Both values are frozen calibration results from
`docs/spectra_production_optimization.md`, computed on the calibration
split only — never the evaluation set, and never each other's threshold.
A regression test (`test_int8_detector_automatically_uses_int8_threshold_not_fp32`)
guards against ever mixing them up.

## 5. Artifact integrity verification (Step 10 — implemented, not yet exercised against the real host)

`ModelConfig` now carries an `expected_sha256` field (wired from
`configs/models.yaml`). `CandidateSpectraAasist3OnnxDetector.load()`
computes the SHA256 of whatever file it is about to load (whether
downloaded via `hf_hub_download` or passed via `local_onnx_path`) and
**raises `RuntimeError` and refuses to construct the ONNX session** if it
does not match `expected_sha256`. Currently configured:

- `spectra_aasist3_onnx`: `expected_sha256 = 5f05c29a01ad80c702b32654db87c2aa6e467c11c67b6d47f2fac873f846cae9` (verified against the real, currently-hosted FP32 artifact)
- `spectra_aasist3_onnx_int8`: `expected_sha256 = 444f832d306a2be4f823119f84e698e8821db6a1aab248593d4b05b7a9a48108`, but **`repository`/`revision` are still the literal placeholder string `PENDING_HF_PUBLISH`** — this config entry cannot successfully download anything yet, by design, until Section 4's manual step is done.

A test (`test_load_rejects_local_file_with_wrong_sha256`) proves the
rejection behavior works, using a deliberately corrupted local file — no
network access required.

## 6. Production dependency audit (Step 11)

Audited every module import chain reachable from `streamlit_app.py` →
`app/model_loader.py` → `audio_deepfake_detector.models.registry` →
`candidate_e.py`. Findings:

- `registry.py` already resolves adapter classes **lazily**
  (`importlib.import_module` inside `create_detector`, not at module top
  level) — this was already correct before this phase.
- `candidate_a.py`/`candidate_b.py` (CAA, Sara) and `candidate_c.py`
  (blocked AntiDeepfake) import `torch`/`transformers` at **module top
  level** — but since they are only imported when
  `create_detector("caa_wav2vec2" | "sara_wav2vec2" | "antideepfake_wav2vec2_small", ...)`
  is actually called, and this branch's `DEPLOYMENT_MODEL_ID` is
  `spectra_aasist3_onnx_int8`, they are never touched by the production
  path.
- `candidate_d.py`/`candidate_e.py` (the two ONNX-based adapters) already
  imported `onnxruntime`/`huggingface_hub` lazily inside `load()`; the one
  stray unconditional `import torch` in `candidate_e.py` (Section 3) has
  been fixed.

**Proof, not just an audit claim** (Section 8): a genuinely clean virtual
environment containing only the packages in this branch's
`requirements.txt` (no torch, no transformers installed at all) was
created, and `streamlit_app.py` was imported successfully in it.

## 7. Cloud-specific dependency set (Step 12)

`requirements.txt` (repository root, this branch only) was rewritten to
contain only what the Spectra INT8 ONNX path needs:

```
huggingface_hub==1.28.0
onnxruntime==1.29.0
numpy==2.5.2
scipy==1.18.1
soundfile==0.14.0
librosa==1.0.0
pydantic==2.13.4
pydantic-settings==2.15.0
PyYAML==6.0.3
psutil==7.2.2
streamlit==1.62.0
matplotlib==3.11.1
```

`torch`, `torchaudio`, `transformers`, and `safetensors` are **removed**
from this branch's `requirements.txt` — they remain real dependencies for
the research adapters (still fully present in
`src/audio_deepfake_detector/models/`, installable locally via `pip
install -e ".[dev]"` or the project's main `.venv`), just not needed by
this branch's *deployed* path. `packages.txt` is unchanged (`ffmpeg`
only).

## 8. Clean cloud-startup verification (Step 13)

Two independent checks, both passing:

1. **In this project's dev `.venv`** (which does have torch/transformers
   installed, for research work): `streamlit_app.py` was imported
   directly (`importlib.util.spec_from_file_location` +
   `exec_module`), and `sys.modules` was checked immediately afterward —
   `torch`, `transformers`, `torchaudio`, `fairseq` were **all absent**.
2. **In a genuinely fresh, isolated virtual environment** (`python -m
   venv`, nothing pre-installed), `pip install -r requirements.txt` was
   run, confirmed via `pip list` that torch/transformers/fairseq are
   **not present anywhere in that environment**, and `streamlit_app.py`
   still imported and ran successfully. This is the strongest available
   proof short of an actual Streamlit Cloud deploy.

## 9. Long-audio / full-clip behavior (Step 15)

The Streamlit app's existing 30-second upload limit
(`app/validation.py`) is unchanged. `predict_full_clip()` (already
implemented and tested in `candidate_e.py` from the prior phase) remains
available as **this project's own application-level extension** —
sequential non-overlapping 64,600-sample windows, never attributed to the
Spectra-AASIST3 authors. The current `streamlit_app.py` calls
`detector.predict()` (the author-compatible single-window path), matching
the "deliberately selected implementation" the phase instructions
permit — the full-clip aggregation experiment (prior phase) found no
measurable benefit from multi-window aggregation over the single-window
score on the In-the-Wild evaluation set, so keeping the simpler,
already-tested single-window path for this validation candidate is a
reasoned choice, not an oversight. Per-window scores remain available
internally via `predict_full_clip()` for any future UI work.

## 10. Result semantics (Step 16)

No UI redesign was performed (out of scope for this phase, and not
needed for cloud validation). The existing UI's language was not
audited/rewritten in this pass beyond what Section 9 required; this is
flagged as follow-up work for the eventual UI redesign phase, not silently
done here.

## 11. Sara preserved (Step 17)

`candidate_b.py`, `docs/inference_calibration.md`,
`results/metrics/calibration_*.json`, and
`results/metrics/spectra_aasist3_sara_comparison.json` are all untouched.
`sara_wav2vec2` remains a fully registered, loadable model
(`create_detector("sara_wav2vec2", ...)` still works) — it is simply not
`DEPLOYMENT_MODEL_ID` on this branch.

## 12. Tests added (Step 18)

- `tests/test_candidate_e.py`: threshold auto-selection
  (FP32 vs INT8), `expected_sha256` wiring, SHA256-mismatch rejection,
  local-path-override integration test (existing, still passing).
- `tests/test_app_model_loader.py`: rewritten for the new
  `DEPLOYMENT_MODEL_ID`, confirms Sara remains registered but is not the
  deployment model.
- `tests/test_deployment_dependencies.py` (new): `requirements.txt`
  excludes torch/transformers/fairseq/CUDA by name; includes
  onnxruntime/huggingface_hub/streamlit; `packages.txt` is exactly
  `ffmpeg`; the production import path adds no banned module to
  `sys.modules`.
- `tests/test_streamlit_app.py` (existing, unmodified, still passing):
  app renders without error, no model load on initial render.

All of the above run in the **default** suite (no network, no large
download). The existing `test_candidate_e.py` integration tests
(model download + real inference) are unaffected and still pass.

## 13. Resource revalidation against the hosted artifact (Step 19) — PENDING

Cannot be performed meaningfully until Section 4 (HF publish) is done —
there is no real hosted artifact to download and benchmark yet. The prior
phase's local-artifact benchmark
(`docs/spectra_production_optimization.md`, Sections 8–10) remains the
best available reference: ~770MB RSS after load, ~821MB complete-app peak
RSS, ~6.5s cold load, ~639ms per-window, ~5.27s for a real 30-second clip.
This section will be updated with the post-publish, hosted-artifact
numbers once Section 4 is complete.

## 14. Architecture (target, once published)

```
Browser
   |
   v
Streamlit Community Cloud (Debian Linux, Python 3.12)
   |
   v
streamlit_app.py + app/  (presentation layer, unchanged from main except DEPLOYMENT_MODEL_ID)
   |
   v
audio_deepfake_detector.models.registry -> candidate_e.py (spectra_aasist3_onnx_int8)
   |
   v
huggingface_hub.hf_hub_download(<HF_USERNAME>/spectra-aasist3-int8-audio-deepfake, revision=<pinned SHA>)
   -> SHA256 verified against expected_sha256 before the ONNX session is ever constructed
   |
   v
onnxruntime.InferenceSession, CPUExecutionProvider only, intra_op_num_threads=2
   |
   v
Spectra-AASIST3 INT8 (351/417 MatMul quantized) -> raw logits -> calibrated threshold -> prediction
```

- **No GPU** anywhere in this path.
- **No PyTorch/Transformers** production dependency (Sections 6–8).
- **No private HF token** needed at runtime — the target repository is
  public (Step 6), so `hf_hub_download` works with no `HF_TOKEN` /
  Streamlit secret configured.
- **No model binary in GitHub** — the 364MB INT8 file lives only on
  Hugging Face and in the local, gitignored `models/cache/`.
- **SHA256 integrity verification** happens before every session
  construction (Section 5).
- Model loading is **lazy** — `st.cache_resource` at the UI layer only,
  triggered by the "Analyze Audio" button, never on page load (Section 8,
  and the existing `test_app_initial_render_does_not_load_model` test).

## 15. Second Streamlit deployment coordinates (Step 22 — prepared, not deployed)

- **Repository**: `munz-aftab/audio-deepfake-detection-fyp`
- **Branch**: `feat/spectra-streamlit-candidate`
- **Main file**: `streamlit_app.py`
- **Python version**: 3.12
- **Suggested app name**: `audio-deepfake-spectra-test` (or similar) — a
  **second, temporary** app, deployed alongside (not replacing) the live
  main-branch app.

Actually creating this second app on Streamlit Community Cloud is a
manual step on the Streamlit Cloud dashboard (choosing the repo/branch/
main-file above) — not something this session can do without the user's
Streamlit Cloud account access, and is not attempted here.

## 16. Reproducibility

```
git checkout feat/spectra-streamlit-candidate
.venv\Scripts\python.exe -m pytest -m "not integration and not slow"
.venv\Scripts\python.exe -m pytest -m integration tests\test_candidate_e.py
```

No ONNX binaries, HF tokens, model caches, datasets, or user/temp audio
are committed.
