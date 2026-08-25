# Phase 5 — Generalization-Focused Replacement Model Evaluation

Branch: `feat/generalization-model-eval` (child of `feat/inference-calibration`).
`main` is unmodified by this phase. The live Streamlit deployment is unchanged.

## 1. Why a replacement was investigated

Phase 4 (`docs/inference_calibration.md`, branch `feat/inference-calibration`,
commit `99e6ecc`) measured the deployed `sara_wav2vec2` detector
(`Sara1708/deepfake-audio-wav2vec2`) on a 320-clip calibration set built
from ASVspoof2019-LA dev (in-domain) plus LibriSpeech test-clean and
WaveFake (out-of-domain):

- baseline accuracy 0.791, combined FPR 7.5%, EER 17.5%, ROC-AUC 0.931
- **in-domain bonafide FPR 0.0% vs. out-of-domain (LibriSpeech) bonafide
  FPR 20.0%** — no threshold reduced OOD bonafide FPR to <=5% without
  destroying spoof recall
- VAD filtering and alternative aggregation strategies did not help
- Decision gate outcome: **C** — `sara_wav2vec2` alone is not judged
  sufficiently reliable for real-world deployment

That result reproduces, with real evidence, the false-positive behavior
the user observed on a real-world MP3 clip, and points at domain
generalization (not windowing/aggregation/threshold choice) as the root
cause. This phase evaluates whether a model explicitly post-trained for
cross-domain generalization performs better.

## 2. Candidate rationale

Candidate: `nii-yamagishilab/wav2vec-small-anti-deepfake` (the default,
data-augmented variant — NOT the `-nda` variant). Chosen because it is
explicitly designed and evaluated for cross-domain generalization
(zero-shot performance on datasets never seen during training/post-training),
which is exactly the failure mode Phase 4 found in `sara_wav2vec2`.

## 3. AntiDeepfake architecture, training scale, verification trail

Verified 2026-08-26 against: the model's Hugging Face card
(`nii-yamagishilab/wav2vec-small-anti-deepfake`, resolved revision
`9a13264b5dcc827a8d5a4f8e01fccefa392f886b`), the official GitHub repository
(`github.com/nii-yamagishilab/AntiDeepfake`), and the associated paper
(arXiv:2506.21090, "Post-training for Deepfake Speech Detection").

- **Repository**: `nii-yamagishilab/wav2vec-small-anti-deepfake`
- **Resolved revision**: `9a13264b5dcc827a8d5a4f8e01fccefa392f886b`
- **Checkpoint**: `model.safetensors`, exactly **380,210,632 bytes**
  (measured via HTTP HEAD against the HF resolve URL, not estimated)
- **Base model**: `facebook/wav2vec2-base` (12 layers, hidden size 768,
  12 attention heads — matches the repo's `config.json`)
- **Parameter count**: ~95M (as documented on the model card)
- **Architecture — IMPORTANT, verified from source, not the model card
  prose**: this is **not** a standard `transformers`
  `Wav2Vec2ForSequenceClassification`. The HF `config.json`'s
  `"architectures": ["Wav2Vec2ForPreTraining"]` field is misleading for
  this checkpoint. The model's own published inference script (in the HF
  README) defines a custom `DeepfakeDetector(torch.nn.Module,
  PyTorchModelHubMixin)` wrapping a **fairseq** SSL frontend
  (`fairseq.models.wav2vec.Wav2Vec2Model`, not `transformers`), an
  `AdaptiveAvgPool1d` pooling layer, and a `Linear(768, 2)` classifier head.
  Full architecture and preprocessing code is reproduced verbatim in
  `src/audio_deepfake_detector/models/candidate_c.py`'s module docstring.
  This is exactly the kind of case CLAUDE.md warns about ("never trust a
  model card's prose alone for architecture") — the prose and the
  `config.json` field both point away from the actual implementation.
- **Sample rate**: 16kHz
- **Length handling**: arbitrary length (author-documented; internally
  batches up to 100s per the HF card). No fixed-length windowing is
  applied by the model itself.
- **Preprocessing** (from the official script): mono-mix (mean over
  channels), resample to 16kHz, then per-utterance `layer_norm` over the
  raw waveform (not mean/std normalization against a fixed statistic).
- **Label mapping** (verified from the official script's own print
  statement — `f"real prob = {prob[1]:.3f}, fake prob = {prob[0]:.3f}"`,
  not assumed): **index 0 = fake/spoof, index 1 = real/bonafide.**
- **Data augmentation**: RawBoost series, applied in this (default)
  variant. The `-nda` (no-data-augmentation) variant was intentionally
  not used.
- **License**: CC BY-NC-SA 4.0 (see Section 4).
- **Documented limitations**: none stated explicitly on the model card
  beyond the license's non-commercial restriction; the paper reports
  markedly worse performance on `Deepfake-Eval-2024` (EER up to 27.77%)
  than on `In-the-Wild` (EER 1.23%), i.e. the authors' own results show
  this model's generalization is domain-dependent, not universal.

### Training/post-training data (AUTHOR-REPORTED, from arXiv:2506.21090 Table I)

~56,000 hours genuine speech + ~18,000 hours fake/artifact speech, 100+
languages. Datasets explicitly listed as used for post-training include:
ASVspoof2019-LA, ASVspoof2021-LA, ASVspoof2021-DF, ASVspoof5, CFAD, DECRO,
DFADD, Diffuse-or-Confuse, DiffSSD, DSD, HABLA, MLAAD, SpoofCeleb, VoiceMOS,
CVoiceFake, LibriTTS, LibriTTS-Vocoded, VoxCeleb2, VoxCeleb2-Vocoded,
WaveFake, FLEURS, FLEURS-R, LibriTTS-R, Codecfake/CodecFake, AISHELL3,
CNCeleb2, MLS.

### Evaluation-only datasets (AUTHOR-REPORTED, per the paper's own statement: "None of them were used for pre-training and post-training")

ADD2023 (Mandarin, author EER 4.67%), DEEP-VOICE (author EER 2.27%),
FakeOrReal (author EER 0.46–0.63%), In-the-Wild (author EER 1.23%),
Deepfake-Eval-2024 (author EER 8.29–27.77%).

**These author-reported numbers are quoted for context only. They are not
project-measured and must not be presented as this project's results.**

## 4. License / attribution

CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International):

- **Attribution required**: any use of this model must credit Yamagishi
  Lab, National Institute of Informatics (NII), and cite arXiv:2506.21090.
- **NonCommercial**: the model may not be used for commercial purposes.
  A non-commercial university Final Year Project demo is consistent with
  this restriction, but any future commercialization of this project would
  require either removing this model or obtaining a separate license from
  the authors. This is a statement about the license terms as written, not
  a legal opinion.
- **ShareAlike**: derivative works built on this model (e.g. a fine-tuned
  version, or a redistributed adapted checkpoint) would need to be
  released under the same CC BY-NC-SA 4.0 terms. This project does not
  fine-tune or redistribute the checkpoint, so ShareAlike is not currently
  triggered, but this constraint should be kept in mind if that changes.
- This model is currently **disabled** (`enabled: false` in
  `configs/models.yaml`) and not used in the deployed app, so no
  attribution is yet required in the live UI. If it is ever enabled for
  deployment, the Streamlit UI and README should credit NII Yamagishi Lab
  and link the paper/repository per the license.

## 5. Adapter implementation — STATUS: BLOCKED

A new adapter, `antideepfake_wav2vec2_small`, was added to
`configs/models.yaml` (`enabled: false`) and implemented in
`src/audio_deepfake_detector/models/candidate_c.py`
(`CandidateAntiDeepfakeWav2Vec2Detector`), registered in
`src/audio_deepfake_detector/models/registry.py`. It faithfully reproduces
the official inference script's classes (`SSLModel`/`_SSLModel`,
`DeepfakeDetector`/`_DeepfakeDetector`) and forward pass, is CPU-only,
loads lazily, calls `.eval()` and uses `torch.inference_mode()`, has no
Streamlit coupling, and pins the exact revision above.

**It cannot currently be loaded in this project's environment.** The
official/only documented inference path requires
`fairseq.models.wav2vec.Wav2Vec2Model`. Two real, reproducible install
attempts against this project's `.venv` (Python 3.12.10, Windows, pip
25.0.1) both failed on 2026-08-26:

1. `pip install fairseq` — fails at the build-requirements stage with
   `FileNotFoundError: [Errno 2] No such file or directory:
   'fairseq\version.txt'` while building the `fairseq==0.12.1` sdist. This
   is a broken upstream PyPI sdist (a known packaging bug in fairseq
   itself), not something specific to this project's setup.
2. `pip install git+https://github.com/facebookresearch/fairseq.git` —
   also fails: its build-time dependencies (torch, numpy, cython, etc.)
   install successfully, but the `fairseq` package's own build step then
   errors out.

No verified ONNX export or `transformers`-native re-implementation of this
specific checkpoint could be found (a targeted search turned up none).

**Per CLAUDE.md policy** ("never invent an architecture merely to make a
checkpoint's `state_dict` load"), a hand-rolled fairseq-to-`transformers`
key-name-mapping reimplementation was **deliberately not attempted**: with
no working fairseq install available in this environment, there is no
reference output to verify numerical parity against, so any such
reimplementation would be unverifiable and could silently produce wrong
predictions — worse than not integrating the model at all.

**Consequence**: no real inference was ever run with this model in this
project. No accuracy, EER, ROC-AUC, F1, latency, or memory numbers for
`antideepfake_wav2vec2_small` are reported anywhere in this project,
because none were measured. This is recorded, not worked around, per
CLAUDE.md's "do not fabricate evaluation results, metrics, or benchmark
numbers" and "if a model hasn't been run... say so explicitly."

### What this blocks (Steps 6, 9–14 of the phase instructions)

- CPU deployment benchmark (cold load time, RSS before/after, 4/10/20/30s
  latency) — **not measured**. Only the checkpoint file size (380,210,632
  bytes, measured via HTTP HEAD) is known.
- Valid-generalization-dataset evaluation (accuracy, balanced accuracy,
  precision, recall, F1, ROC-AUC, EER, FPR, FNR, confusion matrix) — **not
  run**, because there is no working model to evaluate. (Dataset selection
  itself — Section 6 below — was still done, since it doesn't require the
  model.)
- Threshold analysis / calibration — **not run** (no scores exist).
- Inconclusive-zone derivation — **not run**.
- MP3/telephone-band/noise robustness experiment — **not run**.
- Head-to-head comparison vs. `sara_wav2vec2` — **not run**. `sara_wav2vec2`'s
  Phase 4 results remain the only real evidence available; they are
  reported as-is in Section 8, not compared against a number that doesn't
  exist.
- Real-world misclassified-clip qualitative sanity check — could not be
  run either way: the MP3 the user described is not present anywhere in
  this project's working directory (checked; no `.mp3` files exist under
  the repo, and it is not in `data/samples/`), and per instructions it was
  not scraped or invented.

## 6. Valid clean evaluation dataset selection (research only — done, unused pending a working model)

The paper's Table I (Section 3 above) confirms **all** of the datasets the
user flagged as likely-overlapping are in fact confirmed training/
post-training data for this model: ASVspoof2019(-LA), ASVspoof2021(-LA/-DF),
ASVspoof5, WaveFake, MLAAD, DFADD, CFAD, CodecFake, LibriTTS(-derived),
VoxCeleb(-derived), FLEURS(-derived). The Phase 4 calibration set (built
from ASVspoof2019-LA + LibriSpeech + WaveFake) is therefore **rejected as
invalid** for evaluating this candidate — using it would silently leak
training data into a "generalization" measurement.

**Datasets rejected due to confirmed training overlap**: ASVspoof2019-LA,
ASVspoof2021-LA/-DF, ASVspoof5, WaveFake, MLAAD, DFADD, CFAD, CodecFake,
LibriTTS/LibriTTS-Vocoded/LibriTTS-R, VoxCeleb2/VoxCeleb2-Vocoded,
FLEURS/FLEURS-R (i.e. the entire Phase 4 calibration set).

**Selected valid domain**: `mueller91/In-The-Wild` on Hugging Face — a
community-hosted mirror of the well-known "In-the-Wild" audio deepfake
corpus (~20.8h bonafide / ~17.2h spoof speech of 58 public figures,
collected from online video/social media). This is explicitly confirmed by
the AntiDeepfake paper itself as a **held-out evaluation-only** set
("None of them were used for pre-training and post-training"), which is
stronger evidence than inferring non-overlap indirectly. It is also
directly relevant: it is the closest available proxy to the real-world
"news/social-media clip" scenario that originally exposed the
`sara_wav2vec2` false positive.

A second OOD domain (the paper also uses DEEP-VOICE and FakeOrReal as
held-out sets) was investigated but not added: DEEP-VOICE is Kaggle-hosted
only (`birdy654/deep-voice-deepfake-voice-recognition`), not available via
the same streaming/manifest pipeline used elsewhere in this project, and no
FakeOrReal mirror with clear provenance could be confirmed on Hugging Face
in the time available. Per the phase instructions ("do not use a dataset
merely because it is easy to download"), no substitute was used in their
place. **This is a real limitation**: only one clean OOD domain
(In-the-Wild) was identified, not the "at least two" the instructions
preferred.

No download of `mueller91/In-The-Wild` audio was executed in this phase,
because there is no working model to evaluate it against — running the
download step without a usable adapter would just accumulate unused local
data. It remains a documented, ready next step (Section 10).

## 7. Old calibration results

`results/metrics/calibration_*.json`, `results/figures/calibration_*.png`,
and `docs/inference_calibration.md` from Phase 4 are untouched. This
phase's outputs are kept separate: `results/metrics/antideepfake_status.json`
and this document.

## 8. Sara comparison (Step 14)

No new inference was run with either model in this phase. The only real
numbers available are `sara_wav2vec2`'s Phase 4 results on the (for Sara,
valid) ASVspoof2019-LA-dev + LibriSpeech + WaveFake calibration set:
EER 17.5%, combined FPR 7.5% (in-domain bonafide FPR 0%, out-of-domain
bonafide FPR 20%), ROC-AUC 0.931, F1 not separately reported by threshold
in Phase 4's summary (see `docs/inference_calibration.md` for the full
per-threshold table).

That same dataset is **explicitly invalid** for evaluating
`antideepfake_wav2vec2_small` (Section 6), so **no head-to-head comparison
on shared data is possible from work done so far.** Per the phase
instructions, this is reported as **"Not a valid clean comparison"**
rather than reporting a misleading number.

## 9. Streamlit deployment viability (Step 16)

**Gate fails at the first condition (A: generalization evidence) before
condition B (deployment cost) is even reachable**, because no generalization
evidence for this candidate could be measured in this environment. Separately,
condition B carries its own serious risk independent of the modeling result:
Streamlit Community Cloud's build environment would also need to build
`fairseq` from source at deploy time. Given fairseq's confirmed-broken PyPI
sdist and its well-documented history of breaking under newer Python/
`omegaconf`/`numpy` combinations (see search evidence in this phase's
research), there is a real, non-trivial risk that the exact same class of
failure reproduces there — Streamlit Cloud uses Python versions this project
does not control, and fairseq is unmaintained enough that this is not a
"try it and see" I can resolve without deploying (out of scope: this phase
explicitly must not touch the live deployment). This is flagged as a risk,
not asserted as a confirmed second failure.

## 10. ONNX investigation (Step 17)

A targeted search found no official or community ONNX export of this
checkpoint. Producing one would require first getting a working PyTorch
(fairseq-based) reference implementation running to trace/export from —
which is exactly the blocked step. Not pursued further in this phase.

## 11. Limitations

- The single largest limitation of this phase: the candidate model could
  not be run at all, due to an environment-level dependency failure that is
  outside this project's control (an upstream `fairseq` packaging defect).
  Every metric this phase's instructions asked for (Steps 6, 10–14) is
  therefore unmeasured, not merely limited in scope as Phase 4's was.
- Only one clean, non-overlapping OOD evaluation domain (In-the-Wild) was
  identified, not the preferred two.
- The real-world MP3 clip that originally exposed the `sara_wav2vec2`
  false positive is not present in this project and could not be used for
  a qualitative sanity check with either model in this phase.
- Whether `fairseq` would install cleanly in a Linux environment (e.g. via
  WSL, or Streamlit Community Cloud's actual build container) was not
  tested — doing so would require provisioning a new environment/WSL
  distribution outside this project directory, which was judged out of
  scope for this phase without explicit user instruction.

## 12. Decision gate

None of the five given outcomes (A–E) precisely describes "the candidate
could not be technically evaluated due to an unresolved dependency," but
the closest fit, and the one with the same practical next step, is:

> **D. AntiDeepfake does not improve enough; evaluate another pretrained
> model.**

chosen **not** because there is evidence it under-performs (there is no
project-measured evidence either way — the author-reported numbers in
Section 3 are, if anything, promising), but because it is currently
**unusable** in this project's environment, which has the identical
practical consequence as D: this candidate does not currently move the
project forward, and the next step should be evaluating a different,
`transformers`-compatible pretrained model rather than continuing to sink
effort into a `fairseq`-dependent one. **Outcome A (replace Sara) is
explicitly not chosen** — there is no reliable, working implementation of
this candidate to replace anything with, and CLAUDE.md/this phase's own
instructions forbid forcing outcome A regardless.

Recommended next step: prioritize `transformers`-native or ONNX-native
generalization-focused candidates (avoiding a `fairseq` dependency
entirely) in the next phase, and continue treating `sara_wav2vec2`'s
Phase 4 Outcome C (with the inconclusive-zone recommendation) as the
current best-evidence status of the deployed model.

## 13. Reproducibility

```
git checkout feat/generalization-model-eval
.venv\Scripts\python.exe -m pytest -m "not integration and not slow"   # 96 passed
.venv\Scripts\python.exe -m pytest -m integration tests\test_candidate_c.py  # confirms the fairseq ImportError
```

No dataset audio or model weights are committed. `mueller91/In-The-Wild`
was not downloaded in this phase (Section 6).
