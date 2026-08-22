# Wav2Vec2 CPU Deepfake Detector — Candidate Analysis

**Status:** Phase 2, pre-benchmark. This document records what is documented
about each candidate model *before* any CPU benchmarking was performed. Do
not read any number in this document as a result produced by this project —
every metric below is explicitly an **author-reported / model-card result**
unless stated otherwise. Actual measured CPU engineering benchmarks are
recorded separately in
[`docs/cpu_model_benchmark.md`](cpu_model_benchmark.md).

All facts below were verified directly against the Hugging Face Hub API,
each repository's raw `README.md`, and (where cited) the author's GitHub
source repository, fetched on 2026-08-23. Where information could not be
verified, it is marked `Not documented`.

---

## Candidate A — `caa-speech-detection-asvspoof2019/wav2vec2-v2-unfrozen`

**Hugging Face repository:** https://huggingface.co/caa-speech-detection-asvspoof2019/wav2vec2-v2-unfrozen
**Resolved revision (sha):** `71250c8366e164e292e19a0451d85747038d2ed3`
**License:** MIT

### Architecture (author-documented, in `README.md` and `config.yaml`)

- Base model: `facebook/wav2vec2-base`
- Top 4 transformer blocks trainable (`freeze_last_n: 4`); remaining encoder frozen
- Classification head: `Linear(768→256) → GELU → Dropout(0.1) → Linear(256→2)`
- Total parameters: ~95M (base backbone) plus head; top-4 blocks + head trainable
- Input: raw waveform, 16 kHz, **padded/truncated to 64,000 samples (4 s)**
- Forward pass (from README usage example) takes a dict `{"frames": waveform}`
  and returns `{"logits": ...}`

### Training data (author-documented)

- ASVspoof 2019 LA train split (~25,380 utterances, per `summary.json`)
- No data augmentation
- Class weights `[3.0, 1.0]`, AdamW, cosine LR schedule, 40 epochs configured
  (stopped early at epoch 35, `early_stopping_patience: 12`)

### Checkpoint

- Filename: `best.pt`
- **Measured size (verified via direct HTTP `Content-Length` on the resolved
  LFS/Xet object): 606,800,672 bytes (~579 MiB / ~607 MB)**

### Label mapping (author-documented, from README usage snippet)

```
probs = softmax(logits)  # probs[:, 0] = bonafide, probs[:, 1] = spoof
```

Index 0 = bonafide, index 1 = spoof. **Documented directly in the model
card's usage example** — not inferred.

### Author-reported/model-card results

| Split | EER | tandem min t-DCF | In-the-Wild EER |
|---|---|---|---|
| Dev | 0.197% | — | — |
| Eval | 1.55% | 0.3966 | 27.68% |

Baseline comparison quoted by the author: LFCC+GMM EER 8.09%.

### Documented limitations (from README)

- Trained on ASVspoof 2019 LA only; not validated on physical-access or other corpora
- In-the-Wild EER of 27.68% — the author explicitly flags limited generalisation
  beyond the ASVspoof 2019 attack pool
- ~362–579 MB checkpoint (author's stated ~362 MB figure is for the trainable
  subset description; the actual downloadable file is ~579 MiB) — author
  states this is "unsuitable for edge or real-time deployment" and recommends
  a much smaller alternative model (`lcnn_v7_cqt`, ~3.5 MB) for
  resource-constrained scenarios
- No data augmentation used; domain shift may degrade real-world performance

### Source repository verification — **CRITICAL FINDING**

The model card cites a source implementation at
`github.com/sebastiaoteixeira/caa-ai-generated-speech-detector`
(`src.models.wav2vec2.model.Wav2Vec2Model`).

**This repository does not exist.** Verified 2026-08-23 via direct HTTP
request: `https://github.com/sebastiaoteixeira/caa-ai-generated-speech-detector`
returns **HTTP 404**, and the GitHub REST API confirms
`{"message": "Not Found", "status": "404"}`.

This means the exact `Wav2Vec2Model` wrapper class referenced in the model
card's usage example **cannot be independently verified from its documented
source**. Per project policy, we do **not** invent this architecture from
scratch to force the checkpoint to load. Instead:

- The classification head architecture is explicitly documented inline in
  the model card itself (`Linear(768→256) → GELU → Dropout(0.1) → Linear(256→2)`),
  which is a strong, specific claim, not a guess.
- Before any adapter is written for this model, its checkpoint's
  `state_dict` keys/shapes are inspected directly (see
  `results/model_manifest.json` and the loading-attempt log). A module tree
  is reconstructed **only from what the checkpoint's own keys require**,
  cross-checked against the documented head description — not fabricated
  independently of that evidence.
- If the state_dict keys cannot be reconciled with the documented head
  architecture with confidence, this candidate is recorded as a **load
  failure**, not forced into a best-effort guess.

**Verification outcome (measured, 2026-08-23):** the downloaded checkpoint
(`best.pt`, confirmed 606,800,672 bytes, matching the documented size
exactly) contains 215 `state_dict` keys: `class_weights` (shape `[2]`),
`classifier.0.{weight,bias}` (`Linear(768, 256)`), `classifier.3.{weight,bias}`
(`Linear(256, 2)`), and 211 `encoder.*` keys forming a
`facebook/wav2vec2-base`-shaped backbone. This exactly matches the
documented head (`Linear(768→256) → GELU → Dropout(0.1) → Linear(256→2)`
— indices 1 and 2 are parameter-free GELU/Dropout, consistent with the gap
between `classifier.0` and `classifier.3`). Loading this checkpoint into a
module reconstructed from these keys is missing exactly one parameter,
`encoder.masked_spec_embed` (a SpecAugment-style masking embedding used only
during masked pretraining/fine-tuning), and has **zero** unexpected keys.
The model card states encoder masking is disabled at inference, so this
gap is consistent with documented behavior rather than an architecture
mismatch. **Load succeeded** with this one explicitly-verified, allow-listed
exception (see `src/audio_deepfake_detector/models/candidate_a.py`, which
raises rather than silently loading if any *other* key mismatch appears).

**Remaining unverified detail:** the pooling operation applied to the
encoder's per-frame outputs before the classifier head has no learnable
parameters, so it leaves no trace in the checkpoint and cannot be confirmed
without the missing source repository. This adapter assumes **mean pooling
over time** — the same operation Candidate B's verified source code
actually uses — as a documented, explicitly-flagged assumption, not a
verified fact. CPU inference on deterministic smoke audio completed
successfully under this assumption (see
`docs/cpu_model_benchmark.md`), which confirms the tensor shapes and
execution path work end-to-end, but does **not** confirm the pooling
assumption is correct.

### Streamlit / deployment suitability (preliminary, engineering judgement only)

- ~579 MB checkpoint download is large for a Streamlit Cloud free-tier
  deployment (slow cold start, larger memory footprint)
- Loading procedure requires a custom module class; without a working source
  reference, reproducibility risk is elevated
- License (MIT) is permissive and deployment-friendly

---

## Candidate B — `Sara1708/deepfake-audio-wav2vec2`

**Hugging Face repository:** https://huggingface.co/Sara1708/deepfake-audio-wav2vec2
**Resolved revision (sha):** `6c43629c953d6ff008501bf5f3eb983ac2321ad6`
**License:** Apache 2.0

### Architecture (verified against the actual source code, not just the model card)

Source repository `github.com/Saracasm/deepfake-audio-detection` **was
confirmed to exist (HTTP 200)** and was inspected directly:
[`src/models/wav2vec_classifier.py`](https://github.com/Saracasm/deepfake-audio-detection/blob/main/src/models/wav2vec_classifier.py).

```python
class Wav2VecClassifier(nn.Module):
    def __init__(self, backbone_name="facebook/wav2vec2-base", num_classes=2, freeze_backbone=True):
        self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)   # transformers.Wav2Vec2Model
        hidden_size = self.backbone.config.hidden_size                 # 768
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, waveforms):
        outputs = self.backbone(waveforms)
        pooled = outputs.last_hidden_state.mean(dim=1)   # mean-pool over time
        return self.classifier(pooled)                    # logits (B, 2)
```

- Base model: `facebook/wav2vec2-base` (95M params, 12 transformer layers)
- Head: mean-pool over time + `Linear(768 → 2)` — no hidden layer, unlike Candidate A
- Stage 2 checkpoint (`stage2_best.pt`, the published checkpoint): top 2
  transformer layers + final encoder LayerNorm unfrozen (~14M trainable params)
- Input: raw waveform, 16 kHz, 4-second windows (64,000 samples)
- This architecture is loadable with **exact confidence** because the actual
  class definition was retrieved from the cited GitHub repository, not
  reconstructed from a checkpoint alone.

### Windowing / padding (verified from source: `src/data/preprocessing.py`)

- `WINDOW_SAMPLES = 64000` (4 s @ 16 kHz), `HOP_SAMPLES = 32000` (50% overlap)
- Clips ≤ one window are **zero-padded** to exactly one window
- Longer clips are split into overlapping 64,000-sample windows with a
  32,000-sample hop; the final partial window is zero-padded
- This is the documented windowing strategy this project's adapter follows
  for Candidate B specifically (see Step 16 windowing design)

### Training data (author-documented)

- ASVspoof 2019 LA training partition (25,380 utterances)
- Class weighting: bonafide=4.92, spoof=0.56 (~9:1 imbalance compensation)
- AdamW, lr 1e-5, 10% warmup + linear decay, batch size 16, fp16, 10 epochs
  (best at epoch 9), ~2h56m on a single T4 GPU

### Checkpoint

- Filename: `stage2_best.pt`
- **Measured size (verified via direct HTTP `Content-Length`): 491,044,441
  bytes (~468 MiB / ~491 MB)**

### Label mapping (verified directly from source code, not the README prose)

From `src/inference/predict.py`:

```python
probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()  # (n_windows, 2)
window_spoof_probs = probs[:, 1].tolist()
```

**Index 1 = spoof, index 0 = bonafide.** This was confirmed from the actual
inference source code, which is stronger evidence than the README's prose
description alone (the README does not spell out index order explicitly;
the source code does).

### Author-reported/model-card results

| Evaluation | EER |
|---|---|
| ASVspoof 2019 LA dev (seen attacks A01–A06) | 0.69% |
| ASVspoof 2019 LA eval (unseen attacks A07–A19) | 5.55% |
| ASVspoof 2021 LA eval (codec-degraded) | 9.09% |
| WaveFake (LJSpeech vocoders, mean) | 29.4% |

The author explicitly claims the ASVspoof 2021 LA result is competitive with
published baselines (LFCC-LCNN 9.26%, RawNet2 9.50%) — this is a
**documented cross-dataset generalisation claim**, which Candidate A's model
card does not provide (Candidate A only reports ASVspoof 2019 + one
"In-the-Wild" number).

### Documented limitations (from README)

- Poor generalisation to standalone neural vocoders not seen in ASVspoof
  (WaveFake mean EER ~29.4%)
- Codec sensitivity: lossy telephone codecs (GSM, PSTN) degrade performance
  ~6 percentage points
- A10 attack family is a specifically documented weak point (15.54% EER)
- Explicitly self-described as "a research artifact, not a production
  deepfake detector"

### Streamlit / deployment suitability (preliminary, engineering judgement only)

- ~468 MB checkpoint — somewhat smaller than Candidate A but still large for
  a free-tier Streamlit Cloud deployment
- Loading procedure is **fully reproducible**: exact model class, exact
  windowing/padding logic, and exact label index order are all confirmed
  from real, working source code
- License (Apache 2.0) is permissive and deployment-friendly
- Author explicitly provides ASVspoof 2021 (cross-dataset) evidence, which
  is directly relevant to this project's eventual robustness/generalisation
  goals

---

## Optional engineering reference — `mo-thecreator/Deepfake-audio-detection`

**Hugging Face repository:** https://huggingface.co/mo-thecreator/Deepfake-audio-detection
**Resolved revision (sha):** `e4d9874b493362149cec96ced85f00b00b1a04c0`
**License:** Apache 2.0

- Base model: `mo-thecreator/wav2vec2-base-finetuned` (a fine-tuned
  `facebook/wav2vec2-base` derivative)
- `config.architectures = ["Wav2Vec2ForSequenceClassification"]`,
  `transformersInfo.auto_model = "AutoModelForAudioClassification"` —
  loadable directly with standard `transformers` `AutoProcessor` +
  `AutoModelForAudioClassification`, no custom source code required
- Checkpoint format: `model.safetensors`, 94,569,090 parameters (F32),
  ~1.89 GB repo storage (safetensors weight file itself is the dominant
  contributor)
- Training dataset, exact label mapping, and evaluation metrics: **Not
  documented** in the model card (`model-index.results` is empty; card body
  was not inspected for prose beyond the structured metadata in this pass —
  to be filled in if this reference is actually benchmarked)
- Per project instructions, this model is retained only as an **engineering
  reference** for standard Transformers-integration patterns (`AutoProcessor`
  / `AutoModelForAudioClassification`). It is not treated as a primary
  research candidate because of its materially weaker documentation for
  dissertation-grade scientific comparison (no reported EER, no documented
  training corpus provenance beyond the base-model name).

---

## Comparison table (documentation only — no benchmarking yet)

| | Candidate A | Candidate B | Reference (optional) |
|---|---|---|---|
| HF repo | `caa-speech-detection-asvspoof2019/wav2vec2-v2-unfrozen` | `Sara1708/deepfake-audio-wav2vec2` | `mo-thecreator/Deepfake-audio-detection` |
| Base architecture | `facebook/wav2vec2-base`, top-4 layers unfrozen, MLP head | `facebook/wav2vec2-base`, top-2 layers unfrozen, linear head | `facebook/wav2vec2-base`-derived, `Wav2Vec2ForSequenceClassification` |
| Parameters | ~95M | ~95M | 94,569,090 |
| Checkpoint size (measured) | 606,800,672 bytes (~579 MiB) | 491,044,441 bytes (~468 MiB) | Not measured (safetensors, repo storage ~1.89 GB incl. non-weight files) |
| License | MIT | Apache 2.0 | Apache 2.0 |
| Training dataset | ASVspoof 2019 LA | ASVspoof 2019 LA | Not documented |
| Published in-domain result | Eval EER 1.55% (author-reported) | Eval EER 5.55% (author-reported) | Not documented |
| Published cross-dataset evidence | "In-the-Wild" EER 27.68% (author-reported, single number, dataset not fully specified) | ASVspoof 2021 LA EER 9.09%, WaveFake mean EER 29.4% (author-reported, two named external evaluations) | Not documented |
| Input requirements | 16 kHz, 64,000 samples, pad/truncate | 16 kHz, 64,000-sample windows, 50% overlap, zero-pad short clips | Not documented (assume standard `AutoProcessor` framing) |
| Label mapping evidence | Documented in README usage snippet | Confirmed directly from source code | Not documented |
| Loading complexity | High — cited source repo is 404; requires reconstructing from checkpoint keys + documented head | Low — cited source repo verified working, exact class available | Low — standard `AutoModelForAudioClassification` |
| CPU compatibility | Not yet benchmarked (Phase 2 pending) | Not yet benchmarked (Phase 2 pending) | Not yet benchmarked / not in scope unless needed |
| Estimated Streamlit suitability | Large checkpoint, unresolved source reproducibility risk | Large checkpoint, but fully reproducible loading path | Reference only; not evaluated for deployment |
| Major limitations | Source repo missing; author itself calls the checkpoint unsuitable for edge/real-time use | Weak generalisation to non-ASVspoof vocoders; explicitly "research artifact, not production" | Missing metrics/dataset documentation |

**No final winner is declared in this document.** A final CPU benchmarking
pass (Step 14 of the Phase 2 plan) is required before any deployment
recommendation — see [`docs/cpu_model_benchmark.md`](cpu_model_benchmark.md)
and the preliminary recommendation section of the Phase 2 completion report.
