# Spectra-AASIST3 Evaluation

Branch: `feat/spectra-aasist3-eval` (child of `feat/generalization-model-eval`).
`main` is unmodified. The live Streamlit deployment is unchanged.

## 1. Rationale

The prior candidate investigated in this phase, `antideepfake_wav2vec2_small`
/ its ONNX mirror `antideepfake_wav2vec2_onnx`, ended at **Outcome E**: the
ONNX artifact loaded and ran correctly on CPU, but its crop/pad
preprocessing contract was not published anywhere findable, so no
scientific evaluation was performed and it must not become the production
detector (`docs/replacement_model_evaluation.md`, Section 14).

`lab260/Spectra-AASIST3` was investigated next because, unlike that
candidate, its model card documents its preprocessing recipe in prose
(preemphasis coefficient, fixed window length, padding behavior) even
though — as with the previous candidate — the runnable wrapper file the
card links to does not actually exist in the repository. This makes it a
meaningfully different case: reproducible in principle from the card text,
not blocked at the same step.

## 2. Architecture (verified from source, not model-card prose alone)

Verified directly from the model's own vendored `model.py` (32,424 bytes,
downloaded and read in full — not summarized):

- **wav2vec 2.0 XLS-R-300m front-end** — standard HuggingFace
  `transformers.Wav2Vec2Model` (`facebook/wav2vec2-xls-r-300m`), **not**
  `fairseq` (this candidate has no `fairseq` dependency at all, unlike the
  previous two candidates in this phase).
- **MLP bridge** — single-layer `Linear(1024 → 128)` with SELU activation
  and dropout 0.1.
- **KAN-AASIST back-end** — a RawNet2-style residual 2D-conv encoder,
  spectral (`GAT_layer_S`) and temporal (`GAT_layer_T`) graph-attention
  layers, graph pooling, four parallel `HSGALBranch_v1` inference branches
  combined via element-wise `amax`, and a Kolmogorov-Arnold Network
  (`KANLinear`, B-spline-based) 2-class output layer.
- **Output**: 2 logits. **Index 1 = bona fide**, verified directly from
  `model.py`'s own `SpectraAASIST3.classify()` method:
  ```python
  def classify(self, x, threshold: float = -1.0625009):
      x = self.forward(x)[:, 1]
      x = (x > threshold).float()
      return x.item()
  ```
  This confirms index 1 is thresholded directly as the decision score
  (higher = more bona fide), from source code, not just README prose.
- **Parameters**: 319,051,189 (F32), measured via the HF API's own
  `safetensors.parameters` field — close to, but not exactly, the model
  card's stated "~318.95M" (a rounding difference, not a discrepancy worth
  treating as a red flag).

## 3. Unpublished-status limitation (Step 3)

**Spectra-AASIST3 currently has no peer-reviewed or public research paper
associated with this exact model.** It is a pre-release/unpublished
detector — the model card explicitly states "Paper: none — pre-release /
unpublished model" and that it appears in the Speech Anti-Spoofing Arena's
🔓 *Unpublished/Proprietary* tier ("listed but unranked, regardless of
score").

This does **not** automatically prohibit academic experimentation. However:

- External Arena/benchmark results (Section 9) are **supporting evidence
  only**.
- This project independently measures its own performance (Sections
  8–11) rather than citing the Arena numbers as project results.
- It must not be described as a published state-of-the-art architecture in
  this project's own writing (this document and any future report should
  say "an unpublished candidate reporting strong Arena benchmark numbers,"
  never "a published state-of-the-art detector").

## 4. Model artifact / revision (verified via direct HTTP/API calls, not WebFetch summarization)

Consistent with this phase's established practice, the repository was
verified directly against `huggingface.co/api` and CDN resolve endpoints,
not taken on faith from a generated summary.

- **Repository**: `lab260/Spectra-AASIST3`
- **Resolved revision**: `bc0ded888080ddad493177bb53aa6f5b95219d7c`
- **License**: Apache-2.0
- **`model.safetensors`**: 1,276,330,956 bytes, SHA256
  `327762caab3106ecee5f26b821ccdc6ab1fe34906a3e7042f416c87927df2088`
- **ONNX artifact**: `spectra-aasist3.onnx`, 1,279,022,864 bytes, SHA256
  `5f05c29a01ad80c702b32654db87c2aa6e467c11c67b6d47f2fac873f846cae9`
  (confirmed identical between the HF API's reported hash and the hash of
  the file actually downloaded into this project's cache)
- **`config.json`**: literally `{}` — no architecture metadata; the real
  architecture is only in `model.py`.
- **Discrepancy found and documented**: the README links to
  `spectra_aasist3.py` and `spectra_aasist3_net.py` as "the exact wrapper
  that produced the Arena scores" / "the vendored network" — **both return
  HTTP 404**. Only `model.py` (32,424 bytes) actually exists in this
  repository. The maintainer's TensorRT export script
  (`trt_spectra_aasist3.py`, commit `86f4581674050ac20bb0ef4b79bcd2f00bb7826e`,
  "Add TensorRT export script + ONNX export") is the same generic
  per-model export template already seen for
  `antideepfake_wav2vec2_onnx` — it imports the (missing)
  `spectra_aasist3` module and states in its own docstring that "all
  preprocessing already lives in the original `score_batch()`," which is
  not shipped here either. The Arena's own application source (checked in
  the prior sub-phase) contains no such framework code.

## 5. Preprocessing — documented, reproduced with one flagged assumption

Unlike `antideepfake_wav2vec2_onnx`, this candidate's preprocessing is
**not** a total blackbox: the model card's own README states, in prose:

> "Preemphasis (0.97) is applied to the full waveform (matching the source
> README eval pipeline), then a deterministic first-64,600-sample window
> (~4.04s; tile-repeat if shorter — no random crop). No resampling in the
> wrapper (audio arrives at expected_sample_rate=16000)."

This project's implementation
(`src/audio_deepfake_detector/models/candidate_e.py`):

1. Decode/resample to 16 kHz mono float32 (existing project audio loader).
2. Pre-emphasis over the **full** waveform: `y[0] = x[0]`, `y[n] = x[n] -
   0.97*x[n-1]` for `n ≥ 1`. **Flagged assumption**: the README does not
   spell out the first-sample edge case explicitly; `y[0] = x[0]` (no
   wrap-around) is the universal DSP convention for this formula (used by
   Kaldi, ESPnet, etc.) and is applied here as such, not silently presented
   as independently verified.
3. Crop to the deterministic first 64,600 samples for clips ≥ 64,600
   samples; `np.tile`-repeat then truncate for shorter clips (never
   zero-pad — verified as a real behavioral difference in
   `tests/test_candidate_e.py::test_window_short_input_tile_repeats_not_zero_pads`).
4. ONNX graph input length (64,600, read directly via
   `onnxruntime.InferenceSession(...).get_inputs()`, not guessed) matches
   this recipe exactly — cross-validated, not assumed.

This adapter's `predict()` therefore **works** (unlike
`candidate_d.py`'s deliberately-blocked `predict()`), producing a real,
project-measured score, subject to the one flagged edge-case assumption
above.

## 6. CPU Resource Gate (Step 9) — first major decision checkpoint

Full numbers: `results/metrics/spectra_aasist3_resource_gate.json`.
Measured on this project's dev machine, CPU-only, `onnxruntime` 1.29.0,
`CPUExecutionProvider` only (no CUDA anywhere in this investigation).

| Metric | Spectra-AASIST3 (ONNX) | `sara_wav2vec2` |
|---|---|---|
| Checkpoint size | 1,279,022,864 bytes (~1.28 GB) | 491,044,441 bytes |
| Cold load time | ~12.5s | 1.79s |
| RSS after load | ~1,666 MB (threads=2) | 1,199 MB |
| Peak RSS | ~1,659–1,666 MB | not separately measured, ≈ load figure |
| Native-window (4.04s) inference | **1,043.8 ms** (threads=2) | **140.2 ms** |

**Gate outcome**: the user-defined heuristic flags "clearly above ~2.2GB"
as high risk and "~2.5GB" as a hard stop for the *large scientific
evaluation*. The measured isolated peak (~1.67GB) is below both lines, so
the gate does **not** automatically block proceeding to Sections 7–11.
However, this is reported honestly as a first-class deployment concern,
not glossed over: Spectra-AASIST3 uses **~1.4x** the resident memory and
is **~7.4x slower per identical-duration window** than `sara_wav2vec2` on
this machine, and cold-loads **~7x** more slowly. A naive 30-second
application-level clip (8 non-overlapping windows) is projected at
**~8.4 seconds** of pure inference time — a materially worse interactive
experience than the deployed model.

`intra_op_num_threads`: 1 thread → 487–1,867 ms/window; 2 threads →
370–1,044 ms/window (roughly 24–44% faster with 2 threads on this dev
machine, consistent with the pattern seen for the smaller
`antideepfake_wav2vec2_onnx` candidate). Streamlit Community Cloud's
actual CPU allocation was not independently confirmed, so this remains a
dev-machine-only data point.

**Decision**: per instructions, this does not automatically reject the
candidate, but "we do not proceed as though deployment is safe" — the
final decision gate (Section 12) treats this resource profile as a
first-class input alongside detection-quality results, not a footnote.

## 7. InTheWild subset (Step 11)

Source: `SpeechAntiSpoofingBenchmarks/InTheWild` (the Arena's own parquet
mirror of the canonical "In-the-Wild" corpus — the same corpus already
used, via a different HF mirror, as the valid OOD domain for
`antideepfake_wav2vec2_small` in the prior sub-phase), pinned to revision
`a957f2582802cdb5964e118818c2e46b3d61aa35` — the exact snapshot
Spectra-AASIST3's own Arena `InTheWild/result.yaml` was computed against
(fetched and confirmed directly: `eer_percent: 1.2022241146120323`,
`n_trials: 31779`, matching the README's "1.20%" badge).

400 clips deterministically sampled (`numpy.random.default_rng(seed=1337)`,
shuffle-then-take-first-200-per-class on the sorted `utterance_id` list):
**200 bonafide, 200 spoof**. Label convention verified directly from the
dataset's own `notes` field (not assumed): `label == 1` → spoof,
`label == 0` → bonafide. Exact utterance IDs recorded in
`data/raw/spectra_inthewild/manifest.json` (gitignored, not committed).

## 8. Calibration / evaluation split (Step 12)

`numpy.random.default_rng(seed=2024)` shuffles the 200 bonafide and 200
spoof clips independently, then splits **before any scores are computed**:

- **Calibration**: 100 bonafide + 100 spoof
- **Evaluation**: 100 bonafide + 100 spoof, **no overlap** (asserted
  programmatically in `scripts/run_spectra_evaluation.py`)

Exact IDs: `results/metrics/spectra_aasist3_split.json` (gitignored).
The split is fixed before any scoring and is never adjusted afterward.

## 9. Author-compatible calibration results (Step 13/14)

Scored with `predict()` (preemphasis on the full waveform, then a
deterministic single first-64,600-sample window — the author-compatible
path, no multi-window aggregation). Full numbers:
`results/metrics/spectra_aasist3_calibration.json`.

| Threshold candidate | threshold | accuracy | bonafide FPR | F1 |
|---|---|---|---|---|
| EER | 0.9300 | — | — | — (EER = **0.0%** on calibration) |
| best balanced-accuracy | 0.785 | 1.000 | 0.000 | 1.000 |
| best F1 | 0.785 | 1.000 | 0.000 | 1.000 |
| low-FPR (≤5%) | 0.005 | 0.980 | 0.040 | 0.980 |

**Caveat, stated plainly**: a 0% EER / 100% accuracy on a 200-clip
calibration set is a strong result, but with this sample size it should be
read as "this candidate separates these particular calibration clips
essentially perfectly," not as a population-level 0% error rate claim. The
frozen evaluation set (Section 10, a disjoint 200 clips never used for
threshold selection) is the more trustworthy generalization signal, and it
shows a non-zero but still very low error rate, which is the expected,
consistent picture.

The EER threshold (0.9300) was the one frozen and carried into Section 10 —
per Step 14/15's explicit instruction to select a threshold before, and
independently of, the evaluation set.

## 10. Frozen evaluation results — PROJECT-MEASURED (Step 15)

Evaluation subset (100 bonafide + 100 spoof, disjoint from calibration),
scored once, at the threshold frozen in Section 9. Full numbers:
`results/metrics/spectra_aasist3_evaluation.json`.

Terminology: **positive class = spoof** (label 1); **bonafide FPR** =
genuine speech incorrectly flagged as fake (the deployment-relevant failure
mode); **spoof FNR** = fake speech incorrectly accepted as genuine.

| Metric | Value |
|---|---|
| Accuracy | 0.935 |
| Balanced accuracy | 0.935 |
| Precision | 0.968 |
| Recall (spoof) | 0.900 |
| F1 | 0.933 |
| ROC-AUC | 0.977 |
| EER (own, on this set) | 0.020 (2.0%) |
| **Bonafide FPR** | **0.030 (3.0%)** |
| Spoof FNR | 0.100 (10.0%) |

Confusion matrix (n=200): TP (spoof correct) 90, TN (bonafide correct) 97,
**FP (bonafide flagged as spoof) 3**, FN (spoof missed) 10.

This is a **project-measured** result on genuinely held-out, previously
unseen (by this project's threshold selection) real-world audio — not the
model card's 1.20% Arena EER, which remains a maintainer-reported number
computed on the full 31,779-clip set and is not presented as this
project's result anywhere in this document.

## 11. Same-clip Sara comparison — the key head-to-head result (Step 16/17)

`sara_wav2vec2` was run, via its existing unmodified production inference
pipeline, on the **exact same 200 evaluation clips**, at its existing
production threshold (0.5). Full numbers:
`results/metrics/spectra_aasist3_sara_comparison.json`.

| Metric | Spectra-AASIST3 | `sara_wav2vec2` (same clips) |
|---|---|---|
| Accuracy | **0.935** | 0.630 |
| Bonafide FPR | **0.030 (3.0%)** | **0.680 (68.0%)** |
| Spoof FNR | 0.100 | 0.060 |
| F1 | 0.933 | 0.718 |
| ROC-AUC | 0.977 | 0.787 |
| EER (own) | 0.020 | 0.250 |

This is the cleanest, most important evidence produced by this phase: on
**identical** real-world (In-the-Wild) audio, run through each model's own
existing pipeline, Sara flags **68% of genuine speech as fake**, closely
reproducing (on an entirely independent, project-selected dataset) the
same real-world false-positive failure mode Phase 4 found on the
ASVspoof2019/LibriSpeech/WaveFake calibration set. Spectra-AASIST3's
bonafide FPR on the same clips is **3%** — more than 20x lower.

**False-positive analysis (Step 17)**: only 3 of 100 bonafide evaluation
clips were misclassified by Spectra. All three are notably short:

| utterance | speaker | duration (s) | spoof score |
|---|---|---|---|
| ITW_12751 | Barack Obama | 0.86 | 0.985 |
| ITW_28864 | Milton Friedman | 0.74 | 0.970 |
| ITW_23117 | George Carlin | 1.67 | 0.999 |

No shared speaker across the 3 false positives — **not** a speaker-specific
failure. All three are well under the 4.04s required window, meaning the
tile-repeat padding (Section 5) repeats a very short segment 4–5 times to
fill the input; this is a plausible contributing factor to these specific
misclassifications, offered as a reasonable hypothesis given the pattern,
not as a proven causal claim from only 3 data points. This should not be
over-interpreted given the small evaluation subset (100 bonafide clips) —
it is a real, project-measured observation, not a generalizable rate
estimate.

## 12. Full-clip aggregation extension — OUR OWN, not attributed to the authors (Step 18)

Full numbers: `results/metrics/spectra_aasist3_full_clip_aggregation.json`.
Four aggregation strategies (first-window, mean, median, majority-vote
across sequential non-overlapping 64,600-sample windows) were compared on
the calibration set only, then the best one was frozen and evaluated once.

| Aggregation | Calibration EER |
|---|---|
| first | **0.0%** |
| mean | 0.5% |
| median | 0.5% |
| majority | 1.5% |

`first` (i.e., the same single-window behavior as the author-compatible
path) tied for best on calibration and was selected. Its frozen evaluation
result is therefore **identical** to Section 10's author-compatible
numbers. **Finding**: this project's multi-window extension provided **no
measurable benefit** over the author-compatible single-window score on
this dataset — most plausibly because a large share of In-the-Wild clips
are shorter than 4.04s to begin with, so "sequential windows" collapses to
a single window for most clips. **Limitation, stated explicitly**: this
means the In-the-Wild evaluation under-exercises genuine multi-window
aggregation behavior on longer (10–30s) clips, which is exactly the
regime the Streamlit app's 30-second upload limit targets. This is a real
gap: this experiment does not tell us how aggregation behaves on longer
clips, only that it makes no difference on typically-short In-the-Wild
audio.

## 13. Robustness (Step 19)

40-clip fixed subset of the evaluation set (20 bonafide + 20 spoof),
degraded via `ffmpeg` (MP3) / `scipy` (telephone bandpass, Gaussian noise),
never committed. Full numbers: `results/metrics/spectra_aasist3_robustness.json`.

| Variant | EER | ROC-AUC | Bonafide FPR (at frozen threshold) |
|---|---|---|---|
| Original | 0.0% | 1.000 | 0.0% |
| MP3 128 kbps | 0.0% | 1.000 | 0.0% |
| MP3 64 kbps | 0.0% | 1.000 | 0.0% |
| Telephone 300–3400 Hz | 0.0% | 1.000 | 0.0% |
| Gaussian noise, +20 dB SNR | 0.0% | 1.000 | 0.0% |
| Gaussian noise, +10 dB SNR | **7.5%** | 0.995 | **5.0%** |

**Finding**: Spectra-AASIST3 is essentially unaffected by MP3 compression
(even 64kbps), telephone-band filtering, and moderate noise (+20dB SNR) on
this 40-clip subset — only heavy noise (+10dB SNR, a genuinely harsh
condition) produces a measurable but still modest degradation (EER
0%→7.5%). This directly and positively supports the FYP's robustness
research question with real, project-measured evidence, on a small but
real subset (40 clips — a meaningful, not exhaustive, robustness check).

## 14. Streamlit viability

Two independent axes, both must be weighed honestly:

- **Detection quality**: dramatically better than the deployed model on
  the single most important real-world failure mode (bonafide false
  positives on out-of-domain audio) — Section 11's 3% vs. 68% is the
  central, decisive number from this entire investigation.
- **Deployment cost** (Section 6): ~1.28GB checkpoint, ~12.5s cold load,
  ~1.67GB peak process RSS, ~1.04s per 4.04s window (~7.4x slower than
  Sara), and a naive ~8.4s projected total latency for a 30-second
  application clip. This does **not** trip the hard 2.5GB stop, but is a
  real, unresolved UX/latency concern for an "interactive" Streamlit
  demo, and the ~1.67GB isolated figure leaves markedly less headroom
  than Sara's profile once Streamlit's own baseline is added.

Neither axis should be allowed to silently override the other. The quality
result is too strong, and too directly relevant to the project's original
motivating failure (Phase 4), to dismiss on resource grounds alone — but
deploying an 8-second-latency, ~1.7GB-RSS model onto Streamlit Community
Cloud without further optimization work (Section 15) would trade one real
problem (false positives) for another (an unresponsive or resource-starved
app). ONNX INT8 dynamic quantization (Step 10) was **not attempted in this
sub-phase** — it is the concrete, evidence-based next step if this
candidate is pursued toward actual deployment, not a decision made here.

## 15. Final decision gate (Step 22)

> **A, with two mandatory caveats — an academic/reproducibility caveat and
> a deployment-cost caveat.** Not a clean, unconditional "replace Sara
> today."

**Reasoning**:

- The detection-quality bar for Outcome A ("substantially improves false
  positives/OOD accuracy") is met with the strongest evidence produced in
  this entire multi-phase investigation: a **project-measured**, same-clip,
  identical-dataset comparison showing bonafide FPR falling from 68%
  (Sara) to 3% (Spectra) — not an author-reported number, not a different
  dataset, not a proxy metric.
- However, per Step 3 and Section 3, Spectra-AASIST3 is an **unpublished,
  non-peer-reviewed** model. This project's own CLAUDE.md and this phase's
  explicit instructions require that this not be glossed over: it should
  not be presented as "the" published state-of-the-art replacement without
  that caveat attached every time this result is cited.
- Per Section 6/14, the **resource/latency cost is real and unresolved**:
  ~7.4x slower per window and ~1.4x more resident memory than Sara, with a
  ~8.4s naive 30-second-clip latency that has not been validated as
  acceptable on Streamlit Community Cloud, and no optimization
  (quantization, graph optimization) has yet been attempted or measured.
- Outcome **D is explicitly not chosen** — the evidence overwhelmingly
  shows a material improvement, not a null result.
- Outcome **E is explicitly not chosen** — reproducibility was sufficient:
  the one flagged assumption (pre-emphasis edge-sample convention) had no
  visible negative effect on the (very strong) measured results.
- Outcome **F was considered** but not selected as the primary
  recommendation: relegating a model with this much better real-world
  accuracy to "secondary only" while keeping a model with a 68%
  same-dataset bonafide FPR as primary would not serve the project's
  actual deployed users. The academic caveat is real and must be
  communicated, but it does not, on its own, outweigh a 20x same-clip
  false-positive-rate improvement on the project's own central research
  question.

**Recommendation**: treat Spectra-AASIST3 as the primary candidate for
replacing `sara_wav2vec2`'s detection logic, contingent on: (1) always
citing its unpublished status alongside any quality claim in project
write-ups (Section 3), and (2) a follow-up resource-optimization pass
(ONNX INT8 dynamic quantization or graph optimization, Step 10 — not yet
attempted) and a real Streamlit Community Cloud memory/latency test before
any actual redeploy. **No redeploy, UI change, or `main` merge is made in
this phase** — this is a recommendation for the next phase's explicit
scope, not an action taken here.

## 16. Reproducibility

```
git checkout feat/spectra-aasist3-eval
.venv\Scripts\python.exe scripts\download_spectra_inthewild_subset.py
.venv\Scripts\python.exe scripts\run_spectra_evaluation.py
.venv\Scripts\python.exe -m pytest -m "not integration and not slow"
.venv\Scripts\python.exe -m pytest -m integration tests\test_candidate_e.py
```

No dataset audio, degraded audio, or model weights are committed
(`data/raw/*`, `models/cache/*`, `results/metrics/*` all gitignored).
