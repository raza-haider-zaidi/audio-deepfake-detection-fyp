# Phase 4: Real-World False-Positive Diagnosis, Score Calibration, and Inference Robustness

Branch: `feat/inference-calibration` (not merged to `main`; the live
deployment at https://audio-deepfake-detection-fyp.streamlit.app/ is
unchanged by this phase).

## 1. Problem observed

The deployed application (model `sara_wav2vec2`, revision
`6c43629c953d6ff008501bf5f3eb983ac2321ad6`) misclassified a real, human
recording as spoof:

- duration: 28.35 s, windows analyzed: 14
- bonafide score: 28.6%, spoof score: 71.4%
- current production decision rule: `argmax` over `{bonafide, spoof}` mean
  probabilities across 4-second/50%-overlap windows, equivalent to a
  `spoof_score >= 0.5` threshold (see `models/candidate_b.py`,
  `preprocessing/windowing.aggregate_mean_probability`).

This is a single anecdotal observation. It motivated this phase but is
**not** treated below as evaluation evidence on its own — Steps 2 onward
build an actual labeled calibration set to investigate the mechanism.

## 2. Existing baseline (unchanged production behavior)

| | |
|---|---|
| Model | `sara_wav2vec2` (`Sara1708/deepfake-audio-wav2vec2`) |
| Revision | `6c43629c953d6ff008501bf5f3eb983ac2321ad6` |
| Sample rate | 16 kHz mono |
| Window | 4.0 s (64,000 samples), 50% overlap (32,000-sample hop) |
| Short-clip handling | zero-padded to one window |
| Aggregation | arithmetic mean of per-window softmax probabilities |
| Decision | `argmax(mean_probabilities)`, i.e. `spoof_score >= 0.5` |

**Author-reported metrics** (from the model's own documentation — see
`docs/model_candidate_analysis.md`; not reproduced by this project):

| Eval set | Author-reported EER |
|---|---|
| ASVspoof 2019 LA (eval) | 5.55% |
| ASVspoof 2021 LA (cross-dataset) | 9.09% |
| WaveFake (cross-dataset) | ~29.4% |

These author-reported numbers already document meaningful cross-domain
degradation. This phase measures our own numbers on our own calibration
set (Section 4+) — **the two are never conflated below.**

Nothing in this section was changed by this phase.

## 3. Score diagnostics (Step 3)

`src/audio_deepfake_detector/evaluation/diagnostics.py` adds (offline
only — production `predict()` in `models/candidate_b.py` is untouched):

- `compute_clip_diagnostics()`: per-clip mean / median / trimmed mean (10%
  each tail, falling back to the plain mean for ≤2 windows) / min / max /
  std of per-window spoof scores, plus window-level bonafide/spoof "votes"
  at a 0.5 per-window cutoff.
- `aggregate_spoof_score()`: computes a clip-level score under one of
  `mean | median | trimmed_mean | majority_vote`, for the aggregation
  comparison in Section 7.

`scripts/run_calibration.py` is the offline driver that records all of
this per calibration clip; it does not touch the Streamlit UI.

## 4. Voice activity detection (Step 4/5)

`src/audio_deepfake_detector/preprocessing/vad.py` wraps
`webrtcvad-wheels` (a maintained, wheel-only fork of Google's WebRTC VAD —
installs without a C compiler on both the Windows dev machine and
Streamlit Community Cloud's Debian runtime; verified Python 3.12
compatible). It operates on 16 kHz mono audio in 10/20/30 ms frames and
exposes `speech_ratio()` (fraction of a window's frames classified
speech-active) and `speech_active_regions()`. It is explicitly **not** a
deepfake detector and is never presented to users as one.

For every 4-second production window, `speech_ratio` was computed on the
exact same window boundaries the model used (imported
`WINDOW_SAMPLES`/`HOP_SAMPLES` from `models/candidate_b.py` rather than
re-deriving them, to avoid any drift between the diagnostic and the real
windowing). No windows were dropped by default — VAD filtering is only
evaluated experimentally in Section 8; it is off in production
(`configs/deployment_calibration.yaml: vad_enabled: false`).

## 5. Calibration dataset workflow (Step 6/7)

### 5.1 In-domain calibration set

- **Dataset:** `Bisher/ASVspoof_2019_LA` (public Hugging Face mirror of
  ASVspoof 2019 LA)
- **Split used:** `validation` — this corresponds to the official
  ASVspoof2019 LA **development** partition, not the training partition
  (the training partition is explicitly excluded per instructions, since
  it was used to train `sara_wav2vec2`)
- **Revision (dataset commit SHA):** `aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c`
- **Samples used:** 100 bonafide + 100 spoof (streamed; first 100 of each
  label encountered in the validation split, in split order — not a
  random sample, documented here as a limitation)
- **Labels:** `key` field, `0 = bonafide`, `1 = spoof` (per the dataset's
  own `dataset_info` class_label mapping)

### 5.2 Out-of-domain / real-world sanity set

- **Genuine speech:** `openslr/librispeech_asr`, split `test.clean`,
  revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`. 60 clips, capped at
  ≤20 s duration (bounds CPU inference time; documented, not hidden).
  LibriSpeech is read-aloud audiobook speech — a different domain, channel,
  and recording condition than ASVspoof's telephony-style corpus.
- **Synthetic/spoof speech:** `ajaykarthick/wavefake-audio`, split
  `train` (the dataset ships only one split), revision
  `f72691f2f6535f5b5bfeae7b09dba4639967722b`. 60 clips, one per distinct
  source utterance (deduplicated by `audio_id`) to avoid several
  near-identical vocoder renderings of the same sentence dominating a
  small set. WaveFake is LJSpeech-based synthetic speech from multiple
  neural vocoders — a different corpus and synthesis family than ASVspoof
  2019 LA's TTS/VC attacks.

**Total calibration set: 320 samples** (160 bonafide, 160 spoof; 200
in-domain, 120 out-of-domain). This is below the 100–300-per-class target
range on the out-of-domain side (60/class, not 100–300) — reduced
deliberately to keep this phase's CPU inference time and download volume
tractable; see Section 12 (Limitations).

Acquired via `scripts/download_calibration_data.py`
(`pip install -e ".[calibration]"` required — `datasets`, `pyarrow`,
`scikit-learn`, `webrtcvad-wheels` are research-only dependencies, **not**
part of `requirements.txt` / the Streamlit deployment). Output lands under
`data/raw/calibration/` with a `manifest.json` recording, per sample: file
path, label, source dataset, dataset revision, split, and domain. This
directory is gitignored (`data/raw/*`) — **no audio was committed.**

### 5.3 Data separation (Step 7)

The 320 samples above are the **calibration set**: every threshold,
aggregation choice, VAD setting, and probability-calibration parameter in
this document was chosen by looking at these samples. None of this data
may later be presented as an untouched final evaluation/test set — a
genuinely held-out evaluation set (disjoint samples, ideally disjoint
speakers/utterances from calibration) would be required before reporting
a final accuracy claim. No such held-out evaluation has been run in this
phase.

## 6. Baseline evaluation (Step 8)

Ran the **unmodified** production `sara_wav2vec2` adapter
(`create_detector` → `load()` → `predict()` per clip, same windowing/mean
aggregation as deployed) over all 320 calibration clips via
`scripts/run_calibration.py`. Raw per-clip records (label, all four
aggregation scores, full diagnostics, per-window VAD ratios, duration,
window count, source dataset) are written to
`results/metrics/calibration_baseline.json` (gitignored, regenerate with
the script — see Section 13).

Clip stats: duration 0.8–17.5 s (mean 4.6 s); windows per clip 1–8 (mean
1.9; 161/320 clips were single-window).

### Baseline metrics at the current production threshold (0.5, mean aggregation)

| Metric | Value |
|---|---|
| Accuracy | 0.791 |
| Precision | 0.897 |
| Recall (TPR) | 0.656 |
| F1 | 0.758 |
| Balanced accuracy | 0.791 |
| **False Positive Rate (bonafide → spoof)** | **0.075** |
| False Negative Rate | 0.344 |
| ROC-AUC | 0.931 |
| EER | 0.175 (17.5%), at threshold 0.0125 |

Confusion matrix and score-distribution histogram:
`results/figures/calibration_confusion_matrix.png`,
`results/figures/score_distributions.png`, `results/figures/calibration_roc.png`
(gitignored; regenerate via `scripts/run_calibration.py`).

### The false-positive rate split by domain is the key finding

| Bonafide subset | n | FPR @ 0.5 |
|---|---|---|
| In-domain (ASVspoof19 LA dev) | 100 | **0.0%** |
| **Out-of-domain (LibriSpeech)** | 60 | **20.0%** |

The model makes **zero** false-positive errors on bonafide speech drawn
from (a held-out split of) its own training distribution family, but
**1 in 5** genuine LibriSpeech clips is misclassified as spoof at the
current 0.5 threshold. This directly reproduces the mechanism behind the
original anecdotal false positive: it is not a fluke of one MP3 file, it
is a measurable, domain-dependent false-positive rate on this calibration
set.

Out-of-domain spoof detection is also weak: TPR (spoof correctly
identified) on the WaveFake out-of-domain spoof subset is only 8.3% at
threshold 0.5 (vs. 100% in-domain). The model is, on this evidence, much
closer to an "ASVspoof-2019-LA-family detector" than a general AI-speech
detector.

## 7. Threshold calibration (Step 9)

Swept 199 thresholds in (0, 1) using `evaluation/metrics.sweep_thresholds`
(no threshold assumed a priori):

| Candidate | Threshold | TPR | FPR | Precision | F1 | Balanced Acc. |
|---|---|---|---|---|---|---|
| Current production (0.5) | 0.500 | 0.656 | 0.075 | 0.897 | 0.758 | 0.791 |
| EER threshold | 0.0125 | 0.825 | 0.175 | 0.825 | 0.825 | 0.825 |
| Best balanced accuracy | 0.095 | 0.794 | 0.138 | 0.852 | 0.822 | **0.828** |
| Best F1 | 0.010 | 0.825 | 0.175 | 0.825 | 0.825 | 0.825 |
| FPR ≤ 5% (combined data) | 0.740 | 0.644 | **0.050** | 0.928 | 0.760 | 0.797 |

**But per-domain behavior at these same thresholds tells the real story**
(bonafide FPR / spoof TPR, in-domain vs. out-of-domain):

| Threshold | In-domain bonafide FPR | OOD bonafide FPR | In-domain spoof TPR | OOD spoof TPR |
|---|---|---|---|---|
| 0.095 (best balanced acc.) | 0.0% | 36.7% | 100% | 45.0% |
| 0.500 (current) | 0.0% | 20.0% | 100% | 8.3% |
| 0.740 (FPR≤5% combined) | 0.0% | 13.3% | 100% | 5.0% |

**No threshold in the swept range achieves ≤5% FPR on the out-of-domain
bonafide subset specifically** — the "FPR≤5%" candidate only achieves
that bound on the *combined* (200 in-domain + 120 OOD) calibration set,
which is dominated by the in-domain subset's perfect 0% FPR. On
out-of-domain data alone, even the most conservative threshold tested
still falsely flags 13.3% of genuine speech, and simultaneously pushes
out-of-domain spoof recall down to 5% (i.e. it also nearly stops detecting
real deepfakes from a different domain). Raising the threshold trades one
real-world failure mode for another; it does not resolve the underlying
problem.

**Trade-off analysis:** for this application, false-accusing genuine
speech is judged the more harmful failure mode (Phase 4 instructions).
Thresholding alone cannot deliver a low out-of-domain FPR without also
gutting out-of-domain spoof recall — this is evidence of a domain
generalization gap in the underlying representation/classifier, not
merely a badly chosen threshold.

## 8. Probability calibration (Step 10)

Raw softmax outputs are not necessarily calibrated probabilities. Fitted
temperature scaling (`evaluation/metrics.fit_temperature_scaling`, a
scalar `T` minimizing NLL on calibration data, grid+refine search) on the
mean-aggregated spoof score converted to logit space:

| | Before | After (T = 6.216) |
|---|---|---|
| Brier score | 0.1656 | **0.1188** |
| Expected Calibration Error (10 bins) | 0.1690 | **0.1040** |

Temperature scaling meaningfully improves both calibration metrics on
this calibration set — the raw softmax score over-states confidence. It
was **not** applied to production in this phase (Step 22: no behavioral
changes pushed to `main`); see `configs/deployment_calibration.yaml`. If
adopted later, the UI should describe the output as a "model score" /
"spoof probability score," not "certainty" (per the phase instructions),
even after calibration.

## 9. Aggregation experiment (Step 11)

| Strategy | EER | EER threshold | ROC-AUC | F1 @ 0.5 | Balanced Acc. @ 0.5 |
|---|---|---|---|---|---|
| **mean (current production)** | 0.175 | 0.0125 | **0.931** | 0.758 | 0.791 |
| median | 0.181 | 0.0007 | 0.930 | 0.758 | 0.791 |
| trimmed_mean (10%) | 0.175 | 0.0125 | 0.931 | 0.758 | 0.791 |
| majority_vote | 0.175 | 0.125 | 0.849 | 0.770 | 0.797 |

`mean` and `trimmed_mean` are identical here because 161/320 calibration
clips are single-window (10% trimming with ≤2 windows falls back to the
plain mean by construction — see `diagnostics.trimmed_mean`). `median` is
statistically indistinguishable from `mean` on this set. `majority_vote`
has a visibly *lower* ROC-AUC (0.849 vs. 0.931) — collapsing each window
to a hard 0.5-cutoff vote before aggregating discards score magnitude
information the continuous strategies retain.

**Conclusion: no aggregation change from the current `mean` strategy is
justified by this evidence.** No dominant handful of outlier windows was
found to disproportionately drive false positives on this data (windows
per clip were mostly 1–3; `n_windows_spoof_vote` correlated with — did not
diverge sharply from — the mean score for the false-positive cases
inspected in `calibration_baseline.json`).

## 10. VAD experiment (Step 12)

Windows below each candidate `min_speech_ratio` were excluded before mean
aggregation (falling back to using all windows if filtering would empty a
clip — documented, not a silent drop):

| min_speech_ratio | EER | ROC-AUC | FPR @ 0.5 | Balanced Acc. @ 0.5 |
|---|---|---|---|---|
| 0% (no filtering, current) | 0.175 | 0.931 | 0.075 | 0.791 |
| 25% | 0.175 | 0.931 | 0.075 | 0.791 |
| 50% | 0.178 | 0.937 | 0.075 | 0.791 |
| 75% | 0.188 | 0.936 | 0.063 | 0.797 |

VAD filtering produced no consistent improvement — EER is flat-to-slightly-worse
as the cutoff rises, and the ROC-AUC gain at 50-75% is marginal and not
accompanied by a matching EER improvement. **VAD filtering is not
recommended for production** (`configs/deployment_calibration.yaml:
vad_enabled: false`). This suggests the false-positive problem on
out-of-domain bonafide speech is not primarily driven by non-speech
regions (silence/music/transitions) diluting an otherwise-correct signal —
it looks more like the model scoring genuinely voiced, in-domain-looking
speech incorrectly, consistent with a representation/domain-shift issue
rather than a "junk window" issue.

## 11. Consistency / disagreement analysis (Step 13)

`ClipScoreDiagnostics.std_spoof` and `proportion_spoof_vote` are recorded
per clip in `calibration_baseline.json`. Because 161/320 calibration clips
are single-window, within-clip disagreement statistics are only
meaningful for the remaining 159 multi-window clips; a dedicated
multi-window disagreement study (e.g. "do highly split clips like 7
bonafide/7 spoof windows predict worse than unanimous 14/0 clips")
requires more multi-window clips than this calibration set's duration
distribution (mean 4.6 s) provides. This is recorded as an open question
for a follow-up phase (Section 12) rather than answered with weak
evidence from a handful of qualifying clips.

## 12. Inconclusive / mixed-evidence state (Step 14)

Derived — not guessed — from calibration data: for each side, the
tightest threshold at which ≥95% reliability holds (`build_inconclusive_zone`
in `scripts/run_calibration.py`, sweeping cumulative purity from each end
of the sorted score list):

- **Low threshold** (below this, ≥95% of calibration clips are truly
  bonafide): **≈0.0** (practically the score floor)
- **High threshold** (above this, ≥95% of calibration clips are truly
  spoof): **0.966**
- **Coverage** (fraction of calibration clips outside the inconclusive
  band): 65.6%
- **Accuracy on accepted (non-inconclusive) predictions:** 95.2%
- **Proportion marked inconclusive:** 34.4%

This band is evidence-derived, not the arbitrary 40–60%/45–55% ranges the
phase instructions explicitly warn against. Accuracy on *accepted*
predictions (95.2%) is high — but the band's coverage is very uneven
across domains:

| Subset | n | % marked inconclusive |
|---|---|---|
| In-domain bonafide | 100 | 12.0% |
| In-domain spoof | 100 | 0.0% |
| **Out-of-domain bonafide** | 60 | **75.0%** |
| **Out-of-domain spoof** | 60 | **88.3%** |

An inconclusive zone calibrated this way would abstain on roughly
three-quarters to seven-eighths of real-world (out-of-domain) clips in
this calibration set — it protects users from confident-but-wrong verdicts
on accepted predictions, but it does so by declining to give a confident
verdict at all on most real-world audio, not by making the model
correct more often. It is not a fix for the underlying generalization
gap — it is an honest way to stop presenting the model's real-world
guesses as if they were as reliable as its in-domain performance.

## 13. Second-model investigation (Step 16 — research note only, no integration)

`caa-speech-detection-asvspoof2019/lcnn-v7-cqt` (CQT-spectral-feature LCNN,
dramatically smaller than Wav2Vec2) was **not** downloaded, benchmarked, or
integrated in this phase — Step 16 explicitly scopes this to a documentation
note, not implementation. Given (a) VAD filtering showed no benefit
(Section 10, suggesting the problem is not "junk window" noise a
differently-featured model would simply avoid) and (b) both `sara_wav2vec2`
and the LCNN candidate share the same underlying training corpus family
(ASVspoof 2019), a small CQT-based model is not obviously complementary
enough to prioritize for ensembling without first establishing that its
error pattern actually diverges from Wav2Vec2's on genuinely out-of-domain
data — which would itself require the same kind of calibration-set work
performed in this phase for a second model. Recommended only as a
possible future investigation, contingent on evidence not yet collected.

## 14. Decision gate (Step 21)

**Outcome selected: C — the existing `sara_wav2vec2` model, deployed alone
with threshold/aggregation/VAD tuning, is not judged sufficiently reliable
for real-world deployment**, specifically because:

1. No threshold achieves an acceptable out-of-domain bonafide false-positive
   rate (best measured: 13.3%, still well above a 5% target) without
   collapsing out-of-domain spoof recall to near-zero (5% at that same
   threshold) — Section 7.
2. Neither aggregation strategy changes (Section 9) nor VAD-based window
   filtering (Section 10) measurably close this gap — the problem is not
   an artifact of window aggregation or non-speech dilution.
3. An evidence-derived inconclusive zone (Section 12) partially protects
   users from confidently-wrong verdicts, but does so by abstaining on
   75–88% of out-of-domain samples in this calibration set — a real
   mitigation for the *harm* of false accusation, not a resolution of the
   underlying domain generalization gap.
4. This is consistent with, and now measured evidence for, the author's own
   documented cross-dataset degradation (WaveFake EER ≈29% reported by the
   model authors; this project's own out-of-domain WaveFake-based subset
   shows a comparably weak spoof-detection signal).

This is not a rejection of `sara_wav2vec2` as a component — its in-domain
(ASVspoof-family) discrimination remains excellent (0% FPR, 100% TPR on
the in-domain calibration subset) — but it should not be presented to
users as a general-purpose "is this audio AI-generated" detector without
either (a) a visible inconclusive/low-confidence state for scores in the
uncertain band and copy that sets accurate expectations about real-world
(non-ASVspoof-like) audio, or (b) a second, genuinely complementary
detector and/or in-domain-adjacent recalibration in a future phase.
**Recommendation for the next phase:** adopt the inconclusive-zone UI
state (Step 22 defers the actual UI change) and revisit whether a second
model or targeted domain-adaptation is warranted, rather than shipping a
threshold change alone as if it solved the problem.

## 15. Limitations

- Calibration set is modest (320 samples; 60/class rather than the
  100–300/class target on the out-of-domain side) — chosen to keep this
  phase's CPU inference and download time tractable. Confidence intervals
  on the reported rates are correspondingly wide (e.g. a 20% OOD FPR on
  n=60 has a rough 95% CI of roughly ±10 percentage points via a normal
  approximation) — treat point estimates as indicative, not precise.
- In-domain samples were taken in dataset-streaming order, not randomly
  sampled from the full dev partition — a possible source of subtle bias
  if the source dataset is not itself shuffled.
- Out-of-domain genuine speech (LibriSpeech, read audiobooks) and
  out-of-domain spoof (WaveFake, LJSpeech-based vocoders) are not drawn
  from the same underlying corpus/speaker population as each other, unlike
  the in-domain ASVspoof pairing — some of the measured OOD gap could in
  principle reflect corpus-level acoustic differences (recording chain,
  noise floor) rather than "genuine vs. synthetic" differences per se.
  This is a real confound this phase could not fully rule out with a
  single small out-of-domain source pair.
- No disjoint, unseen final evaluation set exists yet — see Section 5.3.
  All numbers in this document are calibration-set numbers.
- The window-disagreement question (Step 13) is left open due to
  insufficient multi-window clips in this calibration set (Section 11).
- No probability calibration, threshold change, VAD, or aggregation change
  was applied to the live app in this phase (by design — Step 22/23).

## 16. Reproducibility

```
# 1. Install calibration-only dependencies (not part of requirements.txt)
.venv\Scripts\python.exe -m pip install -e ".[calibration]"

# 2. Download the calibration set (network access required; ~5 min)
.venv\Scripts\python.exe scripts\download_calibration_data.py `
    --in-domain-per-class 100 --ood-per-class 60

# 3. Run the full calibration pipeline against the UNMODIFIED production
#    sara_wav2vec2 adapter (~90s on this dev machine's CPU)
.venv\Scripts\python.exe scripts\run_calibration.py

# Outputs (gitignored, not committed):
#   results/metrics/calibration_baseline.json      (raw per-clip records)
#   results/metrics/calibration_thresholds.json
#   results/metrics/aggregation_comparison.json
#   results/metrics/vad_comparison.json
#   results/metrics/calibration_extra.json          (calibration + uncertainty)
#   results/figures/calibration_roc.png
#   results/figures/score_distributions.png
#   results/figures/calibration_confusion_matrix.png
```

Dataset revisions (exact commit SHAs, recorded automatically in
`data/raw/calibration/manifest.json` by the download script):

| Dataset | Split | Revision |
|---|---|---|
| `Bisher/ASVspoof_2019_LA` | validation (= ASVspoof2019 LA dev) | `aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c` |
| `openslr/librispeech_asr` | test.clean | `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` |
| `ajaykarthick/wavefake-audio` | train | `f72691f2f6535f5b5bfeae7b09dba4639967722b` |

No random seed is needed for data acquisition (deterministic streaming
order, first-N-per-label selection). `fit_temperature_scaling` uses a
deterministic grid+refine search (no RNG).
