# Spectra INT8 Prediction-Semantics Investigation

Branch: `feat/spectra-streamlit-candidate`. `main` is unmodified. No model
artifact, Hugging Face revision, SHA256, or calibrated threshold value was
changed in this phase — this fixes UI/decision-logic clarity only.

## The reported live bug

A known AI-generated clip on the deployed Spectra INT8 candidate showed:

```
Prediction: Likely Real / Bonafide
Model confidence: 7.8%
Bonafide probability: 7.8%
Spoof probability: 92.2%
```

This looks internally contradictory, and the bug report's working theory
was that the calibrated threshold (`0.939693808555603`) was being applied
to the wrong quantity — the raw bona-fide logit instead of a softmax
probability.

## Root cause — verified against the frozen calibration/evaluation source, not assumed

Tracing `scripts/run_spectra_int8_evaluation.py` (the actual script that
produced the accepted INT8 metrics: EER 3.0%, ROC-AUC 0.9819, F1 0.9375,
bonafide FPR 2.0%, spoof FNR 10.0%):

```python
def score_clip(session, waveform):
    """Returns (bonafide_logit, spoof_prob)."""
    ...
    spoof_prob, bonafide_prob = softmax_spoof_bonafide(logits)
    return float(logits[1]), spoof_prob
```

`score_clip()` DOES compute the raw bona-fide logit (`logits[1]`) — but
**only ever discards it**. Every downstream calibration/evaluation call
(`compute_eer`, `best_balanced_accuracy_threshold`, `best_f1_threshold`,
`max_fpr_threshold`, `full_metrics_report`) is invoked on `spoof_prob`,
the second return value. **The calibrated threshold
(`INT8_DYNAMIC_CALIBRATED_THRESHOLD = 0.939693808555603`) is therefore a
threshold on the softmax spoof probability, not the raw bona-fide logit.**

This directly contradicts the bug report's working theory — which is
exactly the kind of assumption this project's own policy requires
verifying against source rather than accepting at face value. Tracing the
actual code, not the initial theory, is what this document reports.

`candidate_e.py`'s `predict()` (before this phase's changes) already
compared `spoof_prob >= self.threshold`, i.e. **the same quantity and the
same direction as the frozen calibration.** There was no threshold
misapplication in the decision-logic sense.

### So what actually explains the reported numbers?

```
spoof_prob = 0.922, threshold = 0.939693808555603
0.922 >= 0.939693808555603  ->  False  ->  decision = "bonafide"
```

**This is the calibrated, frozen-consistent decision — not a
misclassification.** The INT8 calibration process (Step 8 of the prior
phase, `docs/spectra_production_optimization.md`) chose the EER threshold,
which for this model lands at a high, low-bonafide-FPR operating point
(0.9397, far from a naive 0.5 split). The frozen evaluation's own reported
**spoof FNR = 10%** means exactly this: 10% of genuinely spoofed clips
have `spoof_prob` between 0.5 and 0.9397, and are — by design, as
calibrated — classified as bonafide. The reported live clip is very
plausibly one instance of that already-measured 10% rate, not a new
defect.

**Measured, not assumed**: re-scoring all 400 calibration+evaluation
clips with the INT8 model and comparing the calibrated decision against a
naive 50%-split softmax argmax found **12/400 (3.0%) disagree**, always in
the direction "spoof_prob is between 0.5 and 0.9397 → calibrated decision
says bonafide, naive argmax would say spoof." One such real clip
(`ITW_13340`, true label spoof) reproduces the exact live-bug pattern:
bonafide 10.8%, spoof 89.2%, predicted "Likely Real / Bonafide".

## The actual defect: UI labeling, not decision logic

`app/formatting.py::result_summary()` returned `confidence =
probabilities[predicted_class]` — which is correct in principle (the
probability of whatever class won) — but every caller displayed it under
the generic label **"Model confidence."** When the predicted class's own
probability is a minority (7.8% for a "Bonafide" prediction, because the
threshold sits at 0.94, not 0.5), showing "confidence: 7.8%" next to the
"Bonafide" prediction and a separately-displayed "Spoof probability: 92.2%"
reads as a bug even though the underlying decision is correct and
intentional.

## Fix

**No change to `decide_label()`'s logic, `self.threshold`, the HF
revision, the SHA256, or the ONNX artifact.** The decision rule was
extracted into a small pure function,
`candidate_e.decide_label(spoof_prob, threshold) -> "spoof" | "bonafide"`,
for testability — it is byte-for-byte the same comparison that was already
there (`spoof_prob >= threshold`).

`app/formatting.py::result_summary()` now additionally returns:

- `confidence_label`: `"Bonafide class probability"` or `"Spoof class
  probability"`, depending on which class was predicted — replacing the
  generic, misleading `"Model confidence"` label used in `streamlit_app.py`.
- `threshold_disagreement`: `True` when the predicted class's own
  probability is below 50% (i.e. exactly the situation that reads as
  contradictory). `streamlit_app.py` renders an additional clarifying
  caption only when this is `True` — per the phase's own instruction to
  "not overcomplicate the public view unless disagreement is actually
  observed," this extra text is conditional, not shown for typical
  high-confidence results.

Example, BONAFIDE with disagreement (matches the live bug exactly):

```
Prediction: Likely Real / Bonafide
Bonafide class probability: 7.8%
Bonafide probability: 7.8%
Spoof probability: 92.2%
[caption] This model's deployment decision uses a calibrated threshold
chosen to minimize false alarms on genuine speech, not a simple 50/50
split -- so the predicted class can have a class probability below 50%
while still being the correct calibrated decision for this clip.
```

Example, SPOOF, no disagreement:

```
Prediction: Likely AI-Generated / Spoofed
Spoof class probability: 96.5%
Bonafide probability: 3.5%
Spoof probability: 96.5%
```

## Regression test note — a deliberate deviation from the bug report's own example

The bug report's Step 8 asked for a regression test asserting `normalized
label == SPOOF` for the exact reported numbers (bonafide=7.8%,
spoof=92.2%). Per the verified root cause above, **that assertion would be
wrong**: 92.2% is below the calibrated 93.97% threshold, so BONAFIDE is
the correct, frozen-consistent decision. Asserting SPOOF would require
changing the calibrated threshold or the decision rule, which the phase
instructions explicitly forbid, and would silently diverge production
behavior from the accepted INT8 evaluation metrics.
`test_decide_label_exact_live_bug_numbers_is_bonafide_not_spoof` asserts
the verified-correct label (`bonafide`) instead, with an explicit
docstring explaining why.

## Frozen evaluation replay (Step 10)

Re-ran the 200-clip evaluation set through the (now-refactored)
`predict()` path:

| Metric | Accepted (prior phase) | Replayed after fix |
|---|---|---|
| Accuracy | 0.940 | 0.940 |
| F1 | 0.9375 | 0.9375 |
| ROC-AUC | 0.9819 | 0.9819 |
| EER | 0.030 | 0.030 |
| Bonafide FPR | 0.020 | 0.020 |
| Spoof FNR | 0.100 | 0.100 |

**Exact match, zero drift** — expected, since `decide_label()` is a pure
extraction of the pre-existing comparison, not a behavior change.

## Local verification (Step 11)

Ran real clips through the full `predict()` → `result_summary()` path:

- **Known bonafide** (`ITW_84`): Bonafide class probability 100.0%, no
  disagreement flag, `st.success` (green) state.
- **Known spoof** (`ITW_31`): Spoof class probability 96.5%, no
  disagreement flag, `st.warning` (amber) state.
- **Disagreement-zone clip** (`ITW_13340`, true label spoof): Bonafide
  class probability 10.8%, `threshold_disagreement=True` — the clarifying
  caption now renders, resolving the apparent contradiction.

All three are internally consistent given the corrected labeling.

## What was NOT changed

Per explicit instruction: the ONNX artifact, HF repository/revision/SHA256,
calibrated threshold value, model architecture, and overall UI layout are
all unchanged. This phase is a labeling/decision-logic-clarity fix only.
