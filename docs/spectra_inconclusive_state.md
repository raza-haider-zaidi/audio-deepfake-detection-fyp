# Spectra INT8 Inconclusive Presentation State

Branch: `feat/spectra-streamlit-candidate`. `main` is unmodified. No
model, artifact, HF revision, SHA256, calibrated threshold, evaluation
logic, calibration, or quantization was changed in this phase — this adds
a presentation-only third UI state and fixes the technical-details panel.

## 1. Presentation-state rule

```python
def presentation_state(spoof_prob: float, threshold: float) -> str:
    if spoof_prob >= threshold:
        return "SPOOF"
    if spoof_prob > 0.5:
        return "INCONCLUSIVE"
    return "BONAFIDE"
```

Verified to match the exact disagreement set already measured in
`docs/spectra_prediction_semantics_fix.md`: re-scoring all 400
calibration+evaluation clips end-to-end gives **196 BONAFIDE, 192 SPOOF,
12 INCONCLUSIVE** — 12/400 = **3.0%**, identical to the previously-measured
disagreement rate (this is the same zone, now given a first-class UI
state instead of an ad-hoc caption).

## 2. Binary evaluation decision — unchanged

`decide_label(spoof_prob, threshold)` (`"spoof" if spoof_prob >= threshold
else "bonafide"`) is untouched. `PredictionResult.binary_model_decision`
is a new field that mirrors `raw_label`/`normalized_label` exactly —
verified programmatically across all 400 clips: `decide_label(...) ==
"spoof"` if and only if `presentation_state(...) == "SPOOF"`, for every
clip. `presentation_state` is a strictly additive, UI-only concept; it is
never read by evaluation/calibration code.

## 3. Labels (Step 3)

| State | User-facing label |
|---|---|
| BONAFIDE | Likely Real / Bonafide |
| SPOOF | Likely AI-Generated / Spoofed |
| INCONCLUSIVE | Inconclusive / Mixed Evidence |

Explanation shown for INCONCLUSIVE: *"The model detected elevated spoof
indicators, but the score did not cross the calibrated spoof threshold."*
No "probably real" language is used anywhere for the disagreement zone.

## 4. Result score display (Step 4)

Always shown: Bonafide class probability, Spoof class probability. For
INCONCLUSIVE, additionally: **Calibrated spoof threshold** (e.g. 94.0%)
and **Observed spoof probability**. The word "confidence" is not used
anywhere in the INCONCLUSIVE display; for BONAFIDE/SPOOF the existing
class-specific "Bonafide/Spoof class probability" label (from the prior
phase's fix) is retained.

## 5. `PredictionResult` fields (Step 5)

`binary_model_decision: str | None` and `presentation_state: str | None`
were added to the shared `PredictionResult` dataclass
(`utils/datatypes.py`), both defaulting to `None` for adapters that don't
populate them (verified: `sara_wav2vec2` results still render exactly as
before via a fallback to `normalized_label` in `result_summary()` — see
`test_result_summary_falls_back_to_normalized_label_when_no_presentation_state`).
Only `candidate_e.py`'s `predict()` populates both fields, and
`binary_model_decision` is always set from the same `raw_label` used to
build `normalized_label` and `confidence` — never a separately-derived
value.

## 6/7. Technical-details panel fix — now model-aware (Steps 6/7)

`streamlit_app.py::_render_technical_details` previously hardcoded
Sara-specific text (`facebook/wav2vec2-base`, "Window | 4 seconds", "mean
spoof probability... across all windows") regardless of which model was
actually deployed. It now builds its table entirely from the active
detector's `model_info()` dict plus `model_config`, with no model-specific
string literals in the UI code itself.

New `model_info()` fields added to `candidate_e.py` (Spectra) and
`candidate_b.py` (Sara) so both render through the exact same generic
code path:

| Key | Spectra INT8 value | Sara value |
|---|---|---|
| `display_name` | `Spectra-AASIST3 INT8` | `Sara Wav2Vec2` |
| `architecture_short` | `XLS-R-300M + KAN-enhanced AASIST` | `Wav2Vec2 (facebook/wav2vec2-base backbone)` |
| `runtime` | `ONNX Runtime (CPU)` | `PyTorch (CPU)` |
| `native_window_description` | `64600 samples (~4.04s), deterministic first-window, tile-repeat if shorter` | `64000 samples (4.0s), 32000-sample hop (50% overlap sliding window)` |
| `preemphasis_coefficient` | `0.97` | *(not applicable, omitted)* |
| `threshold_description` | `0.939694 (spoof probability, not raw logit)` | *(not applicable, omitted)* |
| `aggregation_description` | *(see below — explicitly non-author-native)* | *(existing mean-of-windows description)* |

Spectra's `aggregation_description`, verbatim: *"Author-compatible single
deterministic window (first 64600 samples of the clip). Multi-window
aggregation across a full clip is available as a project-level
application extension (`predict_full_clip()`) but is NOT author-native
behavior and is not used by the current production analysis path."* This
directly satisfies the requirement to state clearly that full-clip
handling is a project extension, not author-native.

## 8/9. Tests added

`tests/test_candidate_e.py`:

- `presentation_state()` boundary tests: 0.20→BONAFIDE, 0.70→INCONCLUSIVE,
  0.922 (exact live-bug number)→INCONCLUSIVE, threshold−ε→INCONCLUSIVE,
  threshold→SPOOF, >threshold→SPOOF, exact 0.5 boundary→BONAFIDE
  (exclusive), all 5 previously-measured real disagreement examples→INCONCLUSIVE.
- `test_binary_model_decision_unaffected_by_presentation_state_in_disagreement_zone`:
  confirms `decide_label()` stays 2-valued and unchanged for the same
  inputs that trigger INCONCLUSIVE.
- Integration: `test_load_and_predict_populates_presentation_state_and_binary_decision`.
- `test_spectra_model_info_contains_expected_display_fields_and_not_sara_strings`
  and `test_sara_model_info_contains_display_fields_distinct_from_spectra`:
  confirm `"Sara1708"`/`"facebook/wav2vec2-base"` are absent from Spectra's
  `model_info()`, and `"Spectra-AASIST3"`/`"XLS-R-300M"` are absent from
  Sara's.
- `result_summary()` tests updated/added for the exact live-bug numbers
  (now INCONCLUSIVE, not a BONAFIDE+caption), the Sara-style fallback
  case, and both non-inconclusive directions.

## 10. Frozen evaluation regression (Step 10)

Replayed the 200-clip evaluation set through the unmodified decision path:

| Metric | Accepted | Replayed after this phase |
|---|---|---|
| Accuracy | 0.940 | 0.940 |
| F1 | 0.9375 | 0.9375 |
| ROC-AUC | 0.9819 | 0.9819 |
| EER | 0.030 | 0.030 |
| Bonafide FPR | 0.020 | 0.020 |
| Spoof FNR | 0.100 | 0.100 |

**Exact match, zero drift** — expected, since no scoring/decision code was
changed, only new presentation-layer fields were added.

## 11. Local verification (Step 11)

| Clip | True label | Prediction | State | Bonafide % | Spoof % |
|---|---|---|---|---|---|
| `ITW_84` | bonafide | Likely Real / Bonafide | BONAFIDE | 100.0% | 0.0% |
| `ITW_31` | spoof | Likely AI-Generated / Spoofed | SPOOF | 3.5% | 96.5% |
| `ITW_13340` | spoof | **Inconclusive / Mixed Evidence** | INCONCLUSIVE | 10.8% | 89.2% |

For `ITW_13340`: calibrated spoof threshold shown as 94.0%, observed spoof
probability 89.2%, explanation text rendered, and
`binary_model_decision="bonafide"` preserved for reproducibility even
though the presentation state is INCONCLUSIVE. All three cases are
internally consistent — no case shows a probability contradicting its
displayed prediction.

## What was NOT changed

Model artifact, HF repository/revision/SHA256, calibrated threshold value,
calibration/evaluation methodology, quantization, and the overall page
layout are all unchanged. This phase adds a presentation state and fixes
a display-metadata bug only.
