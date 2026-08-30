# Professional Analysis Platform Expansion

Branch: `feat/spectra-streamlit-candidate`. `main` is unmodified.
**The detector is frozen** — no change to the Spectra-AASIST3 model,
INT8 artifact, Hugging Face revision, SHA256, ONNX Runtime inference,
preprocessing, pre-emphasis, sample rate, model input length, calibrated
threshold, binary decision logic (`decide_label`), inconclusive
presentation rule (`presentation_state`), aggregation implementation,
quantization, evaluation datasets, evaluation thresholds, or any
scientific result. Verified via a full frozen-evaluation replay (Section
9) producing byte-identical metrics and identical spoof probabilities on
three previously-established known clips.

## 1. Information architecture

Streamlit's native multipage navigation (`pages/` directory) was chosen
over a custom router — it is the more robust option for this project
(no extra dependency, automatic sidebar navigation, each page is an
independently-testable script). Final structure:

- `streamlit_app.py` — **Analyze** (primary workflow, unchanged entry point)
- `pages/1_Evaluation.py` — **Evaluation** (project-measured metrics, FP32/INT8 optimization, model development)
- `pages/2_Robustness.py` — **Robustness** (opt-in degradation analysis)
- `pages/3_Methodology.py` — **Methodology** (pipeline, model-native vs. project-implemented)
- `pages/4_About.py` — **About** (model information, privacy, limitations, responsible use)

"Model Development" and "FP32 vs INT8 optimization" were folded into the
Evaluation page rather than given separate pages, since both are small,
data-driven sections that read from the same source (`app/data/evaluation_summary.json`)
— creating separate pages for them would have fragmented one coherent
research narrative into two mostly-empty pages.

## 2. Segment-evidence timeline

`app/analysis/evidence.py` + `plot_segment_timeline()` in `app/visualizations.py`.

Segment scores come from `CandidateSpectraAasist3OnnxDetector.predict_full_clip()`
— an **existing, unmodified** project-level extension (sequential,
non-overlapping ~4.0375-second windows) that has never been part of the
production single-window `predict()` decision. This phase calls it
**in addition to**, never instead of, `predict()`, only for clips longer
than one native segment. The timeline chart plots per-segment spoof
probability against time with the calibrated threshold as a reference
line; the segment table shows number, time range, both class
probabilities, and a per-segment "interpretation" that applies the same
frozen `presentation_state()` rule — explicitly labeled as **not an
independently validated per-segment threshold**, since segment-level
thresholding was never part of the calibration experiment.

**Finding worth recording**: production `predict()` always returns
`windows_analyzed=1` regardless of clip length (it only ever scores the
first native segment, per the documented author-compatible behavior) —
the previous phase's "Segment analysis" UI section was consequently dead
code in practice (`result.windows_analyzed <= 1` was always true). This
phase's Segment Evidence feature is a genuinely new capability, not a
fix to that prior code, and is clearly presented as supplementary,
non-decision-driving evidence.

## 3. Segment agreement

`compute_segment_agreement()` in `app/analysis/evidence.py`. Purely
descriptive: counts segments with `spoof_prob > 0.5` (a plain majority
split, deliberately **not** the calibrated threshold, since this measures
raw directional consistency across segments, not the deployment
decision). Documented rule for the `level` label:

```
High     agreement_percent >= 85%
Moderate agreement_percent >= 65%
Mixed    otherwise
```

Numeric values (agreement percent, score standard deviation, score range)
are always shown alongside the level label, per the instruction to
prefer numeric values over subjective language. `compute_segment_agreement`
returns `None` for single-segment clips agreement is not a meaningful
concept with one data point) and is verified (`test_evidence.py`) to
never touch or derive from `decide_label`/`presentation_state`.

## 4. Audio quality & analysis suitability

`app/analysis/audio_quality.py`. Every metric is a plain signal
calculation on the already-decoded waveform or a file-level fact — no new
ML model:

- Peak amplitude, RMS level (from the waveform)
- Silence ratio (fraction of samples below a fixed amplitude threshold)
- Clipping ratio (fraction of samples at/near full scale)
- Approximate bitrate (file size × 8 / duration)

**Deliberately not computed** (per project policy against inventing
unmeasurable claims): microphone quality, room type, speaker identity,
codec history, or prior-compression count. Speech-active proportion was
also not computed — no lightweight, already-available, reliable method
for it exists in this project's dependency set.

**Analysis suitability** (`assess_analysis_suitability()`) is a
documented, deterministic rule over silence/clipping ratios producing
Good / Limited / Poor, with the specific reasons stated in the UI. This
value is advisory-only: `test_suitability_never_touches_classifier_fields`
asserts the resulting dataclasses contain no field that could plausibly
be confused with a classifier score, and the UI copy explicitly states
"This assessment ... does not influence the detector's classification,
threshold, or presentation state."

## 5. Evidence summary ("Why this result?")

`evidence_summary_sentences()` in `app/analysis/evidence.py`. Only
factual, measurable sentences: overall spoof probability, the calibrated
threshold, whether the score crossed it, and (when segment data exists)
the segment-agreement count. No speculative claims (breathing, pitch,
prosody, "AI artifacts") are ever generated — the function has no code
path that could produce such text, and `test_evidence.py` asserts none of
those phrases appear.

## 6. Robustness analysis

`app/analysis/robustness.py`, `pages/2_Robustness.py`. **Reuses, rather
than reinvents**, the exact degradation methodology already used for this
project's frozen robustness experiment
(`results/metrics/spectra_aasist3_robustness.json`,
`scripts/run_spectra_evaluation.py`) — the MP3 round-trip, telephone-band
Butterworth filter, and additive-Gaussian-noise functions were verified
against that script before being copied here. Conditions: Original, MP3
128 kbps, MP3 64 kbps, telephone-band (300–3400 Hz), +20 dB SNR noise,
+10 dB SNR noise.

## 7. Robustness runtime/resource handling

- **Never runs automatically** — requires an explicit "Run robustness
  analysis" button click on a dedicated page.
- **Sequential**, one condition at a time (`run_robustness_analysis`'s
  `for condition in conditions` loop).
- **Reuses the cached detector** — calls `get_detector(DEPLOYMENT_MODEL_ID)`,
  the same `st.cache_resource`-decorated loader used everywhere else; no
  second model instance is created.
- **No permanent storage** — the MP3 round-trip uses
  `tempfile.TemporaryDirectory()`, which is deleted on exit; all other
  transforms operate purely in memory. Verified directly:
  `test_run_robustness_analysis_cleans_up_temp_files_and_covers_all_conditions`
  snapshots the OS temp directory's contents before and after a real run
  and asserts they are identical.
- **Documented added runtime**: each condition costs one additional
  inference pass (~0.6–1.9s depending on FP32/INT8 and hardware, per the
  Evaluation page's own latency numbers); a 6-condition run therefore adds
  roughly 4–12 seconds versus a standard analysis. The UI states this
  plainly before the button is clicked.
- MP3 conditions are skipped gracefully (not fabricated) if `ffmpeg` is
  not found in the deployment environment.

## 8. Research evaluation dashboard

`pages/1_Evaluation.py` reads **only** from `app/data/evaluation_summary.json`
— a small, git-committed file whose every value was copied verbatim from
this project's actual (gitignored) `results/metrics/*.json` outputs, with
the exact source file recorded per section and in a top-level
`_provenance` block. `app/evaluation_loader.py` returns `None`/omits a
section rather than fabricating a placeholder if data is missing.
`test_evaluation_loader.py::test_int8_evaluation_metrics_match_frozen_project_results`
pins the committed values against the known frozen numbers so any future
drift between the summary and the real experiment output would fail CI.

Displayed: EER (3.0%), ROC-AUC (0.9819), F1 (0.9375), bonafide FPR (2.0%),
spoof FNR (10.0%) for the INT8 model — all labeled **PROJECT-MEASURED
EVALUATION** with the exact dataset/subset methodology stated, never
implying universal performance.

**Confusion matrix** (Evaluation page): plotted directly from the real
INT8 confusion-matrix counts (TP 90, TN 98, FP 2, FN 10) — genuine data,
not derived from the headline metrics.

**Omitted, deliberately**: ROC curve and score-distribution charts. This
project's committed artifacts do not include per-clip raw INT8 scores
(only aggregate metrics were persisted for the INT8 evaluation run), so
per the explicit instruction "if raw values needed for a plot are
unavailable, do not create the plot," these charts were not built rather
than fabricated from the headline numbers.

## 9. Model development / FP32→INT8 optimization

Sourced entirely from `app/data/evaluation_summary.json`'s
`model_development_timeline` and `resource_comparison` sections (which
themselves cite `docs/inference_calibration.md`,
`docs/spectra_aasist3_evaluation.md`, and
`docs/spectra_production_optimization.md`). The Sara-vs-Spectra
comparison is explicitly labeled same-set (`same_evaluation_set: true`,
200 shared In-the-Wild clips) — the comparison shown is Sara vs. the
**FP32** Spectra model (the only same-set comparison this project's
artifacts contain); the page states plainly that there is no same-set
Sara-vs-INT8 comparison and does not invent one.

FP32 vs. INT8 detection performance and resource figures come from
`results/metrics/spectra_int8_acceptance_criteria.json` and
`spectra_resource_fp32_vs_int8.json`/`spectra_streamlit_process_simulation_{fp32,int8}.json`
respectively. The page explicitly states the detection-performance
differences are within small-sample variance, not a claim that
quantization improved accuracy.

## 10. Downloadable analysis report

`app/reporting/report.py`. **HTML report only**, plus a JSON technical
export — no PDF dependency was introduced, per the explicit preference
for a lightweight, reliable format when PDF would add dependency risk.
Both formats include: report ID, generation timestamp, filename, input
SHA-256, duration/sample rate/format/channels, presentation result, both
class probabilities, calibrated threshold, binary model decision, segment
analysis (count, table, agreement), audio quality/suitability, model
identification (name, architecture, runtime, artifact, revision,
threshold), inference time, and the disclaimer. Verified
(`test_reporting.py`) to never contain "verified authentic," "confirmed
deepfake," "forensically validated," or "model confidence."

### Report identifier

`AD-YYYYMMDD-XXXX`, where `XXXX` is derived from the first 4 hex
characters of the input file's SHA-256 — a **local, session-scoped**
reproducibility identifier. Neither the UI nor the report text implies
this belongs to a persistent database.

## 11. Methodology page

`pages/3_Methodology.py` — an 8-step pipeline diagram plus an explicit
table separating what is model-native (pre-emphasis, native input length)
from what this project implemented (decoding/validation, multi-segment
evidence, the calibrated threshold itself, the three-state presentation
logic, audio-quality diagnostics) and a note that the Evaluation page's
scientific metrics use the frozen binary decision, not the presentation
states.

## 12. Model information

`pages/4_About.py`, built from `detector.model_info()` +
`configs/models.yaml` — no hardcoded model-specific strings. States
plainly that the upstream model is pre-release/unpublished and has no
peer-reviewed paper.

## 13. Privacy & data handling

Verified against the actual implementation before writing any claim:

- No account/sign-in exists anywhere in this app — verified by inspection
  (no auth code exists).
- Audio is decoded from the in-memory upload buffer
  (`app/validation.py` → `load_audio_bytes`, operates on `io.BytesIO`) —
  verified in Phase 3's `docs/deployment.md`, unchanged since.
- Robustness-mode temporary files: verified via the actual temp-directory
  snapshot test in Section 7, not merely asserted.
- **Explicitly does NOT claim** "audio never leaves your device" — this
  app performs server-side inference (ONNX Runtime CPU on the hosting
  process), so audio necessarily reaches that server process. The
  About page states this accurately instead.
- No generative-AI API is called anywhere in the inference path (verified:
  the only network call in the analysis path is the one-time, cached
  Hugging Face model download).

## 14. Limitations / responsible use

`pages/4_About.py` — covers unseen synthesis methods, compression,
background noise, short clips, limited/non-speech content, overlapping
speakers, language/domain shift, recording equipment, adversarial
manipulation, and distribution shift. Responsible-use section states the
system must not be the sole basis for legal, forensic, disciplinary,
identity, or security decisions.

## 15. Recommended input

A small "Recommended input" expander near the upload widget
(`streamlit_app.py::_render_recommended_input`) — guidance only, with an
explicit statement that following it does not guarantee a particular
result.

## 16. Separation between supplementary analysis and classifier output

**Stated explicitly, and enforced by code structure**: `app/analysis/`
and `app/reporting/` never import from, call into, or modify
`audio_deepfake_detector`'s decision functions except by *calling*
`decide_label()`/`presentation_state()` (read-only, to describe a score
that was already computed) or `predict()`/`predict_full_clip()`
(read-only, to obtain scores). No function in this phase's new code sets,
overrides, or feeds back into the classifier's threshold, decision, or
presentation state. This is verified by `test_suitability_never_touches_classifier_fields`
and by the fact that the frozen-evaluation replay (Section 9 of the
report) reproduces byte-identical metrics after this entire phase.

## 17. Performance / cloud safety

| | Pre-model RSS (bare-mode import) |
|---|---|
| Before this phase | 95.9 MB |
| After this phase | 95.8 MB |

No material increase despite the substantial feature expansion — no
second neural network, no PyTorch/Transformers runtime dependency, no
heavy dashboard framework, and no large frontend library were introduced.
Robustness mode is explicitly allowed to take longer (documented in the
UI) since it is opt-in and sequential; normal single-clip analysis
performance is unchanged (the only addition is one extra
`predict_full_clip()` call, made only for clips longer than one native
segment, which is a bounded, already-existing operation).
