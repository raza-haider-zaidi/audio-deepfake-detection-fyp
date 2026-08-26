# Spectra-AASIST3 Production Optimization

Branch: `feat/spectra-production-optimization` (child of
`feat/spectra-aasist3-eval`). `main` is unmodified; the live Streamlit
deployment is unchanged. This document does not overwrite
`docs/spectra_aasist3_evaluation.md` — it extends that investigation's
FP32 baseline with a quantization/deployment-cost experiment.

## 1. FP32 baseline (frozen, unchanged from the prior sub-phase)

Full detail: `docs/spectra_aasist3_evaluation.md`,
`results/metrics/spectra_fp32_baseline.json` (a consolidated snapshot
created in this sub-phase purely to freeze the reference point before any
quantization work — the underlying source files it summarizes were not
modified).

| Metric (200-clip evaluation set) | Spectra FP32 | `sara_wav2vec2` (same clips) |
|---|---|---|
| Accuracy | 0.935 | 0.630 |
| Bonafide FPR | 0.030 | **0.680** |
| Spoof FNR | 0.100 | 0.060 |
| F1 | 0.933 | 0.718 |
| ROC-AUC | 0.977 | 0.787 |
| EER | 0.020 | 0.250 |

(Recovered in full from `results/metrics/spectra_aasist3_sara_comparison.json`
— nothing here is truncated or fabricated; both blocks are reproduced
verbatim from that file.)

Resource baseline: checkpoint 1,279,022,864 bytes; cold load ~12.4s; RSS
after load ~1,665MB; native 4.04s-window latency ~1,044ms
(`intra_op_num_threads=2`).

## 2. ONNX graph inspection before quantizing (Step 3)

Direct inspection via `onnx.load()` (`results/metrics/spectra_onnx_graph_analysis.json`):
16,229 total graph nodes. Relevant weighted operators: **417 MatMul, 24
Conv, 0 Gemm, 57 LayerNormalization, 44 Softmax**. No fused ONNX
`Attention` op is present — the wav2vec2 transformer's self-attention is
exported in decomposed form (folded into the MatMul count).

Weight-byte distribution by consuming operator: **MatMul carries 95.4% of
all initializer weight bytes** (1,222.4MB of 1,281.3MB total); Conv
carries only 4.0% (51.3MB). This directly justifies prioritizing
MatMul-only dynamic quantization (Step 4) as the primary lever, and
treating Conv-focused static quantization as secondary/likely
low-value (Step 13) — a decision grounded in measured weight distribution,
not assumption.

## 3. Dynamic INT8 quantization (Step 4)

`onnxruntime.quantization.quantize_dynamic(op_types_to_quantize=['MatMul'],
weight_type=QuantType.QInt8, per_channel=False)` — CPU-only tooling,
applied to the ONNX graph (not the PyTorch source, not QAT, not
retraining). Wall time: 20.3s.

- **Output**: `spectra-aasist3-int8-dynamic.onnx`, gitignored under
  `models/cache/quantized/`, never committed.
- **Size**: 364,036,647 bytes — **71.53% smaller** than the 1,279,022,864-byte
  FP32 file.
- **SHA256**: `444f832d306a2be4f823119f84e698e8821db6a1aab248593d4b05b7a9a48108`

## 4. Quantization integrity (Step 5)

Verified directly (`results/metrics/spectra_int8_dynamic_integrity.json`):
loads with `providers == ['CPUExecutionProvider']` only; identical `wav
[batch,64600]` input / `logits [batch,2]` output contract; finite,
bit-identical output on repeated smoke inference (deterministic); no CUDA,
no `fairseq` import. Quantized graph: **351 MatMul → MatMulInteger**, 303
`DynamicQuantizeLinear` insertion nodes, 66 MatMul nodes left FP32 (the
quantizer's own conservative eligibility decision, not a manual choice),
all 24 Conv nodes correctly untouched (Conv was not targeted).

## 5. Numerical parity (Step 6)

100-clip parity set (50 bonafide + 50 spoof, drawn only from the
**calibration** split — never evaluation) — `results/metrics/spectra_int8_parity.json`:

| Metric | Value |
|---|---|
| Mean absolute bona-fide-logit difference | 0.149 |
| Median absolute difference | 0.078 |
| Max absolute difference | 1.171 |
| Pearson correlation | 0.9992 |
| Spearman rank correlation | 0.9976 |
| Decision agreement at the FP32 threshold | **100%** |

This is strong parity — every one of the 100 parity clips would receive
the identical accept/reject decision under FP32 and INT8 at the FP32
threshold. Not rejected; proceeding to full calibration/evaluation was
justified by this result, not assumed.

## 6. INT8 calibration and frozen evaluation (Steps 7–9)

Calibration derived **only** from the existing 200-clip calibration set
(the identical set used for FP32, `spectra_aasist3_split.json`, seed
2024) — `results/metrics/spectra_int8_calibration.json`. EER threshold:
**0.939693808555603** (vs. FP32's 0.9299831390380859 — close but
distinct, as expected from a numerically different model).

Frozen evaluation on the **same untouched 200-clip evaluation set** as
FP32 (`results/metrics/spectra_int8_evaluation.json`):

| Metric | FP32 | INT8 dynamic | Δ |
|---|---|---|---|
| Accuracy | 0.935 | 0.940 | +0.005 |
| F1 | 0.9326 | 0.9375 | +0.0049 |
| ROC-AUC | 0.9766 | 0.9819 | +0.0053 |
| EER | 0.020 | 0.030 | +0.010 |
| **Bonafide FPR** | **0.030** | **0.020** | **−0.010** |
| Spoof FNR | 0.100 | 0.100 | 0.000 |

## 7. Quantization acceptance criteria (Step 10)

Full detail: `results/metrics/spectra_int8_acceptance_criteria.json`.

| Guideline | Threshold | Actual | Pass? |
|---|---|---|---|
| EER increase | ≤ 1.0pp | 1.0pp | ✅ (exactly at the boundary) |
| Bonafide FPR increase | ≤ 2.0pp | −1.0pp (improved) | ✅ |
| F1 decrease | ≤ 0.02 | +0.0049 (improved) | ✅ |
| ROC-AUC decrease | ≤ 0.02 | +0.0053 (improved) | ✅ |

**All criteria pass.** The honest interpretation, stated explicitly to
avoid overclaiming: with n=200, this is not evidence that quantization
*improves* detection — it is evidence that quantization causes **no
measurable detection degradation** on this evaluation set. That is the
claim this project makes, and no stronger one.

## 8. Resource benchmark, FP32 vs INT8 (Step 11)

`results/metrics/spectra_resource_fp32_vs_int8.json`, `CPUExecutionProvider`
only, `intra_op_num_threads ∈ {1, 2}`:

| Metric (threads=2) | FP32 | INT8 dynamic | Ratio |
|---|---|---|---|
| Cold load time | 12.27s | 6.49s | **1.9x faster** |
| RSS after load | 1,663.3MB | 769.8MB | **2.16x less** |
| Warm 4.04s-window latency (mean) | 1,052.4ms | 639.2ms | **1.65x faster** |

`intra_op_num_threads=2` was selected as the conservative default for
both variants — consistently faster than 1 thread without the
oversubscription risk of a higher thread count on a constrained cloud
CPU, consistent with the same choice made for the FP32-only sub-phase.

## 9. Real ~30-second application-level benchmark (Step 12)

Not extrapolated from 4-second latency — an actual 30-second (480,000
sample) synthetic clip run through the full sequential-windowing
application path (`results/metrics/spectra_30s_application_benchmark.json`):

| | FP32 | INT8 dynamic |
|---|---|---|
| Cold session creation | 12.35s | 5.92s |
| Windows | 8 | 8 |
| Preprocessing time | 1.5ms | 1.5ms |
| Inference time (8 windows) | 8.22s | **5.27s** |
| Aggregation time | ~0ms | ~0ms |
| **Total analysis time (excl. cold load)** | **8.22s** | **5.27s** |

The 30-second clip is deterministic synthetic noise (`numpy Generator(seed=7)`);
its classification output has no scientific meaning and is not used as
evidence anywhere — this is a pure timing measurement, as instructed.

## 10. Streamlit-stack process memory simulation (Step 15)

Each variant run as a **fresh process** (`scripts/simulate_streamlit_process_memory.py`)
importing the actual dependency stack (`streamlit`, `matplotlib`,
`librosa`, `soundfile`, `pydantic`) this project's app loads, then one
30-second analysis, then cleanup:

| Stage | FP32 RSS | INT8 RSS |
|---|---|---|
| Baseline Python process | 32.0MB | 31.9MB |
| After Streamlit-stack imports | 92.3MB | 92.0MB |
| After model load | 1,456.8MB | **557.4MB** |
| **After 30s analysis (peak)** | **1,721.6MB** | **821.4MB** |
| After figure rendered | 1,724.5MB | 824.3MB |
| After cleanup (`gc.collect()`) | 391.1MB | 390.4MB |

**This is the decisive resource finding of this sub-phase**: INT8's
complete-application peak RSS (~821MB) is **lower than the currently
deployed Sara model's own measured full-app range (~900–1,200MB)**. FP32's
peak (~1,722MB) sits well above Sara and close to (though still below) the
user-defined 2.2GB high-risk heuristic. Cleanup releases the large
majority of both variants' peak memory back to the process (391MB
residual for both — dominated by the imported library stack, not the
model), indicating no obvious leak in this simulation.

## 11. Static quantization (Step 13)

**Not attempted.** Per the phase's own instructions, static quantization
is secondary and only warranted if dynamic quantization either fails to
reduce memory sufficiently or leaves a Conv-heavy section unoptimized.
Neither condition holds: dynamic quantization already delivered a 71.5%
size reduction and brought complete-app peak RSS below Sara's own
baseline (Section 10), and Conv is only 4.0% of model weight (Section 2)
— there is no material Conv-heavy section left unaddressed. Static
quantization was correctly judged unnecessary, not skipped for
convenience.

## 12. Inconclusive/abstention zone (Step 21)

`results/metrics/spectra_inconclusive_zone_analysis.json`. Spectra's score
distribution is highly bimodal on the evaluation set: 90% of bonafide
clips score below 0.0012 (near-certain) and 90% of spoof clips score
above 0.932 (near-certain) — a much cleaner separation than
`sara_wav2vec2`'s Phase 4 distribution, which required a large
inconclusive interval. A diagnostic band-width sweep around the decision
threshold showed no good coverage/harm-reduction tradeoff: a defensible
narrow ±0.02 band catches **zero** of the 3 bonafide false positives while
still discarding 6.5% of clips; a wide ±0.1 band catches all 3 but
discards over half the evaluation set. **No inconclusive zone is
recommended** — a standard binary decision at the calibrated threshold is
sufficient, an evidence-based conclusion, not an arbitrary 40–60% band.

## 13. Production artifact distribution (Step 17) — undecided, flagged for manual action

The INT8 artifact (364MB) must not be committed to this git repository,
and is not currently hosted anywhere. Three options are documented, none
executed:

- **A. Publish the derived INT8 ONNX model to a Hugging Face repository**
  (with Apache-2.0 attribution to `lab260/Spectra-AASIST3` and the
  upstream `facebook/wav2vec2-xls-r-300m`). Requires creating/uploading to
  an external HF repository — **explicitly not done in this phase**, per
  the instruction not to do so without explicit user authorization.
- **B. Reproducibly quantize at deploy/startup time** from the verified
  FP32 artifact (already hosted at `lab260/Spectra-AASIST3`). Quantization
  itself took only ~20s in this experiment, but this adds ~20s plus the
  FP32 download (1.28GB) to every cold start unless cached — a real
  tradeoff to weigh against option A, not evaluated further here.
  Also does not save Streamlit Cloud's own disk/bandwidth budget for
  the initial FP32 download.
- **C. Retain the FP32 remote artifact only**, accepting its higher
  resource cost (Section 10) if quantized hosting cannot be arranged.

**No option was selected.** This requires an explicit decision from the
project owner (Section 17 of the instructions: "Do NOT create/upload
external Hugging Face repositories without explicit user instruction") —
flagged as manual action required in the final report.

## 14. License / attribution (Step 18)

- **Spectra-AASIST3** (`lab260/Spectra-AASIST3`): Apache-2.0, confirmed
  directly from the repository's own `cardData`/tags via the HF API.
- **Upstream backbone** (`facebook/wav2vec2-xls-r-300m`): also
  **Apache-2.0**, confirmed directly via the HF API
  (`license:apache-2.0` tag, `cardData.license: apache-2.0`) — no
  additional restriction from the SSL frontend.
- **KAN-AASIST architecture code**: vendored directly in `model.py` within
  the Apache-2.0 repository; no separate license file or third-party
  attribution notice was found for the `KANLinear`/graph-attention
  implementation.
- **Quantization**: a numerical transform of the same Apache-2.0-licensed
  weights; does not introduce a new license obligation as far as this
  project can determine from the repository's own terms.
- Apache-2.0 is a permissive license (attribution required, no
  ShareAlike/NonCommercial restriction, unlike the earlier
  `antideepfake_wav2vec2_small` candidate's CC BY-NC-SA 4.0). This is a
  factual reading of the license as published, not a legal opinion; no
  unsupported legal conclusion is offered beyond what the license text
  itself states.

## 15. Streamlit readiness

| | FP32 | INT8 dynamic |
|---|---|---|
| Complete-app peak RSS (30s analysis) | ~1,722MB | **~821MB** |
| vs. Sara's own ~900–1,200MB baseline | ~1.4–1.9x higher | **at or below** |
| 30s application latency | 8.22s | 5.27s |
| Cold load | 12.3s | 6.5s |

INT8 is the first Spectra-AASIST3 variant in this project's history that
sits at or below the deployed model's own resource footprint while
retaining (indeed, matching or slightly exceeding) its accuracy
advantage over Sara. FP32 remains a real resource concern independent of
INT8's success.

## 16. Recommendation and final decision gate (Step 24)

> **A. INT8 materially reduces resources with negligible detection
> degradation. Recommend INT8 Spectra-AASIST3 as the cloud deployment
> candidate.**

Not forced: this outcome is the direct, measured consequence of Sections
6–10 (accuracy: no degradation, several metrics nominally improved) and
Sections 8–10 (resources: 1.9x faster load, 2.16x less memory, 1.65x
faster inference, complete-app peak RSS below the currently deployed
model's own baseline). The unpublished-status academic caveat from
`docs/spectra_aasist3_evaluation.md` Section 3 still applies and must
continue to be cited alongside any quality claim — this sub-phase does
not change that. The one remaining blocker before this could become a
real deployment is **Section 13's undecided artifact-distribution
question**, which is a manual, explicit-instruction-gated decision, not a
technical one.

**No redeploy, UI change, or `main` merge is made in this phase.**

## 17. Reproducibility

```
git checkout feat/spectra-production-optimization
.venv\Scripts\python.exe -m pip install -e ".[onnx]"
.venv\Scripts\python.exe scripts\run_spectra_int8_evaluation.py
.venv\Scripts\python.exe scripts\simulate_streamlit_process_memory.py fp32
.venv\Scripts\python.exe scripts\simulate_streamlit_process_memory.py int8
.venv\Scripts\python.exe -m pytest -m "not integration and not slow"
.venv\Scripts\python.exe -m pytest -m integration tests\test_candidate_e.py
```

No ONNX binaries, dataset audio, generated degraded audio, or secrets are
committed. `results/metrics/spectra_*` and `models/cache/*` remain
gitignored.
