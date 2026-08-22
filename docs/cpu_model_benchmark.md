# CPU Model Benchmark Report

Measured 2026-08-23. Full machine-readable results:
[`results/metrics/cpu_model_benchmark.json`](../results/metrics/cpu_model_benchmark.json).
Model metadata (no weights): [`results/model_manifest.json`](../results/model_manifest.json).

Environment: Python 3.12.10, PyTorch 2.13.0+cpu, Transformers 5.15.1,
`torch.cuda.is_available() == False` (CPU-only wheel; no CUDA anywhere in
this measurement). Development machine has an RTX 5060 Ti but it was not
used — the GPU was not exercised for any part of this benchmark.

## A. Author-reported detection metrics (NOT reproduced by this project)

See [`docs/model_candidate_analysis.md`](model_candidate_analysis.md) for
the full, cited breakdown. Summary:

| | Candidate A (`caa_wav2vec2`) | Candidate B (`sara_wav2vec2`) |
|---|---|---|
| Eval EER (ASVspoof 2019 LA) | 1.55% (author-reported) | 5.55% (author-reported) |
| Cross-dataset evidence | "In-the-Wild" EER 27.68% (author-reported, single number) | ASVspoof 2021 LA EER 9.09%, WaveFake mean 29.4% (author-reported, two named external evaluations) |

These numbers come from the model authors' published documentation. **This
project has not run either model against ASVspoof, WaveFake, In-the-Wild,
or any other evaluation dataset.** They are included here for context only,
not as results this project produced.

## B. Our measured CPU engineering benchmarks

Measured via `scripts/benchmark_models.py`: each model loaded, benchmarked,
and unloaded sequentially (never both resident in memory at once). Inference
timing used the deterministic synthetic 4-second multitone smoke file
(`data/samples/smoke_multitone_4s.wav` — **not speech; no scientific
detection-quality meaning**), 1 cold + 1 first-warm + 5 additional warm runs.

| Metric | Candidate A (`caa_wav2vec2`) | Candidate B (`sara_wav2vec2`) |
|---|---|---|
| Checkpoint size | 606,800,672 bytes (~579 MiB) | 491,044,441 bytes (~468 MiB) |
| RAM before load | 34.3 MB | 397.3 MB† |
| RAM after load | 1290.4 MB | 1199.1 MB |
| Load memory increase | 1256.1 MB | 801.9 MB† |
| Cold load time | 1860.3 ms | 1793.4 ms |
| First inference time | 173.4 ms | 140.6 ms |
| Warm inference mean | 142.7 ms | 140.2 ms |
| Warm inference median | 136.9 ms | 136.4 ms |
| Warm inference min | 136.1 ms | 135.0 ms |
| Warm inference max | 166.5 ms | 157.5 ms |
| Load result | Succeeded (see note below) | Succeeded, fully verified |

† Candidate B was benchmarked second in the same process, after Candidate A
was unloaded (`gc.collect()`). Python/PyTorch does not always return freed
memory to the OS on Windows, so the "RAM before load" and "load memory
increase" figures for the second-benchmarked model are inflated by
retained allocator memory from the first model — not a true from-empty-process
baseline. The **RAM after load** figures (both ~1.2–1.3 GB) are the more
reliable absolute figures. A from-empty-process comparison would require
running each model in a separate process; both processes involved loading
a ~360 MB `facebook/wav2vec2-base` backbone plus their own head weights.

**Both models loaded and completed CPU inference successfully.** For
Candidate A, load succeeded only after explicitly allow-listing one
specific, documented-as-expected missing key (`encoder.masked_spec_embed`);
see `docs/model_candidate_analysis.md` for the full verification trail and
the caveat that its pooling operation is an unverified assumption.

## Engineering observations (not scientific findings)

- Both checkpoints are large (~470–580 MB) and load into ~1.2–1.3 GB of
  process RAM once the `facebook/wav2vec2-base` backbone is included. This
  is a meaningful consideration for a Streamlit Cloud free-tier deployment
  (typically ~1 GB RAM limit), though **local measurements are not a
  guarantee of Streamlit Cloud's actual behavior** — cloud CPU/RAM
  characteristics may differ.
- Warm CPU inference latency for a single 4-second clip was ~135–170 ms for
  both models on this development machine — comfortably within the "a few
  seconds" soft target from the project brief, for a single window. Longer
  clips requiring multiple windows will scale roughly linearly with window
  count (see the windowing design in `docs/model_candidate_analysis.md` and
  `preprocessing/windowing.py`).
- Cold load time (~1.8–1.9 s) is dominated by downloading/initializing the
  `facebook/wav2vec2-base` backbone and deserializing a large checkpoint;
  this cost is paid once per Streamlit session if the model object is
  cached appropriately in the eventual frontend (Phase 3 concern, not
  implemented yet).
- These are single-machine engineering measurements, not statistically
  robust benchmarks (5 warm runs only) and not run against real speech —
  they validate that the pipeline works and give a rough latency/memory
  order of magnitude, nothing more.
