"""CPU-only benchmark harness for configured deepfake-detector models.

Loads each enabled model one at a time (never simultaneously), measures RAM
and latency, then unloads before moving to the next model. Writes:

    results/metrics/cpu_model_benchmark.json

Does NOT compare raw predictions as an accuracy benchmark — synthetic smoke
audio has no scientific detection-quality meaning. Does NOT benchmark GPU.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_models.py
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_models.py --models sara_wav2vec2
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psutil  # noqa: E402

from audio_deepfake_detector.config.models_config import load_models_config  # noqa: E402
from audio_deepfake_detector.models.registry import create_detector  # noqa: E402
from audio_deepfake_detector.preprocessing.audio_loader import load_audio_file  # noqa: E402
from audio_deepfake_detector.utils.datatypes import BenchmarkStats, ModelBenchmarkResult  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "metrics" / "cpu_model_benchmark.json"
SMOKE_AUDIO_PATH = Path(__file__).resolve().parents[1] / "data" / "samples" / "smoke_multitone_4s.wav"
N_WARM_RUNS = 5


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def benchmark_model(model_id: str, device: str = "cpu") -> ModelBenchmarkResult:
    config = load_models_config()
    model_config = config.get(model_id)

    if not SMOKE_AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Smoke audio not found at {SMOKE_AUDIO_PATH}. "
            "Run scripts/generate_smoke_audio.py first."
        )
    audio_sample = load_audio_file(SMOKE_AUDIO_PATH)

    gc.collect()
    ram_before = _rss_mb()

    detector = create_detector(model_id, device=device, config=config)

    error: str | None = None
    load_succeeded = True
    cold_load_time_ms = 0.0
    first_inference_time_ms = 0.0
    warm_stats: BenchmarkStats | None = None
    ram_after = ram_before

    try:
        t0 = time.perf_counter()
        detector.load()
        cold_load_time_ms = (time.perf_counter() - t0) * 1000.0
        ram_after = _rss_mb()

        first_result = detector.predict(audio_sample)
        first_inference_time_ms = first_result.inference_time_ms

        warm_times = []
        for _ in range(N_WARM_RUNS):
            result = detector.predict(audio_sample)
            warm_times.append(result.inference_time_ms)

        warm_stats = BenchmarkStats(
            mean_ms=statistics.mean(warm_times),
            median_ms=statistics.median(warm_times),
            min_ms=min(warm_times),
            max_ms=max(warm_times),
            n_runs=len(warm_times),
        )
    except Exception as exc:  # record failure accurately, do not fabricate results
        load_succeeded = False
        error = f"{type(exc).__name__}: {exc}"
    finally:
        detector.unload()
        gc.collect()

    return ModelBenchmarkResult(
        model_id=model_id,
        repository=model_config.repository,
        revision=model_config.revision,
        device=device,
        checkpoint_size_bytes=model_config.checkpoint_size_bytes,
        ram_before_load_mb=ram_before,
        ram_after_load_mb=ram_after,
        load_memory_increase_mb=ram_after - ram_before,
        cold_load_time_ms=cold_load_time_ms,
        first_inference_time_ms=first_inference_time_ms,
        warm_inference=warm_stats,
        load_succeeded=load_succeeded,
        error=error,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only benchmark harness for deepfake detectors.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["caa_wav2vec2", "sara_wav2vec2"],
        help="Model ids to benchmark, one at a time.",
    )
    args = parser.parse_args()

    results = []
    for model_id in args.models:
        print(f"\n=== Benchmarking {model_id} (CPU only) ===")
        result = benchmark_model(model_id, device="cpu")
        results.append(result)
        if result.load_succeeded:
            print(f"  load OK: cold_load={result.cold_load_time_ms:.0f}ms "
                  f"ram_delta={result.load_memory_increase_mb:.1f}MB "
                  f"warm_mean={result.warm_inference.mean_ms:.1f}ms")
        else:
            print(f"  LOAD FAILED: {result.error}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": "cpu",
        "n_warm_runs": N_WARM_RUNS,
        "smoke_audio_note": "Benchmarks used deterministic synthetic smoke audio; "
                             "timings are engineering measurements only, not accuracy evidence.",
        "results": [asdict(r) for r in results],
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
