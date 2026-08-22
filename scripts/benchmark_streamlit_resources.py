"""Development/testing-only process memory diagnostic for the Streamlit MVP.

Measures this project's COMPLETE process RSS at three points, simulating
the app's actual lazy-loading lifecycle:

    1. startup (nothing loaded)
    2. after the sara_wav2vec2 detector is loaded (as Streamlit would, via
       app.model_loader.get_detector's underlying logic)
    3. after one real CPU inference

This is a standalone script, not part of the public Streamlit UI. It does
not launch a Streamlit server; it exercises the same detector/service code
paths the app uses so results are representative.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_streamlit_resources.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import psutil  # noqa: E402

from audio_deepfake_detector.models.registry import create_detector  # noqa: E402
from audio_deepfake_detector.preprocessing.audio_loader import load_audio_file  # noqa: E402
from scripts.generate_smoke_audio import generate_all  # noqa: E402

MODEL_ID = "sara_wav2vec2"
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main() -> int:
    startup_rss = _rss_mb()
    print(f"RSS at startup:        {startup_rss:8.1f} MB")

    files = generate_all(output_dir=SAMPLES_DIR)
    smoke_audio_path = next(f for f in files if "multitone_4s" in f.name)
    audio_sample = load_audio_file(smoke_audio_path)

    detector = create_detector(MODEL_ID, device="cpu")
    detector.load()
    after_load_rss = _rss_mb()
    print(f"RSS after model load:  {after_load_rss:8.1f} MB  (+{after_load_rss - startup_rss:.1f} MB)")

    result = detector.predict(audio_sample)
    after_analysis_rss = _rss_mb()
    print(f"RSS after analysis:    {after_analysis_rss:8.1f} MB  (+{after_analysis_rss - after_load_rss:.1f} MB)")
    print(f"\n(inference took {result.inference_time_ms:.1f} ms on this run; "
          f"prediction on smoke audio has no scientific meaning)")

    detector.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
