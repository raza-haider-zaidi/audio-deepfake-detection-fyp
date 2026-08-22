"""CLI prediction interface.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\predict_audio.py sample.wav --model sara_wav2vec2
    .\\.venv\\Scripts\\python.exe scripts\\predict_audio.py sample.wav --model caa_wav2vec2 --device cpu

--device accepts "cpu" or "auto" (auto always resolves to CPU in this project).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_deepfake_detector.inference.service import predict_file  # noqa: E402
from audio_deepfake_detector.models.registry import list_available_model_ids  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPU deepfake-detection inference on an audio file.")
    parser.add_argument("audio_path", help="Path to a WAV/MP3/FLAC file")
    parser.add_argument("--model", required=True, help=f"Model id. Available: {list_available_model_ids()}")
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto"], help="Inference device (CPU-only project)")
    args = parser.parse_args()

    result = predict_file(args.audio_path, model_id=args.model, device=args.device)

    print(f"file:              {args.audio_path}")
    print(f"duration:          {result.audio_duration_seconds:.2f}s")
    print(f"model:             {result.model_id}")
    print(f"repository:        {result.model_repository}")
    print(f"device:            {result.device}")
    print(f"raw label:         {result.raw_label}")
    print(f"normalized label:  {result.normalized_label}")
    print(f"confidence:        {result.confidence:.4f}")
    print(f"probabilities:     {result.probabilities}")
    print(f"windows analyzed:  {result.windows_analyzed}")
    print(f"inference time:    {result.inference_time_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
