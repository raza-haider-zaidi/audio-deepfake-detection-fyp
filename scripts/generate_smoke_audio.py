"""Generate deterministic synthetic WAV files for pipeline smoke testing.

============================================================================
THESE ARE NOT SPEECH SAMPLES.
THE PREDICTIONS A MODEL PRODUCES ON THESE FILES HAVE NO SCIENTIFIC OR
DETECTION-QUALITY MEANING. Do not use their predicted labels to choose the
better detector.
============================================================================

Purpose ONLY:
  - decoder validation
  - resampling validation
  - tensor shape validation
  - model execution validation
  - latency benchmarking

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\generate_smoke_audio.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
SAMPLE_RATE = 16000


def _sine_wave(duration_seconds: float, frequency_hz: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def _multitone_wave(duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)
    signal = (
        0.15 * np.sin(2 * np.pi * 220.0 * t)
        + 0.15 * np.sin(2 * np.pi * 440.0 * t)
        + 0.10 * np.sin(2 * np.pi * 880.0 * t)
    )
    return signal.astype(np.float32)


def generate_all(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    specs = [
        ("smoke_sine_1s.wav", _sine_wave(1.0, 440.0)),
        ("smoke_sine_4s.wav", _sine_wave(4.0, 440.0)),
        ("smoke_sine_10s.wav", _sine_wave(10.0, 440.0)),
        ("smoke_multitone_4s.wav", _multitone_wave(4.0)),
        ("smoke_silence_2s.wav", np.zeros(int(2.0 * SAMPLE_RATE), dtype=np.float32)),
    ]

    for filename, waveform in specs:
        path = output_dir / filename
        sf.write(str(path), waveform, SAMPLE_RATE, subtype="PCM_16")
        files.append(path)

    return files


if __name__ == "__main__":
    print("Generating deterministic smoke-test audio (NOT speech; no scientific meaning).")
    written = generate_all()
    for f in written:
        print(f"  wrote {f}")
    print(f"\n{len(written)} file(s) written to {OUTPUT_DIR}")
