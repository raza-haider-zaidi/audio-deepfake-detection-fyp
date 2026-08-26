"""Deployment-like local resource simulation (Step 15,
docs/spectra_production_optimization.md): import the Streamlit-relevant
stack this project's app actually depends on, then load a Spectra ONNX
variant and run one 30-second analysis, tracking COMPLETE PROCESS RSS at
each stage. Run once per variant (fp32 / int8) as a fresh process --
importing both variants in one process would contaminate the measurement.

Usage:
    .venv\\Scripts\\python.exe scripts\\simulate_streamlit_process_memory.py fp32
    .venv\\Scripts\\python.exe scripts\\simulate_streamlit_process_memory.py int8
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
FP32_PATH = REPO_ROOT / "models" / "cache" / "models--lab260--Spectra-AASIST3" / "snapshots" / "bc0ded888080ddad493177bb53aa6f5b95219d7c" / "spectra-aasist3.onnx"
INT8_PATH = REPO_ROOT / "models" / "cache" / "quantized" / "spectra-aasist3-int8-dynamic.onnx"


def rss_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / 1e6


def main(variant: str) -> None:
    proc = psutil.Process()
    stages = {}

    stages["baseline_python_process"] = rss_mb(proc)

    # Import the same stack the deployed Streamlit app actually loads.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
    import soundfile as sf  # noqa: F401
    import librosa  # noqa: F401
    import pydantic  # noqa: F401
    import streamlit  # noqa: F401

    stages["after_streamlit_stack_imports"] = rss_mb(proc)

    import onnxruntime as ort

    from audio_deepfake_detector.models.candidate_e import make_sequential_windows, softmax_spoof_bonafide

    path = FP32_PATH if variant == "fp32" else INT8_PATH
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    stages["after_model_load"] = rss_mb(proc)

    rng = np.random.default_rng(7)
    thirty_s_audio = (rng.standard_normal(30 * 16000) * 0.02).astype(np.float32)
    windows = make_sequential_windows(thirty_s_audio)

    t0 = time.time()
    spoof_probs = []
    for w in windows:
        (logits,) = sess.run(["logits"], {"wav": w[None, :].astype(np.float32)})
        sp, _ = softmax_spoof_bonafide(logits[0])
        spoof_probs.append(sp)
    analysis_s = time.time() - t0
    stages["after_30s_analysis"] = rss_mb(proc)

    # Simulate a Streamlit run rendering a results figure (matches app.py's use of matplotlib).
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(spoof_probs)), spoof_probs)
    fig.canvas.draw()
    stages["after_figure_rendered"] = rss_mb(proc)
    plt.close(fig)
    del sess, windows, thirty_s_audio, spoof_probs
    gc.collect()
    stages["after_cleanup"] = rss_mb(proc)

    result = {
        "variant": variant,
        "n_windows": len(make_sequential_windows(np.zeros(30 * 16000, dtype=np.float32))),
        "analysis_wall_time_s": round(analysis_s, 2),
        "rss_stages_mb": {k: round(v, 1) for k, v in stages.items()},
        "note": "Complete process RSS, including the full Streamlit/matplotlib/librosa/soundfile/pydantic import stack this project's app.py actually loads -- not an isolated onnxruntime-only measurement.",
    }
    out_path = REPO_ROOT / "results" / "metrics" / f"spectra_streamlit_process_simulation_{variant}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
