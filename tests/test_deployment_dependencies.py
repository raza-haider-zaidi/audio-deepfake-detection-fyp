"""Cloud-candidate dependency-hygiene tests (docs/spectra_streamlit_candidate.md).

Verifies the EXPERIMENTAL branch's requirements.txt/packages.txt do not
carry heavyweight PyTorch/Transformers/fairseq/CUDA dependencies the
Spectra INT8 ONNX production path does not need, and that the production
import path genuinely does not pull them in as a side effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BANNED_PACKAGES = ("torch", "transformers", "torchaudio", "fairseq", "cuda", "nvidia")


def _requirements_lines() -> list[str]:
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def test_requirements_txt_excludes_heavyweight_research_dependencies():
    lines = _requirements_lines()
    lowered = "\n".join(lines).lower()
    for banned in BANNED_PACKAGES:
        assert banned not in lowered, f"requirements.txt must not reference '{banned}' on this deployment branch"


def test_requirements_txt_includes_onnxruntime_and_huggingface_hub():
    lines = _requirements_lines()
    lowered = "\n".join(lines).lower()
    assert "onnxruntime" in lowered
    assert "huggingface_hub" in lowered
    assert "streamlit" in lowered


def test_packages_txt_matches_the_known_deliberate_apt_dependency_set():
    """Guards against SILENT/ACCIDENTAL apt dependency creep -- any new
    entry must be a deliberate, reviewed addition to this exact allow-list,
    not deleted wholesale. `ffmpeg` is the original entry; the `canvas`
    build-toolchain packages were added deliberately for the vendored
    PO-token provider (see app/analysis/pot_provider.py,
    third_party/bgutil-ytdlp-pot-provider/, and docs/input_sources.md,
    "Proof-of-Origin token support") and are NOT a random dependency."""
    text = (REPO_ROOT / "packages.txt").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert lines == [
        "ffmpeg",
        "build-essential",
        "pkg-config",
        "libcairo2-dev",
        "libpango1.0-dev",
        "libjpeg-dev",
        "libgif-dev",
        "librsvg2-dev",
    ]


def test_production_import_path_does_not_add_banned_modules_to_sys_modules():
    """Import the same modules app/model_loader.py + streamlit_app.py's
    production path touches, and confirm no banned heavyweight module is
    pulled in as an import-time side effect -- regardless of whether it
    happens to be installed in the CURRENT dev environment (it is, for
    research adapters), the PRODUCTION import path itself must not import it."""
    banned_modules = ["torch", "transformers", "torchaudio", "fairseq"]
    already_present_before_test = {m for m in banned_modules if m in sys.modules}

    import app.model_loader  # noqa: F401
    from audio_deepfake_detector.models.registry import create_detector

    detector = create_detector("spectra_aasist3_onnx_int8", device="cpu")
    assert type(detector).__name__ == "CandidateSpectraAasist3OnnxDetector"

    newly_present = {m for m in banned_modules if m in sys.modules} - already_present_before_test
    assert not newly_present, f"Production import path unexpectedly imported: {newly_present}"
