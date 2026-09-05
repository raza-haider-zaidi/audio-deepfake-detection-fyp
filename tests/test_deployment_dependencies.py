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


def test_packages_txt_is_exactly_ffmpeg():
    """Streamlit Community Cloud's packages.txt installer does NOT strip
    `#` comments the way plain apt would -- it was observed (deployment
    failure after commit 53e7845) treating every word of a multi-line
    explanatory comment as a literal package name ("E: Unable to locate
    package Runtime", "... package `canvas`", etc.), which aborted the
    entire dependency-install step and broke the whole app, not just
    PO-token support. packages.txt must therefore contain ONLY real
    package names, nothing else -- see test_packages_txt_lines_look_like_
    valid_apt_package_names below for the general-purpose guard, and
    docs/input_sources.md / third_party/.../VENDORED.md for the
    explanations that used to live here as comments.

    Minimal by design: the vendored PO-token provider's native `canvas`
    dependency is installed lazily at runtime (pot_provider.py) and
    degrades gracefully (PO_TOKEN_PROVIDER_UNAVAILABLE) if that install
    fails -- so no build-toolchain packages are added here preemptively.
    They should only be added if a real Streamlit Cloud deployment log
    shows `canvas` actually failing to compile for lack of a specific
    library, at which point the exact missing package name from that log
    is the one to add -- not a speculative list."""
    text = (REPO_ROOT / "packages.txt").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert lines == ["ffmpeg"]


def test_packages_txt_lines_look_like_valid_apt_package_names():
    """General-purpose guard against the exact class of bug that broke
    deployment: every non-empty line must look like a real Debian package
    name (lowercase alphanumerics plus `+.-`) -- never a comment, markdown,
    backtick, sentence, or anything containing whitespace or `#`."""
    import re

    package_name_re = re.compile(r"^[a-z0-9][a-z0-9+.-]*$")
    text = (REPO_ROOT / "packages.txt").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    for line in lines:
        assert "#" not in line, f"packages.txt line looks like a comment: {line!r}"
        assert "`" not in line, f"packages.txt line contains a backtick: {line!r}"
        assert " " not in line.strip(), f"packages.txt line contains whitespace (not a single package name): {line!r}"
        assert package_name_re.match(line.strip()), f"packages.txt line is not a valid apt package name: {line!r}"


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
