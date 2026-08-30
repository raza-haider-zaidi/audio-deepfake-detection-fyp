"""About — model information, privacy & data handling, limitations, and
responsible use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.components import render_section_title, setup_page  # noqa: E402
from app.model_loader import DEPLOYMENT_MODEL_ID, get_detector  # noqa: E402
from audio_deepfake_detector.config.models_config import load_models_config  # noqa: E402

setup_page("About")

st.markdown('<div class="adf-headline" style="font-size:1.7rem;">About</div>', unsafe_allow_html=True)

render_section_title("Model information")
config = load_models_config().get(DEPLOYMENT_MODEL_ID)
try:
    detector = get_detector(DEPLOYMENT_MODEL_ID)
    info = detector.model_info()
    st.markdown(
        f"""
| | |
|---|---|
| Model | {info.get('display_name', DEPLOYMENT_MODEL_ID)} |
| Deployment | Dynamic INT8 ONNX |
| Architecture | {info.get('architecture_short', config.architecture)} |
| Runtime | {info.get('runtime', 'ONNX Runtime CPU')} |
| Input | {info.get('sample_rate', 16000)} Hz audio |
| Native segment | {info.get('native_window_description', '')} |
| Pre-emphasis | {info.get('preemphasis_coefficient', '')} |
| Threshold | {info.get('threshold_description', '')} |
| Model artifact | `{config.repository}` |
| Revision | `{config.revision}` |
| License | {config.license} |
        """
    )
except Exception:  # noqa: BLE001 - metadata-only page, degrade gracefully
    st.markdown(
        f"""
| | |
|---|---|
| Model artifact | `{config.repository}` |
| Revision | `{config.revision}` |
| Architecture | {config.architecture} |
| License | {config.license} |
        """
    )
st.caption(
    "The upstream Spectra-AASIST3 model is currently pre-release/unpublished — it has no "
    "peer-reviewed paper. Project results were evaluated independently (see Evaluation)."
)

render_section_title("Privacy & data handling")
st.markdown(
    """
- No account or sign-in is required to use this application.
- Uploaded audio is processed in memory to produce the current analysis; it is not
  intentionally written to persistent storage by this application.
- If you run the optional Robustness Analysis, temporary degraded audio copies exist only
  in memory or in a short-lived temporary directory for the duration of that analysis, and
  are deleted immediately afterward.
- Inference is performed by this hosted application's own server-side process — audio
  **does** leave your device to reach that process (this is not a fully client-side/offline
  tool).
- No third-party generative-AI API receives your audio; inference uses a locally-loaded
  ONNX model.
"""
)

render_section_title("Limitations")
st.markdown(
    """
Detection performance can be affected by: synthesis methods not represented in the
evaluation data, audio compression and re-encoding, background noise, very short clips,
limited or non-speech audio content, multiple overlapping speakers, language or domain
shift relative to the evaluation dataset, recording equipment characteristics, deliberate
adversarial manipulation, and general distribution shift between the evaluation set and
real-world audio.
"""
)

render_section_title("Responsible use")
st.markdown(
    """
This system is a **research prototype**. Its output should not be used as the sole basis
for legal decisions, forensic conclusions, disciplinary actions, identity verification, or
security decisions.
"""
)
