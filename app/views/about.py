"""About — model information, privacy & data handling, limitations, and
responsible use.
"""

from __future__ import annotations

import streamlit as st

from app.branding import PROJECT_AUTHOR, display_model_artifact
from app.components import render_metadata_grid, render_section_title
from app.model_loader import DEPLOYMENT_MODEL_ID, get_detector
from audio_deepfake_detector.config.models_config import load_models_config


def render() -> None:
    render_section_title("About")

    st.markdown(
        f"""
**Audio Deepfake Detection Research Project**

Developed by
**{PROJECT_AUTHOR}**
"""
    )

    render_section_title("Model information")
    config = load_models_config().get(DEPLOYMENT_MODEL_ID)
    try:
        detector = get_detector(DEPLOYMENT_MODEL_ID)
        info = detector.model_info()
        render_metadata_grid(
            [
                (
                    "Model",
                    [
                        ("Model", info.get("display_name", DEPLOYMENT_MODEL_ID)),
                        ("Architecture", info.get("architecture_short", config.architecture)),
                        ("Model artifact", display_model_artifact(config.repository)),
                        ("License", config.license),
                    ],
                ),
                (
                    "Deployment",
                    [
                        ("Deployment", "Dynamic INT8 ONNX"),
                        ("Runtime", info.get("runtime", "ONNX Runtime CPU")),
                        ("Revision", config.revision),
                    ],
                ),
                (
                    "Input and decision",
                    [
                        ("Input", f'{info.get("sample_rate", 16000)} Hz audio'),
                        ("Native segment", info.get("native_window_description", "")),
                        ("Pre-emphasis", info.get("preemphasis_coefficient", "")),
                        ("Threshold", info.get("threshold_description", "")),
                    ],
                ),
            ]
        )
    except Exception:  # noqa: BLE001 - metadata-only page, degrade gracefully
        render_metadata_grid(
            [
                (
                    "Model",
                    [
                        ("Model artifact", display_model_artifact(config.repository)),
                        ("Revision", config.revision),
                        ("Architecture", config.architecture),
                        ("License", config.license),
                    ],
                )
            ]
        )
    st.caption(
        "The upstream Spectra-AASIST3 model is currently pre-release/unpublished — it has no "
        "peer-reviewed paper. Project results were evaluated independently (see Evaluation)."
    )

    render_section_title("Privacy & data handling")
    st.markdown(
        """
- No account or sign-in is required to use this application.
- This applies to every supported input mode — an uploaded audio file, a microphone
  recording, a voice note, a video's audio track, or a public online video link: the
  file, recording, or link you provide is processed by this hosted application's
  server-side process to produce the current analysis.
- For online video analysis, the public video is retrieved by a separate Local URL
  Ingestion Helper the presenter connects for their own session (see the Video URL
  source and its documentation); only the selected audio interval is then temporarily
  transferred to this hosted application for analysis. Only publicly accessible
  content is retrieved — no login, cookie import, or access-restriction bypass is
  performed. Nothing is permanently archived, and no history of submitted URLs is
  stored beyond the current session.
- Inference is performed by this hosted application's own server-side process — media
  **does** leave your device (and, for online video, the connected helper computer) to
  reach that process. This is not a fully client-side/offline tool, and no processing
  described here happens only on your device.
- Media is processed in memory to produce the current analysis; it is not intentionally
  retained after analysis by this application.
- Voice notes and video uploads that require decoding/transcoding (via ffmpeg) use
  short-lived temporary files for the duration of that single request only, deleted
  immediately afterward — never written under the application's own project directory.
  The connected helper applies the same rule to any media it retrieves.
- If you run the optional Robustness Analysis, temporary degraded audio copies exist only
  in memory or in a short-lived temporary directory for the duration of that analysis, and
  are deleted immediately afterward.
- Downloadable reports (PDF/HTML/JSON) are generated on demand, in memory, for the current
  analysis only — they are not stored by this application after being sent to your browser,
  and the original audio is never embedded inside a generated PDF report.
- No third-party generative-AI API receives your audio; inference uses a locally-loaded
  ONNX model. No visual/video-frame analysis is ever performed — video support extracts
  and analyzes only the audio track.
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
