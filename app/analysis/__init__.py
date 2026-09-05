"""Supplementary, descriptive analysis helpers that sit AROUND the frozen
Spectra-AASIST3 detector.

Nothing in this package feeds back into audio_deepfake_detector's
inference/decision logic. Every function here is display-only: audio
quality diagnostics, segment-agreement descriptive statistics, and
robustness-mode audio transforms are all computed independently of, and
never used to alter, the classifier's raw scores, calibrated threshold,
binary decision, or presentation state. See docs/analysis_platform.md.
"""
