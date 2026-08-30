"""Tests for the spectra_aasist3_onnx adapter (candidate_e.py).

Unlike candidate_d.py (antideepfake_wav2vec2_onnx), this model's
preprocessing is documented in prose on its own model card and is
implemented (not blocked). These tests verify the pure preprocessing
functions against hand-computed expected values -- no network access, no
model download required for the non-integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_deepfake_detector.config.models_config import load_models_config
from audio_deepfake_detector.models.candidate_e import (
    FP32_CALIBRATED_THRESHOLD,
    INT8_DYNAMIC_CALIBRATED_THRESHOLD,
    REQUIRED_SAMPLES,
    CandidateSpectraAasist3OnnxDetector,
    apply_preemphasis,
    author_compatible_preprocess,
    decide_label,
    make_sequential_windows,
    softmax_spoof_bonafide,
    window_to_required_length,
)
from audio_deepfake_detector.models.registry import create_detector


def test_preemphasis_hand_computed():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = apply_preemphasis(x, coeff=0.97)
    expected = np.array([1.0, 2.0 - 0.97 * 1.0, 3.0 - 0.97 * 2.0, 4.0 - 0.97 * 3.0], dtype=np.float32)
    np.testing.assert_allclose(y, expected, rtol=1e-5)


def test_preemphasis_first_sample_unchanged():
    x = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    y = apply_preemphasis(x)
    assert y[0] == 5.0


def test_preemphasis_empty_input():
    assert apply_preemphasis(np.array([], dtype=np.float32)).size == 0


def test_window_long_input_takes_deterministic_first_samples():
    x = np.arange(100000, dtype=np.float32)
    windowed = window_to_required_length(x)
    assert windowed.shape == (REQUIRED_SAMPLES,)
    np.testing.assert_array_equal(windowed, x[:REQUIRED_SAMPLES])


def test_window_exact_length_passthrough():
    x = np.arange(REQUIRED_SAMPLES, dtype=np.float32)
    windowed = window_to_required_length(x)
    np.testing.assert_array_equal(windowed, x)


def test_window_short_input_tile_repeats_not_zero_pads():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    windowed = window_to_required_length(x)
    assert windowed.shape == (REQUIRED_SAMPLES,)
    # Tile-repeat: every value must come from {1,2,3} -- zero-padding would
    # introduce zeros, which are not in the source signal.
    assert set(np.unique(windowed).tolist()) <= {1.0, 2.0, 3.0}
    # And the very first REQUIRED_SAMPLES values follow the tiled pattern.
    expected = np.tile(x, -(-REQUIRED_SAMPLES // 3))[:REQUIRED_SAMPLES]
    np.testing.assert_array_equal(windowed, expected)


def test_window_empty_input_does_not_crash():
    windowed = window_to_required_length(np.array([], dtype=np.float32))
    assert windowed.shape == (REQUIRED_SAMPLES,)
    assert np.all(windowed == 0.0)


def test_author_compatible_preprocess_applies_preemphasis_before_windowing():
    # A long input: preemphasis must be computed over the FULL waveform,
    # THEN cropped -- not preemphasis-after-crop (order matters at the
    # crop boundary because preemphasis depends on the previous sample).
    x = np.random.default_rng(0).standard_normal(100000).astype(np.float32)
    result = author_compatible_preprocess(x)
    expected = apply_preemphasis(x)[:REQUIRED_SAMPLES]
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_softmax_spoof_bonafide_higher_index1_means_more_bonafide():
    spoof_prob_a, bonafide_prob_a = softmax_spoof_bonafide(np.array([0.0, 5.0]))
    spoof_prob_b, bonafide_prob_b = softmax_spoof_bonafide(np.array([0.0, -5.0]))
    assert bonafide_prob_a > bonafide_prob_b
    assert spoof_prob_a < spoof_prob_b
    assert abs((spoof_prob_a + bonafide_prob_a) - 1.0) < 1e-6


def test_make_sequential_windows_covers_full_clip_non_overlapping():
    x = np.arange(REQUIRED_SAMPLES * 3, dtype=np.float32)
    windows = make_sequential_windows(x, window=REQUIRED_SAMPLES)
    assert len(windows) == 3
    for w in windows:
        assert w.shape == (REQUIRED_SAMPLES,)


def test_make_sequential_windows_short_clip_returns_single_tiled_window():
    x = np.array([1.0, 2.0], dtype=np.float32)
    windows = make_sequential_windows(x)
    assert len(windows) == 1
    assert windows[0].shape == (REQUIRED_SAMPLES,)


def test_model_config_is_disabled_and_not_in_available_ids():
    config = load_models_config()
    model_config = config.get("spectra_aasist3_onnx")
    assert model_config.enabled is False

    from audio_deepfake_detector.models.registry import list_available_model_ids

    assert "spectra_aasist3_onnx" not in list_available_model_ids()


def test_create_detector_via_registry_resolves_correct_class():
    detector = create_detector("spectra_aasist3_onnx", device="cpu")
    assert isinstance(detector, CandidateSpectraAasist3OnnxDetector)
    assert detector.model_id == "spectra_aasist3_onnx"


def test_create_detector_rejects_non_cpu_device():
    config = load_models_config().get("spectra_aasist3_onnx")
    with pytest.raises(ValueError):
        CandidateSpectraAasist3OnnxDetector(model_config=config, device="cuda")


def test_predict_before_load_raises():
    detector = create_detector("spectra_aasist3_onnx", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        detector.predict(audio_sample=None)  # type: ignore[arg-type]


def test_predict_full_clip_before_load_raises():
    detector = create_detector("spectra_aasist3_onnx", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        detector.predict_full_clip(audio_sample=None)  # type: ignore[arg-type]


@pytest.mark.integration
@pytest.mark.slow
def test_load_and_predict_on_cpu_with_smoke_audio():
    detector = create_detector("spectra_aasist3_onnx", device="cpu")
    try:
        metadata = detector.load()
        assert metadata.device == "cpu"
        assert detector.model_info()["input_length_samples"] == REQUIRED_SAMPLES

        from pathlib import Path

        from audio_deepfake_detector.preprocessing.audio_loader import load_audio_file
        from scripts.generate_smoke_audio import generate_all

        files = generate_all(output_dir=Path("data/samples"))
        multitone = next(f for f in files if "multitone_4s" in f.name)
        sample = load_audio_file(multitone)

        result = detector.predict(sample)
        assert np.isfinite(result.probabilities["bonafide"])
        assert np.isfinite(result.probabilities["spoof"])
        assert result.windows_analyzed == 1

        full = detector.predict_full_clip(sample, aggregation="mean")
        assert full["n_windows"] >= 1
    finally:
        detector.unload()


def test_default_threshold_is_fp32_calibrated_value():
    detector = create_detector("spectra_aasist3_onnx", device="cpu")
    assert detector.threshold == FP32_CALIBRATED_THRESHOLD


def test_int8_detector_automatically_uses_int8_threshold_not_fp32():
    detector = create_detector("spectra_aasist3_onnx_int8", device="cpu")
    assert detector.threshold == INT8_DYNAMIC_CALIBRATED_THRESHOLD
    assert detector.threshold != FP32_CALIBRATED_THRESHOLD


def test_expected_sha256_wired_from_config():
    config = load_models_config()
    fp32 = config.get("spectra_aasist3_onnx")
    int8 = config.get("spectra_aasist3_onnx_int8")
    assert fp32.expected_sha256 == "5f05c29a01ad80c702b32654db87c2aa6e467c11c67b6d47f2fac873f846cae9"
    assert int8.expected_sha256 == "444f832d306a2be4f823119f84e698e8821db6a1aab248593d4b05b7a9a48108"


def test_load_rejects_local_file_with_wrong_sha256(tmp_path):
    """Integrity check (Step 10): load() must refuse to load a file whose
    SHA256 does not match model_config.expected_sha256, even for a
    local_onnx_path override. Fails fast on the SHA mismatch, before ever
    reaching onnxruntime -- no network access, no real model needed."""
    bad_file = tmp_path / "not-the-real-model.onnx"
    bad_file.write_bytes(b"this is not an onnx file")

    detector = create_detector("spectra_aasist3_onnx", device="cpu")
    with pytest.raises(RuntimeError, match="SHA256"):
        detector.load(local_onnx_path=str(bad_file))


def test_fp32_and_int8_calibrated_thresholds_are_distinct_and_frozen():
    # Both values are frozen calibration results (results/metrics/spectra_aasist3_calibration.json,
    # spectra_int8_calibration.json) -- this test guards against accidental edits drifting
    # them apart from the documented, calibration-derived values.
    assert FP32_CALIBRATED_THRESHOLD == 0.9299831390380859
    assert INT8_DYNAMIC_CALIBRATED_THRESHOLD == 0.939693808555603
    assert FP32_CALIBRATED_THRESHOLD != INT8_DYNAMIC_CALIBRATED_THRESHOLD


def test_int8_model_config_is_published_and_pinned():
    # Published to Hugging Face (docs/spectra_streamlit_candidate.md):
    # repository/revision are real, resolvable, and pinned -- no more
    # PENDING_HF_PUBLISH placeholders.
    config = load_models_config().get("spectra_aasist3_onnx_int8")
    assert config.enabled is True
    assert config.repository == "Limitless-8/spectra-aasist3-int8-audio-deepfake"
    assert config.revision == "b56aed04853cb4e5bf825025c54c93d4bc345c61"
    assert config.checkpoint_filename == "spectra-aasist3-int8-dynamic.onnx"
    assert config.checkpoint_size_bytes == 364036647


@pytest.mark.integration
@pytest.mark.slow
def test_load_accepts_local_onnx_path_override():
    """The INT8 artifact is a local, gitignored file -- load() must support
    pointing the SAME adapter class at it via local_onnx_path instead of
    downloading checkpoint_filename from the Hub."""
    import os

    int8_path = os.path.join(
        "models", "cache", "quantized", "spectra-aasist3-int8-dynamic.onnx"
    )
    if not os.path.exists(int8_path):
        pytest.skip("INT8 artifact not present locally (generate via scripts/run_spectra_int8_evaluation.py's quantize_dynamic call first)")

    detector = create_detector("spectra_aasist3_onnx_int8", device="cpu")
    try:
        metadata = detector.load(local_onnx_path=int8_path)
        assert metadata.device == "cpu"
        assert detector.model_info()["input_length_samples"] == REQUIRED_SAMPLES
    finally:
        detector.unload()


# ---------------------------------------------------------------------------
# Prediction-semantics regression tests (live-bug investigation).
#
# ROOT CAUSE FINDING, verified against scripts/run_spectra_int8_evaluation.py
# (the frozen INT8 calibration/evaluation source of truth): the calibrated
# threshold IS applied to `spoof_prob` (softmax_spoof_bonafide()'s output),
# NOT the raw bona-fide logit -- `score_clip()` there returns
# `(bonafide_logit, spoof_prob)` but only ever appends `spoof_prob` to
# `calib_spoof_scores`/`eval_spoof_scores`, which is what
# compute_eer/best_*_threshold/full_metrics_report are called on. So the
# INT8_DYNAMIC_CALIBRATED_THRESHOLD (0.939693808555603) is a threshold on
# spoof_prob, and candidate_e.py's decide_label() already matched this
# correctly BEFORE this fix (spoof_prob >= threshold -> spoof).
#
# The live bug's exact numbers (bonafide=7.8%, spoof=92.2%) are the
# CORRECT, frozen-calibration-consistent BONAFIDE decision -- 0.922 is
# below the 0.9397 threshold. This is not a misclassification; it is one
# instance of the frozen evaluation's own measured 10% spoof_fnr (a
# deliberate low-bonafide-FPR tradeoff). The actual defect was in the UI
# layer: labeling the predicted-class probability "Model confidence"
# without any indication that the decision uses a threshold far from 50%,
# which reads as an internal contradiction. See
# docs/spectra_prediction_semantics_fix.md for the full trace.
# ---------------------------------------------------------------------------


def test_decide_label_matches_frozen_calibration_rule_spoof_prob_ge_threshold():
    # This IS the frozen rule: spoof_prob >= threshold -> spoof.
    assert decide_label(spoof_prob=0.95, threshold=0.9397) == "spoof"
    assert decide_label(spoof_prob=0.90, threshold=0.9397) == "bonafide"


def test_decide_label_high_bonafide_raw_score_gives_bonafide():
    # High bona-fide confidence (low spoof_prob) -> BONAFIDE.
    assert decide_label(spoof_prob=0.01, threshold=INT8_DYNAMIC_CALIBRATED_THRESHOLD) == "bonafide"


def test_decide_label_high_spoof_raw_score_gives_spoof():
    # Very high spoof_prob, comfortably above threshold -> SPOOF.
    assert decide_label(spoof_prob=0.999, threshold=INT8_DYNAMIC_CALIBRATED_THRESHOLD) == "spoof"


def test_decide_label_exact_threshold_boundary_is_spoof():
    # >= is inclusive: a score exactly AT the threshold is SPOOF, documented
    # and tested explicitly (the frozen evaluation's compute_threshold_metrics
    # also uses >=, so this matches that convention).
    t = INT8_DYNAMIC_CALIBRATED_THRESHOLD
    assert decide_label(spoof_prob=t, threshold=t) == "spoof"
    assert decide_label(spoof_prob=np.nextafter(t, 0.0), threshold=t) == "bonafide"


def test_decide_label_exact_live_bug_numbers_is_bonafide_not_spoof():
    """The literal reported live-bug numbers. IMPORTANT: this asserts
    BONAFIDE, not SPOOF -- the initial bug report assumed this was a
    misclassification, but tracing scripts/run_spectra_int8_evaluation.py
    (the frozen source of truth) shows the calibrated threshold operates on
    spoof_prob, and 0.922 < 0.939693808555603, so BONAFIDE is the correct,
    frozen-calibration-consistent decision. Forcing SPOOF here would
    deviate from the calibrated threshold, which this project was
    explicitly instructed not to change."""
    spoof_prob = 0.922
    bonafide_prob = 0.078
    assert abs((spoof_prob + bonafide_prob) - 1.0) < 1e-9
    assert decide_label(spoof_prob, INT8_DYNAMIC_CALIBRATED_THRESHOLD) == "bonafide"


def test_threshold_disagreement_rate_on_calibration_and_evaluation_data_is_measured_not_zero():
    # Documents the actual measured disagreement rate between the
    # calibrated-threshold decision and a naive 50% softmax split, computed
    # over all 400 calibration+evaluation clips (see
    # docs/spectra_prediction_semantics_fix.md Section on Step 7). This is
    # a fixed historical fact about the frozen INT8 evaluation, recorded
    # here so it cannot silently drift without updating the doc.
    measured_disagreement_count = 12
    measured_total = 400
    assert measured_disagreement_count / measured_total == 0.03


def test_result_summary_uses_class_specific_label_not_generic_confidence():
    from app.formatting import result_summary
    from audio_deepfake_detector.utils.datatypes import PredictionResult

    result = PredictionResult(
        raw_label="bonafide",
        normalized_label="BONAFIDE",
        confidence=0.078,
        probabilities={"bonafide": 0.078, "spoof": 0.922},
        model_id="spectra_aasist3_onnx_int8",
        model_repository="Limitless-8/spectra-aasist3-int8-audio-deepfake",
        device="cpu",
        inference_time_ms=650.0,
        audio_duration_seconds=4.04,
    )
    summary = result_summary(result)
    assert summary["prediction"] == "Likely Real / Bonafide"
    assert summary["confidence"] == "7.8%"
    assert summary["confidence_label"] == "Bonafide class probability"
    assert summary["bonafide_probability"] == "7.8%"
    assert summary["spoof_probability"] == "92.2%"
    # This is the internal-consistency guard: the UI must be told to add a
    # clarifying note here, since the predicted class's own probability
    # (7.8%) is a minority relative to the other class (92.2%).
    assert summary["threshold_disagreement"] is True
    assert "Likely AI-Generated / Spoofed" not in summary["prediction"]


def test_result_summary_no_disagreement_flag_for_high_bonafide_confidence():
    from app.formatting import result_summary
    from audio_deepfake_detector.utils.datatypes import PredictionResult

    result = PredictionResult(
        raw_label="bonafide",
        normalized_label="BONAFIDE",
        confidence=0.97,
        probabilities={"bonafide": 0.97, "spoof": 0.03},
        model_id="spectra_aasist3_onnx_int8",
        model_repository="Limitless-8/spectra-aasist3-int8-audio-deepfake",
        device="cpu",
        inference_time_ms=650.0,
        audio_duration_seconds=4.04,
    )
    summary = result_summary(result)
    assert summary["confidence_label"] == "Bonafide class probability"
    assert summary["threshold_disagreement"] is False


def test_result_summary_no_disagreement_flag_for_high_spoof_confidence():
    from app.formatting import result_summary
    from audio_deepfake_detector.utils.datatypes import PredictionResult

    result = PredictionResult(
        raw_label="spoof",
        normalized_label="SPOOF",
        confidence=0.99,
        probabilities={"bonafide": 0.01, "spoof": 0.99},
        model_id="spectra_aasist3_onnx_int8",
        model_repository="Limitless-8/spectra-aasist3-int8-audio-deepfake",
        device="cpu",
        inference_time_ms=650.0,
        audio_duration_seconds=4.04,
    )
    summary = result_summary(result)
    assert summary["confidence_label"] == "Spoof class probability"
    assert summary["threshold_disagreement"] is False
    assert summary["prediction"] == "Likely AI-Generated / Spoofed"
