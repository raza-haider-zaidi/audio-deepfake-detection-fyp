"""Environment smoke test: confirms the local package is importable."""

import audio_deepfake_detector


def test_package_imports():
    assert audio_deepfake_detector.__version__ == "0.1.0"


def test_subpackages_import():
    import audio_deepfake_detector.config
    import audio_deepfake_detector.evaluation
    import audio_deepfake_detector.inference
    import audio_deepfake_detector.models
    import audio_deepfake_detector.preprocessing
    import audio_deepfake_detector.utils
