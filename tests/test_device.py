"""Tests for device resolution (must always be CPU-safe)."""

import pytest

from audio_deepfake_detector.utils.device import resolve_device


def test_cpu_resolves_to_cpu():
    assert resolve_device("cpu") == "cpu"


def test_auto_falls_back_to_cpu():
    assert resolve_device("auto") == "cpu"


def test_unsupported_device_rejected():
    with pytest.raises(ValueError):
        resolve_device("cuda")
