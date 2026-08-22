"""Tests for the deterministic smoke-audio generator (not speech; no ML)."""

import numpy as np
import soundfile as sf

from scripts.generate_smoke_audio import SAMPLE_RATE, generate_all


def test_generate_all_writes_expected_files(tmp_path):
    written = generate_all(output_dir=tmp_path)

    assert len(written) == 5
    for path in written:
        assert path.exists()
        data, sr = sf.read(str(path))
        assert sr == SAMPLE_RATE
        assert data.shape[0] > 0


def test_generated_audio_is_deterministic(tmp_path):
    generate_all(output_dir=tmp_path)
    first_run = sf.read(str(tmp_path / "smoke_sine_1s.wav"))[0]

    generate_all(output_dir=tmp_path)
    second_run = sf.read(str(tmp_path / "smoke_sine_1s.wav"))[0]

    assert np.array_equal(first_run, second_run)


def test_silence_file_is_all_zero(tmp_path):
    generate_all(output_dir=tmp_path)
    data, _ = sf.read(str(tmp_path / "smoke_silence_2s.wav"))
    assert np.all(data == 0.0)
