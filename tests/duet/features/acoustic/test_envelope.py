"""Unit tests for the VoxAtlas-backed envelope wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from duet.features.acoustic.common import align_by_linear_interpolation
from duet.features.acoustic.envelope import EnvelopeExtractionConfig, extract_envelope_feature


def test_envelope_alignment_matches_requested_eeg_length() -> None:
    """Envelope output should be aligned to the requested EEG sample count."""
    sampling_rate_hz = 16000
    time_seconds = np.arange(0, 1.0, 1.0 / sampling_rate_hz, dtype=np.float32)
    waveform = np.sin(2.0 * np.pi * 220.0 * time_seconds).astype(np.float32)

    result = extract_envelope_feature(
        waveform=waveform,
        audio_sampling_rate_hz=sampling_rate_hz,
        eeg_sampling_rate_hz=512.0,
        eeg_sample_count=512,
        config=EnvelopeExtractionConfig(),
    )

    assert result.eeg_values.shape == (512,)
    assert np.isfinite(result.eeg_values).all()
    assert result.metadata.shape == (512,)


def test_linear_alignment_helper_returns_exact_target_length() -> None:
    """Shared linear alignment helper should preserve the exact target length."""
    source_time_seconds = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    source_values = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    target_time_seconds = np.linspace(0.0, 0.2, 11, dtype=np.float32)

    aligned = align_by_linear_interpolation(source_time_seconds, source_values, target_time_seconds)

    assert aligned.shape == (11,)
    assert float(aligned[5]) == 1.0


def test_envelope_uses_broadband_oganian_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrapper should forward the broadband Oganian configuration to VoxAtlas."""
    captured: dict[str, object] = {}

    class DummyExtractor:
        def compute(self, feature_input, params):
            captured["params"] = dict(params)

            class Output:
                time = np.array([0.0, 0.1], dtype=np.float32)
                values = np.array([1.0, 2.0], dtype=np.float32)

            return Output()

    monkeypatch.setattr("duet.features.acoustic.envelope.OganianEnvelope", lambda: DummyExtractor())

    result = extract_envelope_feature(
        waveform=np.ones(1600, dtype=np.float32),
        audio_sampling_rate_hz=16000,
        eeg_sampling_rate_hz=10.0,
        eeg_sample_count=2,
        config=EnvelopeExtractionConfig(
            envtype="broadband",
            broadband_lowpass_hz=10.0,
            smoothing_frames=1,
        ),
    )

    assert captured["params"] == {
        "frame_length": 0.025,
        "frame_step": 0.010,
        "envtype": "broadband",
        "broadband_lowpass_hz": 10.0,
        "smoothing": 1,
    }
    assert result.metadata.extraction_parameters["envtype"] == "broadband"
    assert result.metadata.extraction_parameters["broadband_lowpass_hz"] == 10.0
