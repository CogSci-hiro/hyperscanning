"""Unit tests for VoxAtlas-backed pitch extraction."""

from __future__ import annotations

import numpy as np
import pytest

from hyper.features.acoustic import pitch as mod
from hyper.features.acoustic.pitch import PitchExtractionConfig, extract_f0_feature


def test_pitch_preserves_unvoiced_regions_in_raw_track() -> None:
    """Raw F0 output should keep unvoiced regions as NaN."""
    sampling_rate_hz = 16000
    voiced = np.sin(2.0 * np.pi * 120.0 * np.arange(0, 0.25, 1.0 / sampling_rate_hz))
    silence = np.zeros(int(0.25 * sampling_rate_hz), dtype=np.float32)
    waveform = np.concatenate([voiced.astype(np.float32), silence, voiced.astype(np.float32)])

    result = extract_f0_feature(
        waveform=waveform,
        audio_sampling_rate_hz=sampling_rate_hz,
        eeg_sampling_rate_hz=512.0,
        eeg_sample_count=384,
        config=PitchExtractionConfig(fill_strategy="nan"),
    )

    assert result.eeg_values.shape == (384,)
    assert np.count_nonzero(~np.isfinite(result.raw_values)) > 0
    assert result.metadata.raw_unvoiced_frame_count > 0
    assert result.metadata.fill_strategy == "nan"


def test_pitch_linear_fill_produces_finite_trf_ready_values() -> None:
    """Linear filling should yield a finite EEG-aligned regressor."""
    sampling_rate_hz = 16000
    voiced = np.sin(2.0 * np.pi * 180.0 * np.arange(0, 0.5, 1.0 / sampling_rate_hz))
    silence = np.zeros(int(0.10 * sampling_rate_hz), dtype=np.float32)
    waveform = np.concatenate([voiced.astype(np.float32), silence, voiced.astype(np.float32)])

    result = extract_f0_feature(
        waveform=waveform,
        audio_sampling_rate_hz=sampling_rate_hz,
        eeg_sampling_rate_hz=256.0,
        eeg_sample_count=281,
        config=PitchExtractionConfig(fill_strategy="linear"),
    )

    assert result.eeg_values.shape == (281,)
    assert np.isfinite(result.eeg_values).all()
    assert result.metadata.shape == (281,)


def test_pitch_falls_back_when_filtered_autocorrelation_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Older Parselmouth/Praat builds should trigger the local autocorrelation fallback."""
    class DummyExtractor:
        def compute(self, feature_input, params):
            raise RuntimeError(
                "Praat filtered-autocorrelation pitch extraction failed. "
                "Check the F0 configuration and ensure the installed Parselmouth "
                "version supports 'To Pitch (filtered autocorrelation)...'."
            )

    monkeypatch.setattr(mod, "F0Extractor", DummyExtractor)

    sampling_rate_hz = 16000
    waveform = np.sin(2.0 * np.pi * 140.0 * np.arange(0, 0.5, 1.0 / sampling_rate_hz)).astype(np.float32)

    result = extract_f0_feature(
        waveform=waveform,
        audio_sampling_rate_hz=sampling_rate_hz,
        eeg_sampling_rate_hz=256.0,
        eeg_sample_count=128,
        config=PitchExtractionConfig(fill_strategy="linear"),
    )

    assert result.eeg_values.shape == (128,)
    assert np.isfinite(result.eeg_values).all()
    assert result.metadata.extraction_parameters["extractor_backend"] == "parselmouth.to_pitch_ac"
    assert result.metadata.voxatlas_function == "parselmouth.to_pitch_ac"
