"""Envelope extraction backed by VoxAtlas acoustic envelope features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from voxatlas.audio.audio import Audio
from voxatlas.features.acoustic.envelope.oganian import OganianEnvelope
from voxatlas.features.feature_input import FeatureInput

from .common import (
    DEFAULT_ALIGNMENT_TARGET,
    LINEAR_RESAMPLING_METHOD,
    ContinuousFeatureMetadata,
    ContinuousFeatureResult,
    align_by_linear_interpolation,
    eeg_time_axis,
    get_voxatlas_version,
)


@dataclass(frozen=True, slots=True)
class EnvelopeExtractionConfig:
    """Configuration for VoxAtlas Oganian-style envelope extraction."""

    frame_length_seconds: float = 0.025
    frame_step_seconds: float = 0.010
    smoothing_frames: int = 7
    peak_threshold: float = 0.1
    alignment_target: str = DEFAULT_ALIGNMENT_TARGET
    notes: list[str] = field(default_factory=list)


EnvelopeFeatureResult = ContinuousFeatureResult


def extract_envelope_feature(
    waveform: np.ndarray,
    audio_sampling_rate_hz: int,
    eeg_sampling_rate_hz: float,
    eeg_sample_count: int,
    config: EnvelopeExtractionConfig | None = None,
) -> EnvelopeFeatureResult:
    """Extract and EEG-align the VoxAtlas Oganian envelope.

    Parameters
    ----------
    waveform
        One-dimensional speech waveform.
    audio_sampling_rate_hz
        Source audio sampling rate in Hertz.
    eeg_sampling_rate_hz
        Target EEG sampling rate in Hertz.
    eeg_sample_count
        Target EEG sample count.
    config
        Envelope extraction configuration.

    Returns
    -------
    EnvelopeFeatureResult
        Raw framewise envelope and EEG-aligned envelope derivative.
    """
    resolved_config = config or EnvelopeExtractionConfig()
    extractor = OganianEnvelope()
    params = {
        "frame_length": resolved_config.frame_length_seconds,
        "frame_step": resolved_config.frame_step_seconds,
        "smoothing": resolved_config.smoothing_frames,
        "peak_threshold": resolved_config.peak_threshold,
    }

    feature_input = FeatureInput(
        audio=Audio(waveform=np.asarray(waveform, dtype=np.float32), sample_rate=audio_sampling_rate_hz),
        units=None,
        context={},
    )
    raw_output = extractor.compute(feature_input, params)
    eeg_time_seconds = eeg_time_axis(eeg_sample_count, eeg_sampling_rate_hz)
    eeg_values = align_by_linear_interpolation(raw_output.time, raw_output.values, eeg_time_seconds)

    metadata = ContinuousFeatureMetadata(
        feature_name="envelope",
        audio_sampling_rate_hz=int(audio_sampling_rate_hz),
        eeg_sampling_rate_hz=float(eeg_sampling_rate_hz),
        eeg_sample_count=int(eeg_sample_count),
        extraction_parameters={
            "frame_length_seconds": resolved_config.frame_length_seconds,
            "frame_step_seconds": resolved_config.frame_step_seconds,
            "smoothing_frames": resolved_config.smoothing_frames,
            "peak_threshold": resolved_config.peak_threshold,
        },
        voxatlas_version=get_voxatlas_version(),
        voxatlas_function="voxatlas.features.acoustic.envelope.oganian.OganianEnvelope",
        resampling_method=LINEAR_RESAMPLING_METHOD,
        alignment_target=resolved_config.alignment_target,
        units="amplitude",
        shape=tuple(eeg_values.shape),
        notes=list(resolved_config.notes),
    )

    return EnvelopeFeatureResult(
        raw_time_seconds=np.asarray(raw_output.time, dtype=np.float32),
        raw_values=np.asarray(raw_output.values, dtype=np.float32),
        eeg_time_seconds=eeg_time_seconds,
        eeg_values=eeg_values,
        metadata=metadata,
    )
