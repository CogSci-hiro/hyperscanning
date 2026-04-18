"""Pitch extraction and TRF-ready F0 postprocessing backed by VoxAtlas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from voxatlas.audio.audio import Audio
from voxatlas.features.acoustic.pitch.f0 import F0Extractor
from voxatlas.features.feature_input import FeatureInput

from .common import (
    DEFAULT_ALIGNMENT_TARGET,
    LINEAR_RESAMPLING_METHOD,
    NEAREST_RESAMPLING_METHOD,
    ContinuousFeatureMetadata,
    ContinuousFeatureResult,
    align_by_linear_interpolation,
    align_by_nearest_samples,
    eeg_time_axis,
    get_voxatlas_version,
)


PitchFillStrategy = Literal["nan", "zero", "linear", "forward_fill"]
NAN_FILL_STRATEGY: PitchFillStrategy = "nan"


@dataclass(frozen=True, slots=True)
class PitchExtractionConfig:
    """Configuration for VoxAtlas F0 extraction and TRF filling."""

    fmin_hz: float = 75.0
    fmax_hz: float = 500.0
    frame_length_seconds: float = 0.040
    frame_step_seconds: float = 0.010
    fill_strategy: PitchFillStrategy = NAN_FILL_STRATEGY
    alignment_target: str = DEFAULT_ALIGNMENT_TARGET


@dataclass(frozen=True, slots=True)
class F0Metadata(ContinuousFeatureMetadata):
    """Pitch-specific metadata with explicit unvoiced handling details."""

    raw_unvoiced_frame_count: int
    raw_frame_count: int
    raw_unvoiced_value: str
    fill_strategy: PitchFillStrategy


F0FeatureResult = ContinuousFeatureResult

FILTERED_AUTOCORRELATION_COMMAND = "To Pitch (filtered autocorrelation)..."


def _is_filtered_autocorrelation_compatibility_error(exc: Exception) -> bool:
    """Return True when Praat/Parselmouth lacks filtered-autocorrelation support."""
    text = str(exc)
    return FILTERED_AUTOCORRELATION_COMMAND in text or "filtered-autocorrelation" in text.lower()


def _compute_f0_with_parselmouth_autocorrelation(
    waveform: np.ndarray,
    audio_sampling_rate_hz: int,
    config: PitchExtractionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback F0 extraction using Parselmouth's standard autocorrelation tracker."""
    import parselmouth

    sound = parselmouth.Sound(
        np.asarray(waveform, dtype=np.float64),
        sampling_frequency=float(audio_sampling_rate_hz),
    )
    pitch = sound.to_pitch_ac(
        time_step=float(config.frame_step_seconds),
        pitch_floor=float(config.fmin_hz),
        pitch_ceiling=float(config.fmax_hz),
    )
    raw_time_seconds = np.asarray(pitch.xs(), dtype=np.float32)
    raw_values = np.asarray(pitch.selected_array["frequency"], dtype=np.float32)
    raw_values[raw_values <= 0.0] = np.nan
    return raw_time_seconds, raw_values


def _extract_raw_f0_contour(
    waveform: np.ndarray,
    audio_sampling_rate_hz: int,
    config: PitchExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Extract a raw F0 contour, falling back when VoxAtlas needs newer Praat support."""
    extractor = F0Extractor()
    params = {
        "fmin": config.fmin_hz,
        "fmax": config.fmax_hz,
        "frame_length": config.frame_length_seconds,
        "frame_step": config.frame_step_seconds,
    }
    feature_input = FeatureInput(
        audio=Audio(waveform=np.asarray(waveform, dtype=np.float32), sample_rate=audio_sampling_rate_hz),
        units=None,
        context={},
    )
    try:
        raw_output = extractor.compute(feature_input, params)
    except RuntimeError as exc:
        if not _is_filtered_autocorrelation_compatibility_error(exc):
            raise
        raw_time_seconds, raw_values = _compute_f0_with_parselmouth_autocorrelation(
            waveform=waveform,
            audio_sampling_rate_hz=audio_sampling_rate_hz,
            config=config,
        )
        return raw_time_seconds, raw_values, "parselmouth.to_pitch_ac"

    return (
        np.asarray(raw_output.time, dtype=np.float32),
        np.asarray(raw_output.values, dtype=np.float32),
        "voxatlas.features.acoustic.pitch.f0.F0Extractor",
    )


def _fill_unvoiced_frames(
    time_seconds: np.ndarray,
    f0_values_hz: np.ndarray,
    strategy: PitchFillStrategy,
) -> tuple[np.ndarray, str]:
    """Apply the configured filling strategy to a raw F0 contour."""
    if strategy == "nan":
        return f0_values_hz.astype(np.float32, copy=True), NEAREST_RESAMPLING_METHOD

    voiced_mask = np.isfinite(f0_values_hz)
    if not np.any(voiced_mask):
        if strategy == "zero":
            return np.zeros(f0_values_hz.shape, dtype=np.float32), LINEAR_RESAMPLING_METHOD
        return np.full(f0_values_hz.shape, np.nan, dtype=np.float32), LINEAR_RESAMPLING_METHOD

    if strategy == "zero":
        filled = np.where(voiced_mask, f0_values_hz, 0.0)
        return filled.astype(np.float32), LINEAR_RESAMPLING_METHOD

    if strategy == "linear":
        filled = np.interp(
            time_seconds.astype(np.float64, copy=False),
            time_seconds[voiced_mask].astype(np.float64, copy=False),
            f0_values_hz[voiced_mask].astype(np.float64, copy=False),
            left=float(f0_values_hz[voiced_mask][0]),
            right=float(f0_values_hz[voiced_mask][-1]),
        )
        return filled.astype(np.float32), LINEAR_RESAMPLING_METHOD

    if strategy == "forward_fill":
        filled = f0_values_hz.astype(np.float32, copy=True)
        valid_indices = np.flatnonzero(voiced_mask)
        first_valid_index = int(valid_indices[0])
        filled[:first_valid_index] = filled[first_valid_index]
        for index in range(first_valid_index + 1, filled.size):
            if not np.isfinite(filled[index]):
                filled[index] = filled[index - 1]
        return filled.astype(np.float32), LINEAR_RESAMPLING_METHOD

    raise ValueError(f"Unsupported fill strategy: {strategy}")


def extract_f0_feature(
    waveform: np.ndarray,
    audio_sampling_rate_hz: int,
    eeg_sampling_rate_hz: float,
    eeg_sample_count: int,
    config: PitchExtractionConfig | None = None,
) -> F0FeatureResult:
    """Extract raw F0 with explicit unvoiced regions and align to EEG samples."""
    resolved_config = config or PitchExtractionConfig()
    raw_time_seconds, raw_values, extractor_backend = _extract_raw_f0_contour(
        waveform=waveform,
        audio_sampling_rate_hz=audio_sampling_rate_hz,
        config=resolved_config,
    )

    filled_values, resampling_method = _fill_unvoiced_frames(
        raw_time_seconds,
        raw_values,
        resolved_config.fill_strategy,
    )

    eeg_time_seconds = eeg_time_axis(eeg_sample_count, eeg_sampling_rate_hz)
    if resampling_method == NEAREST_RESAMPLING_METHOD:
        eeg_values = align_by_nearest_samples(raw_time_seconds, filled_values, eeg_time_seconds)
    else:
        eeg_values = align_by_linear_interpolation(raw_time_seconds, filled_values, eeg_time_seconds)

    metadata = F0Metadata(
        feature_name="f0",
        audio_sampling_rate_hz=int(audio_sampling_rate_hz),
        eeg_sampling_rate_hz=float(eeg_sampling_rate_hz),
        eeg_sample_count=int(eeg_sample_count),
        extraction_parameters={
            "fmin_hz": resolved_config.fmin_hz,
            "fmax_hz": resolved_config.fmax_hz,
            "frame_length_seconds": resolved_config.frame_length_seconds,
            "frame_step_seconds": resolved_config.frame_step_seconds,
            "extractor_backend": extractor_backend,
        },
        voxatlas_version=get_voxatlas_version(),
        voxatlas_function=extractor_backend,
        resampling_method=resampling_method,
        alignment_target=resolved_config.alignment_target,
        units="Hz",
        shape=tuple(eeg_values.shape),
        notes=[
            "Raw extracted F0 stores unvoiced frames as NaN.",
            "EEG-aligned values reflect the configured fill strategy.",
            "Falls back to Parselmouth autocorrelation if VoxAtlas requests an unsupported Praat filtered-autocorrelation command.",
        ],
        raw_unvoiced_frame_count=int(np.count_nonzero(~np.isfinite(raw_values))),
        raw_frame_count=int(raw_values.size),
        raw_unvoiced_value="NaN",
        fill_strategy=resolved_config.fill_strategy,
    )

    return F0FeatureResult(
        raw_time_seconds=raw_time_seconds,
        raw_values=raw_values,
        eeg_time_seconds=eeg_time_seconds,
        eeg_values=eeg_values,
        metadata=metadata,
    )
