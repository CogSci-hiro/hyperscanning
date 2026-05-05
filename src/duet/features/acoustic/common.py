"""Shared acoustic feature datatypes and alignment helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np


DEFAULT_ALIGNMENT_TARGET: str = "eeg_samples"
LINEAR_RESAMPLING_METHOD: str = "linear"
NEAREST_RESAMPLING_METHOD: str = "nearest"


@dataclass(frozen=True, slots=True)
class ContinuousFeatureMetadata:
    """Metadata shared by EEG-aligned continuous acoustic derivatives.

    Parameters
    ----------
    feature_name
        Human-readable feature identifier.
    audio_sampling_rate_hz
        Source audio sampling rate in Hertz.
    eeg_sampling_rate_hz
        EEG sampling rate used as the alignment target.
    eeg_sample_count
        Number of EEG samples in the aligned output.
    extraction_parameters
        Resolved feature extraction parameters.
    voxatlas_version
        Installed VoxAtlas package version if available.
    voxatlas_function
        VoxAtlas class or helper used for extraction.
    resampling_method
        Resampling method used to align to EEG samples.
    alignment_target
        Semantic target for the aligned vector.
    units
        Measurement units for the output values.
    shape
        Shape of the aligned output.
    notes
        Optional implementation notes and caveats.
    """

    feature_name: str
    audio_sampling_rate_hz: int
    eeg_sampling_rate_hz: float
    eeg_sample_count: int
    extraction_parameters: dict[str, Any]
    voxatlas_version: str | None
    voxatlas_function: str
    resampling_method: str
    alignment_target: str
    units: str
    shape: tuple[int, ...]
    notes: list[str]


@dataclass(frozen=True, slots=True)
class ContinuousFeatureResult:
    """Continuous feature representation with raw and EEG-aligned views."""

    raw_time_seconds: np.ndarray
    raw_values: np.ndarray
    eeg_time_seconds: np.ndarray
    eeg_values: np.ndarray
    metadata: ContinuousFeatureMetadata


def get_voxatlas_version() -> str | None:
    """Return the installed VoxAtlas version when available."""
    try:
        return version("voxatlas")
    except PackageNotFoundError:
        return None


def eeg_time_axis(sample_count: int, sampling_rate_hz: float) -> np.ndarray:
    """Construct the canonical EEG time axis for a sample count."""
    if sample_count < 0:
        raise ValueError("eeg_sample_count must be non-negative")
    return np.arange(sample_count, dtype=np.float32) / np.float32(sampling_rate_hz)


def align_by_linear_interpolation(
    source_time_seconds: np.ndarray,
    source_values: np.ndarray,
    target_time_seconds: np.ndarray,
) -> np.ndarray:
    """Linearly resample a finite source contour onto target timestamps."""
    if source_time_seconds.size == 0:
        raise ValueError("source_time_seconds must not be empty")

    finite_mask = np.isfinite(source_values)
    if not np.any(finite_mask):
        return np.full(target_time_seconds.shape, np.nan, dtype=np.float32)

    finite_times = source_time_seconds[finite_mask].astype(np.float64, copy=False)
    finite_values = source_values[finite_mask].astype(np.float64, copy=False)

    if finite_times.size == 1:
        return np.full(
            target_time_seconds.shape,
            np.float32(finite_values[0]),
            dtype=np.float32,
        )

    aligned = np.interp(
        target_time_seconds.astype(np.float64, copy=False),
        finite_times,
        finite_values,
        left=finite_values[0],
        right=finite_values[-1],
    )
    return aligned.astype(np.float32)


def align_by_nearest_samples(
    source_time_seconds: np.ndarray,
    source_values: np.ndarray,
    target_time_seconds: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor alignment that preserves explicit NaN regions."""
    if source_time_seconds.size == 0:
        raise ValueError("source_time_seconds must not be empty")

    insertion_indices = np.searchsorted(source_time_seconds, target_time_seconds, side="left")
    insertion_indices = np.clip(insertion_indices, 0, source_time_seconds.size - 1)
    previous_indices = np.clip(insertion_indices - 1, 0, source_time_seconds.size - 1)

    choose_previous = np.abs(target_time_seconds - source_time_seconds[previous_indices]) <= np.abs(
        source_time_seconds[insertion_indices] - target_time_seconds
    )
    aligned_indices = np.where(choose_previous, previous_indices, insertion_indices)
    return source_values[aligned_indices].astype(np.float32, copy=False)


def dataclass_to_dict(value: Any) -> Any:
    """Convert nested dataclasses and NumPy scalars into JSON-safe objects."""
    if is_dataclass(value):
        return dataclass_to_dict(asdict(value))

    if isinstance(value, dict):
        return {str(key): dataclass_to_dict(item) for key, item in value.items()}

    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]

    if isinstance(value, tuple):
        return [dataclass_to_dict(item) for item in value]

    if isinstance(value, np.generic):
        return value.item()

    return value
