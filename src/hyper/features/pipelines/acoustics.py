"""Thin application/pipeline wrappers for acoustic feature derivatives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hyper.features.acoustic.common import dataclass_to_dict
from hyper.features.acoustic.envelope import (
    EnvelopeExtractionConfig,
    EnvelopeFeatureResult,
    extract_envelope_feature,
)
from hyper.features.acoustic.formants import (
    FormantEventExtractionConfig,
    FormantEventResult,
    extract_vowel_formant_events,
    load_vowel_intervals_from_textgrid,
)
from hyper.features.acoustic.pitch import F0FeatureResult, PitchExtractionConfig, extract_f0_feature


JSON_INDENT_SPACES: int = 2


def load_audio_waveform(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load a mono waveform from disk for acoustic feature extraction."""
    import parselmouth

    sound = parselmouth.Sound(str(audio_path))
    waveform = sound.values.T
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    return waveform, int(sound.sampling_frequency)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=JSON_INDENT_SPACES, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_continuous_derivative(
    result: EnvelopeFeatureResult | F0FeatureResult,
    output_values_path: Path,
    output_sidecar_path: Path,
) -> None:
    output_values_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_values_path, result.eeg_values)
    _write_json(
        output_sidecar_path,
        {
            "metadata": dataclass_to_dict(result.metadata),
            "raw_frame_count": int(result.raw_values.size),
            "raw_time_seconds_shape": list(result.raw_time_seconds.shape),
            "aligned_time_seconds_shape": list(result.eeg_time_seconds.shape),
        },
    )


def _write_event_derivative(result: FormantEventResult, output_tsv_path: Path, output_sidecar_path: Path) -> None:
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    result.event_table.to_csv(output_tsv_path, sep="\t", index=False)
    _write_json(output_sidecar_path, {"metadata": dataclass_to_dict(result.metadata)})


def run_envelope_pipeline(
    audio_path: Path,
    eeg_sampling_rate_hz: float,
    eeg_sample_count: int,
    output_values_path: Path,
    output_sidecar_path: Path,
    config: EnvelopeExtractionConfig | None = None,
) -> EnvelopeFeatureResult:
    """Extract the EEG-aligned acoustic envelope derivative and write it."""
    waveform, sampling_rate_hz = load_audio_waveform(audio_path)
    result = extract_envelope_feature(
        waveform=waveform,
        audio_sampling_rate_hz=sampling_rate_hz,
        eeg_sampling_rate_hz=eeg_sampling_rate_hz,
        eeg_sample_count=eeg_sample_count,
        config=config,
    )
    _write_continuous_derivative(result, output_values_path, output_sidecar_path)
    return result


def run_pitch_pipeline(
    audio_path: Path,
    eeg_sampling_rate_hz: float,
    eeg_sample_count: int,
    output_values_path: Path,
    output_sidecar_path: Path,
    config: PitchExtractionConfig | None = None,
) -> F0FeatureResult:
    """Extract the EEG-aligned F0 derivative and write it."""
    waveform, sampling_rate_hz = load_audio_waveform(audio_path)
    result = extract_f0_feature(
        waveform=waveform,
        audio_sampling_rate_hz=sampling_rate_hz,
        eeg_sampling_rate_hz=eeg_sampling_rate_hz,
        eeg_sample_count=eeg_sample_count,
        config=config,
    )
    _write_continuous_derivative(result, output_values_path, output_sidecar_path)
    return result


def run_vowel_formant_pipeline(
    audio_path: Path,
    textgrid_path: Path,
    tier_name: str,
    output_tsv_path: Path,
    output_sidecar_path: Path,
    config: FormantEventExtractionConfig | None = None,
    speaker: str | None = None,
) -> FormantEventResult:
    """Extract vowel-centered F1/F2 event features from TextGrid intervals."""
    resolved_config = config or FormantEventExtractionConfig()
    waveform, sampling_rate_hz = load_audio_waveform(audio_path)
    vowel_intervals = load_vowel_intervals_from_textgrid(
        textgrid_path=str(textgrid_path),
        tier_name=tier_name,
        speaker=speaker,
        language=resolved_config.language,
        resource_root=resolved_config.resource_root,
    )
    result = extract_vowel_formant_events(
        waveform=waveform,
        audio_sampling_rate_hz=sampling_rate_hz,
        vowel_intervals=vowel_intervals,
        config=resolved_config,
    )
    _write_event_derivative(result, output_tsv_path, output_sidecar_path)
    return result
