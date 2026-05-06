"""Thin application/pipeline wrappers for acoustic feature derivatives."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from duet.features.acoustic.common import dataclass_to_dict
from duet.features.acoustic.envelope import (
    EnvelopeExtractionConfig,
    EnvelopeFeatureResult,
    extract_envelope_feature,
)
from duet.features.acoustic.formants import (
    FormantEventExtractionConfig,
    FormantEventResult,
    extract_vowel_formant_events,
    load_vowel_intervals,
)
from duet.features.acoustic.pitch import F0FeatureResult, PitchExtractionConfig, extract_f0_feature


JSON_INDENT_SPACES: int = 2
DEFAULT_IPU_SILENCE_LABELS: tuple[str, ...] = ("#",)
ALIGNMENT_EVENT_COLUMNS: tuple[str, ...] = (
    "onset_seconds",
    "duration_seconds",
    "label",
    "speaker",
    "source_subject",
    "source_role",
    "source_interval_id",
)


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


def _resolve_ipu_time_bounds(table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Resolve interval start/end times from a supported IPU table schema."""
    if {"start", "end"}.issubset(table.columns):
        start = pd.to_numeric(table["start"], errors="coerce")
        end = pd.to_numeric(table["end"], errors="coerce")
        return start, end
    if {"onset", "duration"}.issubset(table.columns):
        start = pd.to_numeric(table["onset"], errors="coerce")
        duration = pd.to_numeric(table["duration"], errors="coerce")
        return start, start + duration
    if {"onset_seconds", "duration_seconds"}.issubset(table.columns):
        start = pd.to_numeric(table["onset_seconds"], errors="coerce")
        duration = pd.to_numeric(table["duration_seconds"], errors="coerce")
        return start, start + duration
    raise ValueError("IPU table must contain either start/end, onset/duration, or onset_seconds/duration_seconds columns.")


def _resolve_ipu_label_series(table: pd.DataFrame) -> pd.Series:
    """Resolve the label column from a supported IPU table schema."""
    for column_name in ("annotation", "label"):
        if column_name in table.columns:
            return table[column_name].astype(str)
    raise ValueError("IPU table must contain an annotation or label column.")


def zero_waveform_for_ipu_silences(
    waveform: np.ndarray,
    *,
    audio_sampling_rate_hz: int,
    ipu_path: Path,
    silence_labels: Sequence[str] = DEFAULT_IPU_SILENCE_LABELS,
) -> tuple[np.ndarray, dict[str, object]]:
    """Zero out waveform samples covered by silence-labeled IPU intervals."""
    silence_label_set = {str(label).strip() for label in silence_labels}
    if len(silence_label_set) == 0:
        return np.array(waveform, dtype=np.float32, copy=True), {
            "ipu_path": str(ipu_path),
            "silence_labels": [],
            "silence_interval_count": 0,
            "silence_sample_count": 0,
            "silence_duration_seconds": 0.0,
        }

    table = pd.read_csv(ipu_path)
    start_seconds, end_seconds = _resolve_ipu_time_bounds(table)
    labels = _resolve_ipu_label_series(table).str.strip()
    silence_rows = labels.isin(silence_label_set)

    masked_waveform = np.array(waveform, dtype=np.float32, copy=True)
    masked_sample_count = 0
    silence_interval_count = 0
    sample_rate = float(audio_sampling_rate_hz)

    for start_value, end_value in zip(start_seconds.loc[silence_rows], end_seconds.loc[silence_rows], strict=False):
        if not np.isfinite(start_value) or not np.isfinite(end_value):
            continue
        start_sample = max(0, int(np.floor(float(start_value) * sample_rate)))
        stop_sample = min(masked_waveform.shape[0], int(np.ceil(float(end_value) * sample_rate)))
        if stop_sample <= start_sample:
            continue
        masked_waveform[start_sample:stop_sample] = 0.0
        masked_sample_count += stop_sample - start_sample
        silence_interval_count += 1

    return masked_waveform, {
        "ipu_path": str(ipu_path),
        "silence_labels": sorted(silence_label_set),
        "silence_interval_count": int(silence_interval_count),
        "silence_sample_count": int(masked_sample_count),
        "silence_duration_seconds": float(masked_sample_count / sample_rate),
    }


def _write_continuous_derivative(
    result: EnvelopeFeatureResult | F0FeatureResult,
    output_values_path: Path,
    output_sidecar_path: Path,
    *,
    feature_name: str | None = None,
    source_subject: str | None = None,
    source_role: str | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> None:
    output_values_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_values_path, result.eeg_values)
    metadata = dataclass_to_dict(result.metadata)
    if feature_name is not None:
        metadata["feature_name"] = feature_name
    if source_subject is not None:
        metadata["source_subject"] = source_subject
    if source_role is not None:
        metadata["source_role"] = source_role
    if extra_metadata:
        metadata.update(extra_metadata)
    _write_json(
        output_sidecar_path,
        {
            "metadata": metadata,
            "raw_frame_count": int(result.raw_values.size),
            "raw_time_seconds_shape": list(result.raw_time_seconds.shape),
            "aligned_time_seconds_shape": list(result.eeg_time_seconds.shape),
        },
    )


def _write_event_derivative(
    result: FormantEventResult,
    output_tsv_path: Path,
    output_sidecar_path: Path,
    *,
    feature_name: str | None = None,
    source_subject: str | None = None,
    source_role: str | None = None,
) -> None:
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    result.event_table.to_csv(output_tsv_path, sep="\t", index=False)
    metadata = dataclass_to_dict(result.metadata)
    if feature_name is not None:
        metadata["feature_name"] = feature_name
    if source_subject is not None:
        metadata["source_subject"] = source_subject
    if source_role is not None:
        metadata["source_role"] = source_role
    _write_json(output_sidecar_path, {"metadata": metadata})


def run_alignment_event_pipeline(
    alignment_path: Path,
    tier_name: str,
    output_tsv_path: Path,
    output_sidecar_path: Path,
    *,
    feature_name: str,
    exclude_labels: tuple[str, ...] = (),
    source_subject: str | None = None,
    source_role: str | None = None,
) -> pd.DataFrame:
    """Export interval onsets from a simple alignment CSV as an event table."""
    intervals_df = pd.read_csv(
        alignment_path,
        header=None,
        names=["tier", "start", "end", "label"],
        encoding="utf-8-sig",
    )
    intervals_df["tier"] = intervals_df["tier"].astype(str).str.strip().str.strip('"')
    intervals_df["label"] = intervals_df["label"].astype(str).str.strip().str.strip('"')

    filtered = intervals_df.loc[intervals_df["tier"] == tier_name].copy()
    if exclude_labels:
        filtered = filtered.loc[~filtered["label"].isin(set(exclude_labels))].copy()
    filtered = filtered.loc[filtered["label"] != ""].reset_index(drop=True)

    inferred_speaker = None
    if source_subject is not None:
        try:
            _, inferred_speaker = infer_dyad_index_and_speaker(source_subject)
        except ValueError:
            inferred_speaker = None

    event_table = pd.DataFrame(
        {
            "onset_seconds": filtered["start"].astype(float),
            "duration_seconds": filtered["end"].astype(float) - filtered["start"].astype(float),
            "label": filtered["label"].astype(str),
            "speaker": inferred_speaker,
            "source_subject": source_subject,
            "source_role": source_role,
            "source_interval_id": filtered.index.astype(str),
        },
        columns=ALIGNMENT_EVENT_COLUMNS,
    )
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    event_table.to_csv(output_tsv_path, sep="\t", index=False)
    _write_json(
        output_sidecar_path,
        {
            "metadata": {
                "feature_name": feature_name,
                "alignment_target": "event_onsets",
                "source_alignment_path": str(alignment_path),
                "source_tier": tier_name,
                "excluded_labels": list(exclude_labels),
                "source_subject": source_subject,
                "source_role": source_role,
                "shape": list(event_table.shape),
            }
        },
    )
    return event_table


def infer_dyad_index_and_speaker(subject: str) -> tuple[str, str]:
    """Infer dyad index and speaker label from the project's odd/even subject pairing rule."""
    subject_num = int(str(subject).removeprefix("sub-"))
    dyad_index = str((subject_num + 1) // 2).zfill(3)
    speaker = "A" if (subject_num % 2 == 1) else "B"
    return dyad_index, speaker


def run_token_event_pipeline(
    tokens_path: Path,
    subject: str,
    run: str,
    output_tsv_path: Path,
    output_sidecar_path: Path,
    *,
    feature_name: str = "tokens",
    exclude_labels: tuple[str, ...] = (),
    source_subject: str | None = None,
    source_role: str | None = None,
) -> pd.DataFrame:
    """Export subject-specific token onsets from a dyad-level token table."""
    _, speaker = infer_dyad_index_and_speaker(subject)
    token_df = pd.read_csv(tokens_path)
    filtered = token_df.loc[
        (token_df["run"].astype(str) == str(run)) & (token_df["speaker"].astype(str) == speaker)
    ].copy()
    if exclude_labels:
        filtered = filtered.loc[~filtered["token"].astype(str).isin(set(exclude_labels))].copy()
    filtered = filtered.reset_index(drop=True)

    event_table = pd.DataFrame(
        {
            "onset_seconds": filtered["start"].astype(float),
            "duration_seconds": filtered["end"].astype(float) - filtered["start"].astype(float),
            "label": filtered["token"].astype(str),
            "speaker": speaker,
            "source_subject": source_subject if source_subject is not None else subject,
            "source_role": source_role,
            "source_interval_id": filtered.index.astype(str),
        },
        columns=ALIGNMENT_EVENT_COLUMNS,
    )
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    event_table.to_csv(output_tsv_path, sep="\t", index=False)
    _write_json(
        output_sidecar_path,
        {
            "metadata": {
                "feature_name": feature_name,
                "alignment_target": "event_onsets",
                "source_alignment_path": str(tokens_path),
                "source_tier": "tokens",
                "subject": subject,
                "run": str(run),
                "speaker": speaker,
                "excluded_labels": list(exclude_labels),
                "source_subject": source_subject if source_subject is not None else subject,
                "source_role": source_role,
                "shape": list(event_table.shape),
            }
        },
    )
    return event_table


def run_envelope_pipeline(
    audio_path: Path,
    eeg_sampling_rate_hz: float,
    eeg_sample_count: int,
    output_values_path: Path,
    output_sidecar_path: Path,
    config: EnvelopeExtractionConfig | None = None,
    *,
    feature_name: str | None = None,
    source_subject: str | None = None,
    source_role: str | None = None,
    ipu_path: Path | None = None,
    silence_labels: Sequence[str] = DEFAULT_IPU_SILENCE_LABELS,
) -> EnvelopeFeatureResult:
    """Extract the EEG-aligned acoustic envelope derivative and write it."""
    waveform, sampling_rate_hz = load_audio_waveform(audio_path)
    mask_metadata: dict[str, object] | None = None
    if ipu_path is not None:
        waveform, mask_metadata = zero_waveform_for_ipu_silences(
            waveform,
            audio_sampling_rate_hz=sampling_rate_hz,
            ipu_path=ipu_path,
            silence_labels=silence_labels,
        )
    result = extract_envelope_feature(
        waveform=waveform,
        audio_sampling_rate_hz=sampling_rate_hz,
        eeg_sampling_rate_hz=eeg_sampling_rate_hz,
        eeg_sample_count=eeg_sample_count,
        config=config,
    )
    _write_continuous_derivative(
        result,
        output_values_path,
        output_sidecar_path,
        feature_name=feature_name,
        source_subject=source_subject,
        source_role=source_role,
        extra_metadata=(
            {"silence_masking": {"applied": True, **mask_metadata}}
            if mask_metadata is not None
            else None
        ),
    )
    return result


def run_pitch_pipeline(
    audio_path: Path,
    eeg_sampling_rate_hz: float,
    eeg_sample_count: int,
    output_values_path: Path,
    output_sidecar_path: Path,
    config: PitchExtractionConfig | None = None,
    *,
    feature_name: str | None = None,
    source_subject: str | None = None,
    source_role: str | None = None,
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
    _write_continuous_derivative(
        result,
        output_values_path,
        output_sidecar_path,
        feature_name=feature_name,
        source_subject=source_subject,
        source_role=source_role,
    )
    return result


def run_vowel_formant_pipeline(
    audio_path: Path,
    alignment_path: Path,
    tier_name: str,
    output_tsv_path: Path,
    output_sidecar_path: Path,
    config: FormantEventExtractionConfig | None = None,
    speaker: str | None = None,
    feature_name: str | None = None,
    source_subject: str | None = None,
    source_role: str | None = None,
) -> FormantEventResult:
    """Extract vowel-centered F1/F2 event features from alignment intervals."""
    resolved_config = config or FormantEventExtractionConfig()
    waveform, sampling_rate_hz = load_audio_waveform(audio_path)
    vowel_intervals = load_vowel_intervals(
        alignment_path=alignment_path,
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
    _write_event_derivative(
        result,
        output_tsv_path,
        output_sidecar_path,
        feature_name=feature_name,
        source_subject=source_subject,
        source_role=source_role,
    )
    return result
