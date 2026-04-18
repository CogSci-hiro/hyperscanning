"""Vowel-centered formant event extraction backed by VoxAtlas formant utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from voxatlas.phonology.articulatory_utils import load_phonology_resources, lookup_articulatory_features
from voxatlas.phonology.formant_utils import compute_formant_tracks
from voxatlas.units.alignment_loader import load_textgrid

from .common import dataclass_to_dict, get_voxatlas_version


SECONDS_PER_MILLISECOND: float = 0.001


@dataclass(frozen=True, slots=True)
class VowelInterval:
    """Annotation interval describing one vowel token candidate."""

    interval_id: str
    onset_seconds: float
    offset_seconds: float
    vowel_label: str
    speaker: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Return the interval duration in seconds."""
        return self.offset_seconds - self.onset_seconds


@dataclass(frozen=True, slots=True)
class FormantEventExtractionConfig:
    """Configuration for vowel-centered F1/F2 extraction."""

    language: str | None = None
    resource_root: str | None = None
    frame_length_seconds: float = 0.025
    frame_step_seconds: float = 0.010
    lpc_order: int = 10
    max_formant_hz: float = 5500.0
    min_interval_duration_seconds: float = 0.030
    use_parselmouth: bool = True


@dataclass(frozen=True, slots=True)
class FormantEventMetadata:
    """Metadata sidecar for vowel formant event tables."""

    feature_name: str
    audio_sampling_rate_hz: int
    extraction_parameters: dict[str, Any]
    voxatlas_version: str | None
    voxatlas_function: str
    alignment_target: str
    units: dict[str, str]
    shape: tuple[int, int]
    notes: list[str]


@dataclass(frozen=True, slots=True)
class FormantEventResult:
    """Tabular vowel formant event result plus metadata."""

    event_table: pd.DataFrame
    metadata: FormantEventMetadata


def _is_vowel_label(
    label: str,
    language: str | None,
    resource_root: str | None,
) -> bool:
    resources = load_phonology_resources(language=language, resource_root=resource_root)
    _, features = lookup_articulatory_features(label, resources)
    return bool(features is not None and float(features.get("vowel", 0.0)) == 1.0)


def load_vowel_intervals_from_textgrid(
    textgrid_path: str,
    tier_name: str,
    speaker: str | None = None,
    language: str | None = None,
    resource_root: str | None = None,
) -> list[VowelInterval]:
    """Load vowel intervals from a TextGrid tier using VoxAtlas phonology tables."""
    tiers = load_textgrid(textgrid_path)
    if tier_name not in tiers:
        raise ValueError(f"TextGrid tier not found: {tier_name}")

    intervals_df = tiers[tier_name]
    intervals: list[VowelInterval] = []
    for _, row in intervals_df.iterrows():
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        if not _is_vowel_label(label, language=language, resource_root=resource_root):
            continue

        intervals.append(
            VowelInterval(
                interval_id=str(row.get("id")),
                onset_seconds=float(row["start"]),
                offset_seconds=float(row["end"]),
                vowel_label=label,
                speaker=speaker,
            )
        )

    return intervals


def _intervals_to_phoneme_table(intervals: list[VowelInterval]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": interval.interval_id,
                "start": interval.onset_seconds,
                "end": interval.offset_seconds,
                "label": interval.vowel_label,
                "speaker": interval.speaker,
            }
            for interval in intervals
        ]
    )


def extract_vowel_formant_events(
    waveform: np.ndarray,
    audio_sampling_rate_hz: int,
    vowel_intervals: list[VowelInterval],
    config: FormantEventExtractionConfig | None = None,
) -> FormantEventResult:
    """Compute F1/F2 vowel medians for a pluggable set of vowel intervals."""
    resolved_config = config or FormantEventExtractionConfig()
    records: list[dict[str, Any]] = []

    valid_intervals = [
        interval
        for interval in vowel_intervals
        if interval.duration_seconds >= resolved_config.min_interval_duration_seconds
    ]
    if valid_intervals:
        phoneme_table = _intervals_to_phoneme_table(valid_intervals)
        tracks = compute_formant_tracks(
            signal=np.asarray(waveform, dtype=np.float32),
            sr=int(audio_sampling_rate_hz),
            phonemes=phoneme_table,
            language=resolved_config.language,
            resource_root=resolved_config.resource_root,
            frame_length=resolved_config.frame_length_seconds,
            frame_step=resolved_config.frame_step_seconds,
            lpc_order=resolved_config.lpc_order,
            max_formant=resolved_config.max_formant_hz,
            use_parselmouth=resolved_config.use_parselmouth,
        )
    else:
        tracks = pd.DataFrame()

    for interval in vowel_intervals:
        extraction_status = "ok"
        notes: list[str] = []
        f1_median_hz = np.nan
        f2_median_hz = np.nan

        if interval.duration_seconds < resolved_config.min_interval_duration_seconds:
            extraction_status = "too_short"
            notes.append("Interval shorter than minimum duration threshold.")
        else:
            interval_tracks = tracks.loc[tracks["phoneme_id"].astype(str) == interval.interval_id].copy()
            f1_values = pd.to_numeric(interval_tracks.get("F1", pd.Series(dtype=float)), errors="coerce")
            f2_values = pd.to_numeric(interval_tracks.get("F2", pd.Series(dtype=float)), errors="coerce")
            valid_tracks = interval_tracks.loc[np.isfinite(f1_values) & np.isfinite(f2_values)]

            if valid_tracks.empty:
                extraction_status = "failed"
                notes.append("No stable F1/F2 estimates available for this vowel interval.")
            else:
                f1_median_hz = float(valid_tracks["F1"].median())
                f2_median_hz = float(valid_tracks["F2"].median())

        records.append(
            {
                "onset_seconds": interval.onset_seconds,
                "duration_seconds": interval.duration_seconds,
                "vowel_label": interval.vowel_label,
                "speaker": interval.speaker or "",
                "f1_median_hz": f1_median_hz,
                "f2_median_hz": f2_median_hz,
                "source_interval_id": interval.interval_id,
                "extraction_status": extraction_status,
                "notes": " ".join(notes),
            }
        )

    event_table = pd.DataFrame.from_records(records)
    metadata = FormantEventMetadata(
        feature_name="vowel_formants",
        audio_sampling_rate_hz=int(audio_sampling_rate_hz),
        extraction_parameters=dataclass_to_dict(resolved_config),
        voxatlas_version=get_voxatlas_version(),
        voxatlas_function="voxatlas.phonology.formant_utils.compute_formant_tracks",
        alignment_target="vowel_events",
        units={
            "onset_seconds": "seconds",
            "duration_seconds": "seconds",
            "f1_median_hz": "Hz",
            "f2_median_hz": "Hz",
        },
        shape=tuple(event_table.shape),
        notes=[
            "Intervals are supplied externally; this extractor does not perform forced alignment.",
            "F1/F2 medians are computed within each vowel interval from VoxAtlas formant tracks.",
        ],
    )
    return FormantEventResult(event_table=event_table, metadata=metadata)
