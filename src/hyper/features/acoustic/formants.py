"""Vowel-centered formant event extraction backed by local Parselmouth formant tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from voxatlas.phonology.articulatory_utils import load_phonology_resources, lookup_articulatory_features
from voxatlas.units.alignment_loader import load_textgrid

from .common import dataclass_to_dict, get_voxatlas_version


SECONDS_PER_MILLISECOND: float = 0.001
FRENCH_XSAMPA_VOWEL_LABELS: frozenset[str] = frozenset(
    {
        "a",
        "a~",
        "A/",
        "e",
        "E",
        "e~",
        "E~",
        "i",
        "o",
        "O",
        "O/",
        "O~",
        "u",
        "U",
        "U~/",
        "y",
        "2",
        "2~",
        "9",
        "9~",
        "@",
    }
)
FORMANT_EVENT_COLUMNS: tuple[str, ...] = (
    "onset_seconds",
    "duration_seconds",
    "vowel_label",
    "speaker",
    "f1_median_hz",
    "f2_median_hz",
    "source_interval_id",
    "extraction_status",
    "notes",
)


def _safe_float32(value: float | None) -> np.float32:
    if value is None or not np.isfinite(value):
        return np.float32(np.nan)
    return np.float32(value)


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
    if features is not None:
        return bool(float(features.get("vowel", 0.0)) == 1.0)

    # Some environments ship without populated VoxAtlas phonology tables.
    # Fall back to the XSAMPA-style French labels used in this dataset so
    # vowel-centered formant extraction can still proceed deterministically.
    return label in FRENCH_XSAMPA_VOWEL_LABELS


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


def load_vowel_intervals_from_palign_csv(
    csv_path: str | Path,
    tier_name: str,
    speaker: str | None = None,
    language: str | None = None,
    resource_root: str | None = None,
) -> list[VowelInterval]:
    """Load vowel intervals from a SPPAS palign CSV export."""
    intervals_df = pd.read_csv(
        csv_path,
        header=None,
        names=["tier", "start", "end", "label"],
        encoding="utf-8-sig",
    )
    intervals_df["tier"] = intervals_df["tier"].astype(str).str.strip().str.strip('"')
    intervals_df["label"] = intervals_df["label"].astype(str).str.strip().str.strip('"')

    tier_df = intervals_df.loc[intervals_df["tier"] == tier_name].copy()
    if tier_df.empty:
        raise ValueError(f"Alignment tier not found in CSV: {tier_name}")

    intervals: list[VowelInterval] = []
    for idx, row in tier_df.reset_index(drop=True).iterrows():
        label = str(row["label"]).strip()
        if not label:
            continue
        if not _is_vowel_label(label, language=language, resource_root=resource_root):
            continue

        intervals.append(
            VowelInterval(
                interval_id=str(idx),
                onset_seconds=float(row["start"]),
                offset_seconds=float(row["end"]),
                vowel_label=label,
                speaker=speaker,
            )
        )

    return intervals


def load_vowel_intervals(
    alignment_path: str | Path,
    tier_name: str,
    speaker: str | None = None,
    language: str | None = None,
    resource_root: str | None = None,
) -> list[VowelInterval]:
    """Load vowel intervals from either a TextGrid or palign CSV file."""
    suffix = Path(alignment_path).suffix.lower()
    if suffix == ".csv":
        return load_vowel_intervals_from_palign_csv(
            alignment_path,
            tier_name=tier_name,
            speaker=speaker,
            language=language,
            resource_root=resource_root,
        )

    return load_vowel_intervals_from_textgrid(
        str(alignment_path),
        tier_name=tier_name,
        speaker=speaker,
        language=language,
        resource_root=resource_root,
    )


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


def _compute_interval_formant_tracks(
    waveform: np.ndarray,
    audio_sampling_rate_hz: int,
    intervals: list[VowelInterval],
    config: FormantEventExtractionConfig,
) -> pd.DataFrame:
    """Compute local formant tracks for intervals already vetted as vowels."""
    try:
        import parselmouth
    except ImportError as exc:
        raise ImportError("Formant extraction requires Parselmouth in this environment.") from exc

    rows: list[dict[str, Any]] = []
    frame_length = float(config.frame_length_seconds)
    frame_step = float(config.frame_step_seconds)

    for interval in intervals:
        start_sample = max(0, int(round(interval.onset_seconds * audio_sampling_rate_hz)))
        end_sample = min(len(waveform), int(round(interval.offset_seconds * audio_sampling_rate_hz)))
        segment = np.asarray(waveform[start_sample:end_sample], dtype=np.float64)
        if segment.size == 0:
            continue

        sound = parselmouth.Sound(segment, sampling_frequency=float(audio_sampling_rate_hz))
        formant = sound.to_formant_burg(
            time_step=frame_step,
            max_number_of_formants=5,
            maximum_formant=float(config.max_formant_hz),
            window_length=frame_length,
            pre_emphasis_from=50.0,
        )

        duration = segment.size / float(audio_sampling_rate_hz)
        sample_times = np.arange(frame_length / 2.0, max(duration, frame_length / 2.0) + 1e-8, frame_step)
        if sample_times.size == 0:
            sample_times = np.array([min(duration / 2.0, frame_length / 2.0)], dtype=np.float32)

        for frame_id, local_time in enumerate(sample_times, start=1):
            global_time = interval.onset_seconds + float(local_time)
            values = []
            for formant_index in (1, 2, 3):
                value = formant.get_value_at_time(formant_index, float(local_time))
                values.append(_safe_float32(value if value and value > 0 else np.nan))
            f1, f2, f3 = values
            rows.append(
                {
                    "frame_id": frame_id,
                    "start": np.float32(max(interval.onset_seconds, global_time - frame_length / 2.0)),
                    "end": np.float32(min(interval.offset_seconds, global_time + frame_length / 2.0)),
                    "time": np.float32(global_time),
                    "phoneme_id": interval.interval_id,
                    "label": interval.vowel_label,
                    "ipa": interval.vowel_label,
                    "is_vowel": np.float32(1.0),
                    "F1": f1,
                    "F2": f2,
                    "F3": f3,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["frame_id", "start", "end", "time", "phoneme_id", "label", "ipa", "is_vowel", "F1", "F2", "F3"],
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
        tracks = _compute_interval_formant_tracks(
            waveform=np.asarray(waveform, dtype=np.float32),
            audio_sampling_rate_hz=int(audio_sampling_rate_hz),
            intervals=valid_intervals,
            config=resolved_config,
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

    event_table = pd.DataFrame.from_records(records, columns=FORMANT_EVENT_COLUMNS)
    metadata = FormantEventMetadata(
        feature_name="vowel_formants",
        audio_sampling_rate_hz=int(audio_sampling_rate_hz),
        extraction_parameters=dataclass_to_dict(resolved_config),
        voxatlas_version=get_voxatlas_version(),
        voxatlas_function="hyper.features.acoustic.formants._compute_interval_formant_tracks",
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
            "F1/F2 medians are computed within each vowel interval from local Parselmouth Burg formant tracks.",
        ],
    )
    return FormantEventResult(event_table=event_table, metadata=metadata)
