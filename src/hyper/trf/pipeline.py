"""Run-wise TRF pipeline with blockwise segmentation and nested grouped CV."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import GroupKFold

from hyper.config import ProjectConfig
from hyper.paths import ProjectPaths
from hyper.trf.config import DEFAULT_TRF_TASK, TrfConfig

LOGGER = logging.getLogger(__name__)

TRF_RESULTS_DIRNAME: str = "trf"
TIMING_JSON_SUFFIX: str = "_raw_ds_timing.json"
FILTERED_RAW_SUFFIX: str = "_raw_filt.fif"
DEFAULT_JSON_INDENT: int = 2


@dataclass(frozen=True, slots=True)
class PredictorSpec:
    """Storage and loading rules for one TRF predictor family."""

    storage_kind: str
    descriptor: str
    dirname: str
    role: str
    path_root: str = "out"
    onset_column: str | None = None
    value_columns: tuple[str, ...] = ()
    aligned_to_conversation: bool = True


PREDICTOR_ALIASES: dict[str, tuple[str, ...]] = {
    "self_f1_f2": ("self_f1", "self_f2"),
    "other_f1_f2": ("other_f1", "other_f2"),
}


PREDICTOR_SPECS: dict[str, PredictorSpec] = {
    "speech_envelope": PredictorSpec("continuous", "self_envelope", "envelope", "self"),
    "envelope": PredictorSpec("continuous", "self_envelope", "envelope", "self"),
    "self_speech_envelope": PredictorSpec("continuous", "self_envelope", "envelope", "self"),
    "other_speech_envelope": PredictorSpec("continuous", "other_envelope", "envelope", "other"),
    "self_envelope": PredictorSpec("continuous", "self_envelope", "envelope", "self"),
    "other_envelope": PredictorSpec("continuous", "other_envelope", "envelope", "other"),
    "f0": PredictorSpec("continuous", "self_f0", "f0", "self"),
    "self_f0": PredictorSpec("continuous", "self_f0", "f0", "self"),
    "other_f0": PredictorSpec("continuous", "other_f0", "f0", "other"),
    "self_f1": PredictorSpec(
        "event",
        "self_vowels",
        "vowels",
        "self",
        onset_column="onset_seconds",
        value_columns=("f1_median_hz",),
    ),
    "other_f1": PredictorSpec(
        "event",
        "other_vowels",
        "vowels",
        "other",
        onset_column="onset_seconds",
        value_columns=("f1_median_hz",),
    ),
    "self_f2": PredictorSpec(
        "event",
        "self_vowels",
        "vowels",
        "self",
        onset_column="onset_seconds",
        value_columns=("f2_median_hz",),
    ),
    "other_f2": PredictorSpec(
        "event",
        "other_vowels",
        "vowels",
        "other",
        onset_column="onset_seconds",
        value_columns=("f2_median_hz",),
    ),
    "self_phoneme_onsets": PredictorSpec(
        "event",
        "self_phonemes",
        "phonemes",
        "self",
        onset_column="onset_seconds",
    ),
    "other_phoneme_onsets": PredictorSpec(
        "event",
        "other_phonemes",
        "phonemes",
        "other",
        onset_column="onset_seconds",
    ),
    "self_syllable_onsets": PredictorSpec(
        "event",
        "self_syllables",
        "syllables",
        "self",
        onset_column="onset_seconds",
    ),
    "other_syllable_onsets": PredictorSpec(
        "event",
        "other_syllables",
        "syllables",
        "other",
        onset_column="onset_seconds",
    ),
    "self_token_onsets": PredictorSpec(
        "event",
        "self_tokens",
        "tokens",
        "self",
        onset_column="onset_seconds",
    ),
    "other_token_onsets": PredictorSpec(
        "event",
        "other_tokens",
        "tokens",
        "other",
        onset_column="onset_seconds",
    ),
    "self_surprisal": PredictorSpec(
        "event",
        "lmSurprisal",
        "lm_surprisal",
        "self",
        path_root="lm",
        onset_column="onset",
        value_columns=("surprisal",),
        aligned_to_conversation=False,
    ),
    "other_surprisal": PredictorSpec(
        "event",
        "lmSurprisal",
        "lm_surprisal",
        "other",
        path_root="lm",
        onset_column="onset",
        value_columns=("surprisal",),
        aligned_to_conversation=False,
    ),
    "self_entropy": PredictorSpec(
        "event",
        "lmShannonEntropy",
        "lm_shannon_entropy",
        "self",
        path_root="lm",
        onset_column="onset",
        value_columns=("entropy",),
        aligned_to_conversation=False,
    ),
    "other_entropy": PredictorSpec(
        "event",
        "lmShannonEntropy",
        "lm_shannon_entropy",
        "other",
        path_root="lm",
        onset_column="onset",
        value_columns=("entropy",),
        aligned_to_conversation=False,
    ),
}


@dataclass(frozen=True, slots=True)
class TrfRunInput:
    """Continuous run-level TRF input after cropping/alignment."""

    subject_id: str
    task: str
    run_id: str
    predictor_names: tuple[str, ...]
    predictor_values: np.ndarray
    target_values: np.ndarray
    sampling_rate_hz: float
    conversation_start_seconds: float
    cropped_start_sample: int
    cropped_stop_sample: int
    source_duration_seconds: float


@dataclass(frozen=True, slots=True)
class TrfSegment:
    """Metadata-bearing segment derived from a cropped run."""

    subject_id: str
    task: str
    run_id: str
    segment_id: str
    predictor_names: tuple[str, ...]
    predictor_values: np.ndarray
    target_values: np.ndarray
    sampling_rate_hz: float
    start_sample: int
    stop_sample: int
    start_seconds: float
    stop_seconds: float


@dataclass(frozen=True, slots=True)
class TrfLaggedSegmentDesign:
    """Lag-safe per-segment design matrix."""

    segment: TrfSegment
    design_matrix: np.ndarray
    target_matrix: np.ndarray
    lag_samples: np.ndarray


@dataclass(frozen=True, slots=True)
class TrfFoldResult:
    """Outer-fold TRF result summary."""

    outer_fold_index: int
    score: float
    selected_alpha: float
    actual_outer_splits: int
    actual_inner_splits: int
    train_segment_ids: tuple[str, ...]
    test_segment_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrfSubjectResult:
    """Persistable TRF result bundle for one subject."""

    subject_id: str
    task: str
    predictor_names: tuple[str, ...]
    lag_seconds: np.ndarray
    fold_results: tuple[TrfFoldResult, ...]
    coefficient_paths: tuple[Path, ...]
    skipped_runs: tuple[str, ...]
    skipped_segments: tuple[str, ...]
    out_dir: Path


def _predictor_spec(predictor_name: str) -> PredictorSpec:
    """Return the storage spec for one configured predictor."""
    try:
        return PREDICTOR_SPECS[predictor_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported TRF predictor: {predictor_name!r}") from exc


def _expand_predictor_names(predictor_names: Sequence[str]) -> tuple[str, ...]:
    """Expand configured predictor aliases into concrete TRF design columns."""
    expanded: list[str] = []
    for predictor_name in predictor_names:
        resolved_names = PREDICTOR_ALIASES.get(str(predictor_name), (str(predictor_name),))
        for resolved_name in resolved_names:
            _predictor_spec(resolved_name)
            expanded.append(resolved_name)
    return tuple(expanded)


def _run_stem(subject_id: str, task: str, run_id: str) -> str:
    """Return the canonical run stem used across derived paths."""
    return f"{subject_id}_task-{task}_run-{run_id}"


def _filtered_raw_path(paths: ProjectPaths, *, subject_id: str, task: str, run_id: str) -> Path:
    """Return the filtered raw FIF path for one run."""
    return paths.out_dir / "eeg" / "filtered" / f"{_run_stem(subject_id, task, run_id)}{FILTERED_RAW_SUFFIX}"


def _timing_sidecar_path(paths: ProjectPaths, *, subject_id: str, task: str, run_id: str) -> Path:
    """Return the conversation-timing JSON sidecar path for one run."""
    return paths.out_dir / "eeg" / "downsampled" / f"{_run_stem(subject_id, task, run_id)}{TIMING_JSON_SUFFIX}"


def _partner_subject_id(subject_id: str) -> str:
    """Return the paired partner subject id using the project's odd/even dyad rule."""
    subject_num = int(str(subject_id).removeprefix("sub-"))
    partner_num = subject_num + 1 if (subject_num % 2 == 1) else subject_num - 1
    if partner_num <= 0:
        raise ValueError(f"Cannot infer partner subject for {subject_id!r}.")
    return f"sub-{partner_num:03d}"


def _predictor_path(
    paths: ProjectPaths,
    *,
    subject_id: str,
    task: str,
    run_id: str,
    predictor_name: str,
) -> Path:
    """Return the stored predictor path for one run."""
    spec = _predictor_spec(predictor_name)
    root = paths.out_dir if spec.path_root == "out" else paths.lm_feature_root
    if spec.storage_kind == "continuous":
        run_stem = _run_stem(subject_id, task, run_id)
        return root / "features" / "continuous" / spec.dirname / f"{run_stem}_desc-{spec.descriptor}_feature.npy"
    if spec.storage_kind == "event":
        storage_subject = subject_id if spec.role == "self" else _partner_subject_id(subject_id)
        run_stem = _run_stem(storage_subject, task, run_id)
        canonical_path = root / "features" / "events" / spec.dirname / f"{run_stem}_desc-{spec.descriptor}_features.tsv"
        if spec.path_root != "lm":
            return canonical_path
        if canonical_path.exists():
            return canonical_path
        session_glob = (
            root
            / "features"
            / "events"
            / spec.dirname
            / f"{storage_subject}_ses-*_task-{task}_run-{run_id}_desc-{spec.descriptor}_features.tsv"
        )
        matches = sorted(session_glob.parent.glob(session_glob.name))
        if len(matches) > 0:
            return matches[0]
        return canonical_path
    raise ValueError(f"Unsupported predictor storage kind: {spec.storage_kind!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a stable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=DEFAULT_JSON_INDENT, sort_keys=True) + "\n", encoding="utf-8")


def _is_explicitly_missing(cfg: Mapping[str, Any], *, subject_id: str, task: str, run_id: str) -> bool:
    """Return whether a run is marked missing in config."""
    missing_runs_cfg = cfg.get("subjects", {}).get("missing_runs", {})
    task_runs = missing_runs_cfg.get(subject_id, {}).get(task, [])
    return str(run_id) in {str(value) for value in task_runs}


def compute_lag_samples(*, tmin_seconds: float, tmax_seconds: float, sampling_rate_hz: float) -> np.ndarray:
    """Return sample lags matching SpyEEG's internal lag ordering."""
    sample_min = int(np.ceil(tmin_seconds * sampling_rate_hz))
    sample_max = int(np.ceil(tmax_seconds * sampling_rate_hz))
    return np.arange(sample_min, sample_max, dtype=int)[::-1]


def min_samples_for_lag_support(lag_samples: np.ndarray) -> int:
    """Return the smallest segment length that yields at least one valid lagged row."""
    if lag_samples.size == 0:
        return 1
    max_future = int(max(0, np.max(lag_samples)))
    max_past = int(max(0, abs(np.min(lag_samples))))
    return max_future + max_past + 1


def _resample_array(values: np.ndarray, *, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    """Resample a time-by-feature matrix onto a new sampling rate."""
    if abs(source_sfreq - target_sfreq) <= 1e-9:
        return values.astype(np.float32, copy=False)
    sample_count = max(1, int(round(values.shape[0] * target_sfreq / source_sfreq)))
    resampled = signal.resample(values, sample_count, axis=0)
    return np.asarray(resampled, dtype=np.float32)


def _trim_to_shared_length(predictor_values: np.ndarray, target_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim predictor and target arrays to the same number of samples."""
    shared_length = min(predictor_values.shape[0], target_values.shape[0])
    return predictor_values[:shared_length], target_values[:shared_length]


def _load_conversation_start_seconds(timing_json_path: Path) -> float:
    """Load conversation start time from an upstream timing artifact."""
    payload = json.loads(timing_json_path.read_text(encoding="utf-8"))
    if "conversation_start_seconds" not in payload:
        raise ValueError(f"Timing sidecar missing 'conversation_start_seconds': {timing_json_path}")
    return float(payload["conversation_start_seconds"])


def _stack_predictors(predictor_paths: Sequence[Path]) -> np.ndarray:
    """Load and column-stack stored continuous predictors."""
    columns: list[np.ndarray] = []
    for path in predictor_paths:
        values = np.load(path)
        column = np.asarray(values, dtype=np.float32)
        if column.ndim == 1:
            column = column[:, np.newaxis]
        columns.append(column)
    if len(columns) == 0:
        raise ValueError("No predictor arrays were provided for TRF loading.")
    return np.concatenate(columns, axis=1)


def _load_event_predictor(
    event_path: Path,
    *,
    onset_column: str | None,
    value_columns: Sequence[str],
    target_sfreq: float,
    target_length: int,
    onset_offset_seconds: float,
) -> np.ndarray:
    """Rasterize an event TSV into one or more single-sample event predictors."""
    table = pd.read_csv(event_path, sep="\t")
    resolved_onset_column = onset_column
    if resolved_onset_column is None:
        if "onset" in table.columns:
            resolved_onset_column = "onset"
        elif "onset_seconds" in table.columns:
            resolved_onset_column = "onset_seconds"
    if resolved_onset_column is None or resolved_onset_column not in table.columns:
        raise ValueError(
            f"Event predictor table missing required onset column {onset_column!r}: {event_path}"
        )

    onsets = pd.to_numeric(table[resolved_onset_column], errors="coerce")
    relative_onsets = onsets.to_numpy(dtype=float) - float(onset_offset_seconds)
    sample_indices = np.rint(relative_onsets * float(target_sfreq)).astype(int)
    within_bounds = (sample_indices >= 0) & (sample_indices < int(target_length))

    if len(value_columns) == 0:
        raster = np.zeros(int(target_length), dtype=np.float32)
        valid = np.isfinite(relative_onsets) & within_bounds
        if np.any(valid):
            np.add.at(raster, sample_indices[valid], 1.0)
        return raster[:, np.newaxis]

    rasters: list[np.ndarray] = []
    for value_column in value_columns:
        if value_column not in table.columns:
            raise ValueError(f"Event predictor table missing required {value_column!r} column: {event_path}")
        values = pd.to_numeric(table[value_column], errors="coerce")
        valid = onsets.notna() & values.notna() & within_bounds
        raster = np.zeros(int(target_length), dtype=np.float32)
        if bool(valid.any()):
            np.add.at(
                raster,
                sample_indices[valid.to_numpy(dtype=bool)],
                values.loc[valid].to_numpy(dtype=np.float32),
            )
        rasters.append(raster[:, np.newaxis])
    return np.concatenate(rasters, axis=1)


def _load_predictor_matrix(
    *,
    predictor_names: Sequence[str],
    predictor_paths: Sequence[Path],
    paths: ProjectPaths,
    subject_id: str,
    task: str,
    run_id: str,
    source_sfreq: float,
    target_sfreq: float,
    target_length: int,
    conversation_start_seconds: float,
) -> np.ndarray:
    """Load and align heterogeneous predictor families onto a shared TRF time base."""
    del paths, subject_id, task, run_id
    columns: list[np.ndarray] = []
    for predictor_name, predictor_path in zip(predictor_names, predictor_paths, strict=True):
        spec = _predictor_spec(predictor_name)
        if spec.storage_kind == "continuous":
            values = np.load(predictor_path)
            column = np.asarray(values, dtype=np.float32)
            if column.ndim == 1:
                column = column[:, np.newaxis]
            column = _resample_array(column, source_sfreq=source_sfreq, target_sfreq=target_sfreq)
        elif spec.storage_kind == "event":
            column = _load_event_predictor(
                predictor_path,
                onset_column=spec.onset_column,
                value_columns=spec.value_columns,
                target_sfreq=target_sfreq,
                target_length=target_length,
                onset_offset_seconds=0.0 if spec.aligned_to_conversation else conversation_start_seconds,
            )
        else:
            raise ValueError(f"Unsupported predictor storage kind: {spec.storage_kind!r}")
        columns.append(column.astype(np.float32, copy=False))
    if len(columns) == 0:
        raise ValueError("No predictor arrays were provided for TRF loading.")
    shared_length = min(target_length, *(column.shape[0] for column in columns))
    if shared_length <= 0:
        raise ValueError("Predictor loading produced an empty time axis.")
    return np.concatenate([column[:shared_length] for column in columns], axis=1)


def crop_target_to_conversation_window(
    target_values: np.ndarray,
    *,
    sampling_rate_hz: float,
    conversation_start_seconds: float,
    duration_seconds: float,
) -> tuple[np.ndarray, int, int]:
    """Crop a run-aligned EEG target array to the conversation window.

    Notes
    -----
    Acoustic predictors are stored conversation-aligned upstream, so only the
    EEG target should be cropped by the conversation-start offset here.
    If the run ends before the requested duration, the function crops to the
    available tail and leaves validity checks to the segmentation step.
    """
    start_sample = max(0, int(round(conversation_start_seconds * sampling_rate_hz)))
    requested_stop = start_sample + int(round(duration_seconds * sampling_rate_hz))
    available_stop = min(target_values.shape[0], requested_stop)
    if start_sample >= available_stop:
        return target_values[:0], start_sample, available_stop
    return target_values[start_sample:available_stop], start_sample, available_stop


def load_trf_run_inputs(
    *,
    cfg: ProjectConfig,
    paths: ProjectPaths,
    subject_id: str,
    task: str = DEFAULT_TRF_TASK,
    run_ids: Sequence[str] | None = None,
) -> tuple[list[TrfRunInput], list[str]]:
    """Load all available continuous TRF runs for one subject."""
    trf_cfg = TrfConfig.from_mapping(cfg.raw.get("trf"))
    resolved_predictor_names = _expand_predictor_names(trf_cfg.predictors)
    requested_run_ids = [str(run_id) for run_id in (run_ids or cfg.raw.get("runs", {}).get("include", {}).get(task, []))]
    loaded_runs: list[TrfRunInput] = []
    skipped_runs: list[str] = []

    for run_id in requested_run_ids:
        if _is_explicitly_missing(cfg.raw, subject_id=subject_id, task=task, run_id=run_id):
            LOGGER.info("Skipping expected-missing TRF run %s %s run-%s.", subject_id, task, run_id)
            skipped_runs.append(run_id)
            continue

        raw_path = _filtered_raw_path(paths, subject_id=subject_id, task=task, run_id=run_id)
        timing_path = _timing_sidecar_path(paths, subject_id=subject_id, task=task, run_id=run_id)
        predictor_paths = [
            _predictor_path(paths, subject_id=subject_id, task=task, run_id=run_id, predictor_name=name)
            for name in resolved_predictor_names
        ]

        missing_paths = [path for path in [raw_path, timing_path, *predictor_paths] if not path.exists()]
        if missing_paths:
            LOGGER.warning(
                "Skipping TRF run %s %s run-%s because required inputs are missing: %s",
                subject_id,
                task,
                run_id,
                ", ".join(str(path) for path in missing_paths),
            )
            skipped_runs.append(run_id)
            continue

        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
        source_sfreq = float(raw.info["sfreq"])
        target_values = np.asarray(raw.get_data().T, dtype=np.float32)
        conversation_start_seconds = _load_conversation_start_seconds(timing_path)
        cropped_target, start_sample, stop_sample = crop_target_to_conversation_window(
            target_values,
            sampling_rate_hz=source_sfreq,
            conversation_start_seconds=conversation_start_seconds,
            duration_seconds=trf_cfg.conversation.duration_seconds,
        )
        if cropped_target.shape[0] == 0:
            LOGGER.warning(
                "Skipping TRF run %s %s run-%s because the conversation window is empty after cropping.",
                subject_id,
                task,
                run_id,
            )
            skipped_runs.append(run_id)
            continue

        aligned_target = _resample_array(
            cropped_target,
            source_sfreq=source_sfreq,
            target_sfreq=trf_cfg.target_sfreq,
        )
        aligned_predictors = _load_predictor_matrix(
            predictor_names=resolved_predictor_names,
            predictor_paths=predictor_paths,
            paths=paths,
            subject_id=subject_id,
            task=task,
            run_id=run_id,
            source_sfreq=source_sfreq,
            target_sfreq=trf_cfg.target_sfreq,
            target_length=aligned_target.shape[0],
            conversation_start_seconds=conversation_start_seconds,
        )
        aligned_predictors, aligned_target = _trim_to_shared_length(aligned_predictors, aligned_target)

        loaded_runs.append(
            TrfRunInput(
                subject_id=subject_id,
                task=task,
                run_id=str(run_id),
                predictor_names=resolved_predictor_names,
                predictor_values=aligned_predictors,
                target_values=aligned_target,
                sampling_rate_hz=float(trf_cfg.target_sfreq),
                conversation_start_seconds=conversation_start_seconds,
                cropped_start_sample=int(start_sample),
                cropped_stop_sample=int(stop_sample),
                source_duration_seconds=float(target_values.shape[0] / source_sfreq),
            )
        )

    return loaded_runs, skipped_runs


def split_run_into_segments(run_input: TrfRunInput, config: TrfConfig) -> tuple[list[TrfSegment], list[str]]:
    """Split one cropped run into contiguous blockwise TRF segments."""
    n_samples = run_input.predictor_values.shape[0]
    n_blocks = config.segmentation.n_blocks_per_run
    lag_samples = compute_lag_samples(
        tmin_seconds=config.lags.tmin_seconds,
        tmax_seconds=config.lags.tmax_seconds,
        sampling_rate_hz=run_input.sampling_rate_hz,
    )
    min_design_samples = min_samples_for_lag_support(lag_samples)
    min_block_samples = max(
        int(np.ceil(config.segmentation.min_block_duration_seconds * run_input.sampling_rate_hz)),
        min_design_samples,
    )

    if config.segmentation.drop_remainder:
        base_size = n_samples // n_blocks
        if base_size <= 0:
            return [], [f"{run_input.run_id}:not_enough_samples_for_requested_blocks"]
        block_sizes = [base_size] * n_blocks
    else:
        base_size = n_samples // n_blocks
        remainder = n_samples % n_blocks
        block_sizes = [base_size + (1 if index < remainder else 0) for index in range(n_blocks)]

    segments: list[TrfSegment] = []
    skipped_segments: list[str] = []
    cursor = 0

    for block_index, block_size in enumerate(block_sizes):
        start = cursor
        stop = cursor + block_size
        cursor = stop
        segment_id = f"{run_input.subject_id}_task-{run_input.task}_run-{run_input.run_id}_segment-{block_index + 1}"
        if block_size < min_block_samples:
            LOGGER.warning(
                "Skipping TRF segment %s because its duration is too short (%d samples < %d).",
                segment_id,
                block_size,
                min_block_samples,
            )
            skipped_segments.append(segment_id)
            continue
        segment_predictors = run_input.predictor_values[start:stop]
        segment_target = run_input.target_values[start:stop]
        segments.append(
            TrfSegment(
                subject_id=run_input.subject_id,
                task=run_input.task,
                run_id=run_input.run_id,
                segment_id=segment_id,
                predictor_names=run_input.predictor_names,
                predictor_values=segment_predictors,
                target_values=segment_target,
                sampling_rate_hz=run_input.sampling_rate_hz,
                start_sample=int(start),
                stop_sample=int(stop),
                start_seconds=float(start / run_input.sampling_rate_hz),
                stop_seconds=float(stop / run_input.sampling_rate_hz),
            )
        )

    return segments, skipped_segments


def build_lagged_segment_design(segment: TrfSegment, lag_samples: np.ndarray) -> TrfLaggedSegmentDesign:
    """Build one lag-safe design matrix without crossing segment boundaries."""
    if lag_samples.ndim != 1:
        raise ValueError("lag_samples must be one-dimensional.")

    max_future = int(max(0, np.max(lag_samples))) if lag_samples.size else 0
    max_past = int(max(0, abs(np.min(lag_samples)))) if lag_samples.size else 0
    valid_time_indices = np.arange(max_future, segment.predictor_values.shape[0] - max_past, dtype=int)
    if valid_time_indices.size == 0:
        raise ValueError(f"Segment {segment.segment_id} is too short for the requested lag window.")

    lagged_columns = [
        segment.predictor_values[valid_time_indices - lag]
        for lag in lag_samples
    ]
    design_matrix = np.concatenate(lagged_columns, axis=1).astype(np.float32, copy=False)
    target_matrix = segment.target_values[valid_time_indices].astype(np.float32, copy=False)

    return TrfLaggedSegmentDesign(
        segment=segment,
        design_matrix=design_matrix,
        target_matrix=target_matrix,
        lag_samples=lag_samples,
    )


def _segment_group_value(segment_design: TrfLaggedSegmentDesign, group_by: str) -> str:
    """Return the group label requested by config."""
    if group_by == "run_id":
        return str(segment_design.segment.run_id)
    if group_by == "segment_id":
        return str(segment_design.segment.segment_id)
    raise ValueError(f"Unsupported TRF grouping key: {group_by!r}")


def prepare_group_kfold(
    segment_designs: Sequence[TrfLaggedSegmentDesign],
    *,
    requested_splits: int,
    group_by: str,
    context: str,
) -> tuple[GroupKFold, np.ndarray, int]:
    """Prepare a grouped K-fold splitter, downgrading splits when needed."""
    groups = np.asarray([_segment_group_value(design, group_by) for design in segment_designs], dtype=object)
    unique_group_count = np.unique(groups).size
    if unique_group_count < 2:
        raise ValueError(
            f"TRF {context} requires at least 2 unique groups for group_by={group_by!r}, got {unique_group_count}."
        )
    actual_splits = min(int(requested_splits), int(unique_group_count))
    if actual_splits < requested_splits:
        LOGGER.warning(
            "Downgrading TRF %s n_splits from %d to %d because only %d groups are available.",
            context,
            requested_splits,
            actual_splits,
            unique_group_count,
        )
    return GroupKFold(n_splits=actual_splits), groups, actual_splits


def _concat_designs(segment_designs: Sequence[TrfLaggedSegmentDesign]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate lagged segment designs along the sample axis."""
    x = np.concatenate([design.design_matrix for design in segment_designs], axis=0)
    y = np.concatenate([design.target_matrix for design in segment_designs], axis=0)
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


def _standardize_train_and_apply(
    train_values: np.ndarray,
    test_values: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score data using train-set statistics only."""
    if not enabled:
        return train_values, test_values
    mean = np.mean(train_values, axis=0, keepdims=True)
    std = np.std(train_values, axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return (train_values - mean) / std, (test_values - mean) / std


def _safe_mean_channel_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean channel-wise Pearson correlation."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch for Pearson scoring: {y_true.shape} vs {y_pred.shape}")
    scores: list[float] = []
    for channel_index in range(y_true.shape[1]):
        true_channel = y_true[:, channel_index]
        pred_channel = y_pred[:, channel_index]
        if np.std(true_channel) <= 0.0 or np.std(pred_channel) <= 0.0:
            continue
        score = float(np.corrcoef(true_channel, pred_channel)[0, 1])
        if np.isfinite(score):
            scores.append(score)
    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))


def _import_spyeeg_trf_estimator() -> type[Any]:
    """Import SpyEEG lazily so unit tests can exercise helpers without it."""
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    from spyeeg.models.TRF import TRFEstimator

    return TRFEstimator


def _fit_one_alpha(
    train_designs: Sequence[TrfLaggedSegmentDesign],
    test_designs: Sequence[TrfLaggedSegmentDesign],
    *,
    alpha: float,
    config: TrfConfig,
    lag_samples: np.ndarray,
) -> tuple[float, Any]:
    """Fit and score one alpha value."""
    x_train, y_train = _concat_designs(train_designs)
    x_test, y_test = _concat_designs(test_designs)
    x_train, x_test = _standardize_train_and_apply(x_train, x_test, enabled=config.model.standardize_x)
    y_train, y_test = _standardize_train_and_apply(y_train, y_test, enabled=config.model.standardize_y)

    trf_estimator_cls = _import_spyeeg_trf_estimator()
    model = trf_estimator_cls(
        tmin=config.lags.tmin_seconds,
        tmax=config.lags.tmax_seconds,
        srate=config.target_sfreq,
        alpha=[float(alpha)],
        fit_intercept=config.model.fit_intercept,
        mtype="forward",
    )
    model.fit(x_train, y_train, lagged=True, drop=False, feat_names=train_designs[0].segment.predictor_names)
    predictions = np.asarray(model.predict(x_test), dtype=np.float32)
    if predictions.ndim != 3 or predictions.shape[-1] != 1:
        raise ValueError(f"Unexpected SpyEEG prediction shape: {predictions.shape}")
    score = _safe_mean_channel_pearsonr(y_test, predictions[..., 0])
    return score, model


def fit_nested_trf(
    segment_designs: Sequence[TrfLaggedSegmentDesign],
    *,
    config: TrfConfig,
    coefficient_path: Path | None = None,
) -> tuple[list[TrfFoldResult], list[Path]]:
    """Fit TRF with nested grouped CV on segment-level metadata units."""
    if len(segment_designs) == 0:
        raise ValueError("TRF nested CV requires at least one segment.")

    lag_samples = segment_designs[0].lag_samples
    alpha_values = config.alpha_values()
    outer_splitter, outer_groups, actual_outer_splits = prepare_group_kfold(
        segment_designs,
        requested_splits=config.cv.outer.n_splits,
        group_by=config.cv.outer.group_by,
        context="outer CV",
    )

    segment_indices = np.arange(len(segment_designs))
    fold_results: list[TrfFoldResult] = []
    coefficient_paths: list[Path] = []
    coefficient_payload: dict[str, np.ndarray] = {}

    for outer_fold_index, (train_index, test_index) in enumerate(
        outer_splitter.split(segment_indices, groups=outer_groups),
        start=1,
    ):
        train_designs = [segment_designs[index] for index in train_index]
        test_designs = [segment_designs[index] for index in test_index]
        inner_splitter, inner_groups, actual_inner_splits = prepare_group_kfold(
            train_designs,
            requested_splits=config.cv.inner.n_splits,
            group_by=config.cv.inner.group_by,
            context=f"inner CV (outer fold {outer_fold_index})",
        )

        inner_segment_indices = np.arange(len(train_designs))
        alpha_scores: list[float] = []

        for alpha in alpha_values:
            fold_scores: list[float] = []
            for inner_train_index, inner_valid_index in inner_splitter.split(inner_segment_indices, groups=inner_groups):
                inner_train_designs = [train_designs[index] for index in inner_train_index]
                inner_valid_designs = [train_designs[index] for index in inner_valid_index]
                fold_score, _ = _fit_one_alpha(
                    inner_train_designs,
                    inner_valid_designs,
                    alpha=float(alpha),
                    config=config,
                    lag_samples=lag_samples,
                )
                fold_scores.append(fold_score)
            alpha_scores.append(float(np.nanmean(fold_scores)))

        best_alpha_index = int(np.nanargmax(np.asarray(alpha_scores, dtype=float)))
        selected_alpha = float(alpha_values[best_alpha_index])
        outer_score, outer_model = _fit_one_alpha(
            train_designs,
            test_designs,
            alpha=selected_alpha,
            config=config,
            lag_samples=lag_samples,
        )
        if coefficient_path is not None:
            coefficient_payload[f"outer_fold_{outer_fold_index}"] = np.asarray(outer_model.get_coef(), dtype=np.float32)

        fold_results.append(
            TrfFoldResult(
                outer_fold_index=outer_fold_index,
                score=float(outer_score),
                selected_alpha=selected_alpha,
                actual_outer_splits=actual_outer_splits,
                actual_inner_splits=actual_inner_splits,
                train_segment_ids=tuple(segment_designs[index].segment.segment_id for index in train_index),
                test_segment_ids=tuple(segment_designs[index].segment.segment_id for index in test_index),
            )
        )

    if coefficient_path is not None:
        coefficient_path.parent.mkdir(parents=True, exist_ok=True)
        coefficient_payload["lag_seconds"] = (lag_samples[::-1] / config.target_sfreq).astype(np.float32)
        np.savez_compressed(coefficient_path, **coefficient_payload)
        coefficient_paths.append(coefficient_path)

    return fold_results, coefficient_paths


def _default_output_dir(paths: ProjectPaths, *, subject_id: str, task: str) -> Path:
    """Return the default results directory for one subject-level TRF run."""
    return paths.out_dir / TRF_RESULTS_DIRNAME / subject_id / f"task-{task}"


def _fold_results_payload(fold_results: Sequence[TrfFoldResult]) -> dict[str, Any]:
    """Convert fold results into JSON-friendly dictionaries."""
    return {
        "folds": [
            {
                "outer_fold_index": result.outer_fold_index,
                "score": result.score,
                "selected_alpha": result.selected_alpha,
                "actual_outer_splits": result.actual_outer_splits,
                "actual_inner_splits": result.actual_inner_splits,
                "train_segment_ids": list(result.train_segment_ids),
                "test_segment_ids": list(result.test_segment_ids),
            }
            for result in fold_results
        ]
    }


def run_trf_pipeline(
    *,
    cfg: ProjectConfig,
    subject_id: str,
    task: str = DEFAULT_TRF_TASK,
    out_dir: Path | None = None,
) -> TrfSubjectResult:
    """Run the subject-level TRF pipeline and persist configured outputs."""
    trf_cfg = TrfConfig.from_mapping(cfg.raw.get("trf"))
    resolved_predictor_names = _expand_predictor_names(trf_cfg.predictors)
    paths = ProjectPaths.from_config(cfg)
    target_out_dir = out_dir or _default_output_dir(paths, subject_id=subject_id, task=task)
    target_out_dir.mkdir(parents=True, exist_ok=True)

    run_ids = [str(run_id) for run_id in cfg.raw.get("runs", {}).get("include", {}).get(task, [])]
    run_inputs, skipped_runs = load_trf_run_inputs(cfg=cfg, paths=paths, subject_id=subject_id, task=task, run_ids=run_ids)
    if len(run_inputs) == 0:
        raise ValueError(f"No TRF runs could be loaded for {subject_id} task={task!r}.")

    lag_samples = compute_lag_samples(
        tmin_seconds=trf_cfg.lags.tmin_seconds,
        tmax_seconds=trf_cfg.lags.tmax_seconds,
        sampling_rate_hz=trf_cfg.target_sfreq,
    )
    segments: list[TrfSegment] = []
    skipped_segments: list[str] = []
    for run_input in run_inputs:
        run_segments, run_skipped_segments = split_run_into_segments(run_input, trf_cfg)
        segments.extend(run_segments)
        skipped_segments.extend(run_skipped_segments)

    if len(segments) == 0:
        raise ValueError(f"No valid TRF segments were available for {subject_id} task={task!r}.")

    lagged_designs = [build_lagged_segment_design(segment, lag_samples) for segment in segments]
    coefficient_path = target_out_dir / "coefficients.npz" if trf_cfg.outputs.save_coefficients else None
    fold_results, coefficient_paths = fit_nested_trf(
        lagged_designs,
        config=trf_cfg,
        coefficient_path=coefficient_path,
    )

    if trf_cfg.outputs.save_fold_scores:
        _write_json(target_out_dir / "fold_scores.json", _fold_results_payload(fold_results))

    if trf_cfg.outputs.save_selected_alpha_per_fold:
        _write_json(
            target_out_dir / "selected_alpha_per_fold.json",
            {
                "folds": [
                    {"outer_fold_index": result.outer_fold_index, "selected_alpha": result.selected_alpha}
                    for result in fold_results
                ]
            },
        )

    if trf_cfg.outputs.save_design_info:
        _write_json(
            target_out_dir / "design_info.json",
            {
                "subject_id": subject_id,
                "task": task,
                "configured_predictors": list(trf_cfg.predictors),
                "predictors": list(resolved_predictor_names),
                "target_sfreq": trf_cfg.target_sfreq,
                "lag_samples": lag_samples.tolist(),
                "lag_seconds": (lag_samples[::-1] / trf_cfg.target_sfreq).tolist(),
                "available_runs": [run_input.run_id for run_input in run_inputs],
                "skipped_runs": list(skipped_runs),
                "skipped_segments": list(skipped_segments),
                "segments": [
                    {
                        "segment_id": segment.segment_id,
                        "run_id": segment.run_id,
                        "start_sample": segment.start_sample,
                        "stop_sample": segment.stop_sample,
                        "start_seconds": segment.start_seconds,
                        "stop_seconds": segment.stop_seconds,
                    }
                    for segment in segments
                ],
            },
        )

    return TrfSubjectResult(
        subject_id=subject_id,
        task=task,
        predictor_names=resolved_predictor_names,
        lag_seconds=(lag_samples[::-1] / trf_cfg.target_sfreq).astype(np.float32),
        fold_results=tuple(fold_results),
        coefficient_paths=tuple(coefficient_paths),
        skipped_runs=tuple(skipped_runs),
        skipped_segments=tuple(skipped_segments),
        out_dir=target_out_dir,
    )
