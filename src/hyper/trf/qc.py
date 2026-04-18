"""QC figure builders for TRF outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyper.config import ProjectConfig
from hyper.paths import ProjectPaths
from hyper.viz import plot_joint_map, sanitize_token

TRF_COEFFICIENTS_FILENAME = "coefficients.npz"
TRF_DESIGN_INFO_FILENAME = "design_info.json"


@dataclass(frozen=True, slots=True)
class SubjectKernelSummary:
    """Subject-level TRF kernel summary after fold averaging."""

    subject_id: str
    predictor_names: tuple[str, ...]
    lag_seconds: np.ndarray
    channel_names: tuple[str, ...]
    mean_kernel_lag_feature_channel: np.ndarray
    outer_fold_count: int


def _discover_subject_ids(cfg: ProjectConfig, paths: ProjectPaths) -> list[str]:
    """Discover subjects from the BIDS root while honoring config filters."""
    if bool(cfg.raw.get("debug", {}).get("enabled", False)):
        debug_subjects = [str(subject) for subject in cfg.raw.get("debug", {}).get("subjects", [])]
        if len(debug_subjects) > 0:
            return debug_subjects

    pattern = str(cfg.raw.get("subjects", {}).get("pattern", "sub-*"))
    excluded = {str(subject) for subject in cfg.raw.get("subjects", {}).get("exclude", [])}
    return sorted(path.name for path in paths.raw_root.glob(pattern) if path.is_dir() and path.name not in excluded)


def _trf_subject_dir(paths: ProjectPaths, *, subject_id: str, task: str) -> Path:
    """Return the subject-level TRF output directory."""
    return paths.out_dir / "trf" / subject_id / f"task-{task}"


def _filtered_raw_path(paths: ProjectPaths, *, subject_id: str, task: str, run_id: str) -> Path:
    """Return the filtered raw path used to recover channel order."""
    stem = f"{subject_id}_task-{task}_run-{run_id}"
    return paths.out_dir / "eeg" / "filtered" / f"{stem}_raw_filt.fif"


def _load_channel_names_for_subject(
    paths: ProjectPaths,
    *,
    subject_id: str,
    task: str,
    available_runs: Sequence[str],
) -> list[str]:
    """Load EEG channel names from the first available filtered raw file."""
    import mne

    for run_id in available_runs:
        raw_path = _filtered_raw_path(paths, subject_id=subject_id, task=task, run_id=str(run_id))
        if raw_path.exists():
            raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="ERROR")
            return [str(name) for name in raw.ch_names]
    raise FileNotFoundError(f"No filtered raw file found for {subject_id} task={task!r} using runs {list(available_runs)}.")


def _reduce_fold_kernel(fold_kernel: np.ndarray, *, predictor_count: int) -> np.ndarray:
    """Normalize one stored fold kernel to (n_lags, n_predictors, n_channels)."""
    kernel = np.asarray(fold_kernel, dtype=np.float32)
    if kernel.ndim == 4:
        if kernel.shape[-1] != 1:
            raise ValueError(f"Expected one fitted alpha per outer fold, got coefficient shape {kernel.shape}.")
        kernel = kernel[..., 0]
    if kernel.ndim != 3:
        raise ValueError(f"Expected TRF coefficients with 3 dims after squeeze, got {kernel.shape}.")
    if kernel.shape[1] != predictor_count:
        raise ValueError(
            "TRF coefficient predictor axis does not match design_info predictors: "
            f"shape={kernel.shape}, predictor_count={predictor_count}."
        )
    return kernel


def _load_subject_kernel_summary(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    subject_id: str,
    task: str,
) -> SubjectKernelSummary | None:
    """Load and fold-average one subject's TRF kernels."""
    del cfg
    subject_dir = _trf_subject_dir(paths, subject_id=subject_id, task=task)
    coefficient_path = subject_dir / TRF_COEFFICIENTS_FILENAME
    design_info_path = subject_dir / TRF_DESIGN_INFO_FILENAME
    if not coefficient_path.exists() or not design_info_path.exists():
        return None

    design_info = json.loads(design_info_path.read_text(encoding="utf-8"))
    predictor_names = tuple(str(name) for name in design_info.get("predictors", []))
    if len(predictor_names) == 0:
        return None
    available_runs = [str(run_id) for run_id in design_info.get("available_runs", [])]
    if len(available_runs) == 0:
        return None

    coefficient_npz = np.load(coefficient_path)
    lag_seconds = np.asarray(coefficient_npz["lag_seconds"], dtype=np.float32)
    outer_fold_keys = sorted(key for key in coefficient_npz.files if key.startswith("outer_fold_"))
    if len(outer_fold_keys) == 0:
        return None

    fold_kernels = [
        _reduce_fold_kernel(coefficient_npz[fold_key], predictor_count=len(predictor_names))
        for fold_key in outer_fold_keys
    ]
    mean_kernel = np.nanmean(np.stack(fold_kernels, axis=0), axis=0)
    channel_names = tuple(
        _load_channel_names_for_subject(
            paths,
            subject_id=subject_id,
            task=task,
            available_runs=available_runs,
        )
    )
    if mean_kernel.shape[0] != lag_seconds.shape[0]:
        raise ValueError(
            f"TRF lag axis mismatch for {subject_id}: kernel shape {mean_kernel.shape}, lag_seconds {lag_seconds.shape}."
        )
    if mean_kernel.shape[2] != len(channel_names):
        raise ValueError(
            f"TRF channel axis mismatch for {subject_id}: kernel shape {mean_kernel.shape}, n_channels {len(channel_names)}."
        )

    return SubjectKernelSummary(
        subject_id=subject_id,
        predictor_names=predictor_names,
        lag_seconds=lag_seconds,
        channel_names=channel_names,
        mean_kernel_lag_feature_channel=mean_kernel,
        outer_fold_count=len(outer_fold_keys),
    )


def build_group_average_trf_kernel_manifest(
    *,
    cfg: ProjectConfig,
    task: str,
    manifest_path: Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    line_width: float = 2.5,
) -> dict[str, object]:
    """Render one group-average joint plot per TRF predictor."""
    paths = ProjectPaths.from_config(cfg)
    subject_summaries: list[SubjectKernelSummary] = []
    for subject_id in _discover_subject_ids(cfg, paths):
        summary = _load_subject_kernel_summary(cfg, paths, subject_id=subject_id, task=task)
        if summary is None:
            continue
        if not subject_summaries:
            subject_summaries.append(summary)
            continue
        reference = subject_summaries[0]
        if summary.predictor_names != reference.predictor_names:
            continue
        if summary.channel_names != reference.channel_names:
            continue
        if not np.array_equal(summary.lag_seconds, reference.lag_seconds, equal_nan=True):
            continue
        subject_summaries.append(summary)

    if len(subject_summaries) == 0:
        raise ValueError(f"No TRF kernels were available to plot for task={task!r}.")

    reference = subject_summaries[0]
    group_stack = np.stack(
        [summary.mean_kernel_lag_feature_channel for summary in subject_summaries],
        axis=0,
    )
    group_mean = np.nanmean(group_stack, axis=0)

    plot_entries: list[dict[str, object]] = []
    figure_dir = manifest_path.parent / "group_average"
    for predictor_index, predictor_name in enumerate(reference.predictor_names):
        kernel_map = np.asarray(group_mean[:, predictor_index, :], dtype=np.float32).T
        output_stem = figure_dir / sanitize_token(predictor_name)
        written_paths = plot_joint_map(
            kernel_map,
            times=np.asarray(reference.lag_seconds, dtype=float),
            channel_names=list(reference.channel_names),
            output_stem=output_stem,
            title=f"TRF group mean kernel | {task} | {predictor_name}",
            formats=formats,
            dpi=dpi,
            line_width=line_width,
        )
        plot_entries.append(
            {
                "task": task,
                "predictor": predictor_name,
                "subject_count": len(subject_summaries),
                "subjects": [summary.subject_id for summary in subject_summaries],
                "outer_fold_counts": {summary.subject_id: summary.outer_fold_count for summary in subject_summaries},
                "files": [str(path) for path in written_paths],
            }
        )

    manifest = {
        "status": "ok",
        "task": task,
        "subject_count": len(subject_summaries),
        "subjects": [summary.subject_id for summary in subject_summaries],
        "plot_count": len(plot_entries),
        "plots": plot_entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
