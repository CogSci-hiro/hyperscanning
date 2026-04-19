"""QC figure builders for TRF outputs."""

from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from hyper.config import ProjectConfig
from hyper.paths import ProjectPaths
from hyper.trf.config import load_trf_config
from hyper.trf.pipeline import (
    _fit_one_alpha,
    build_lagged_segment_design,
    compute_lag_samples,
    load_trf_run_inputs,
    prepare_group_kfold,
    split_run_into_segments,
)
from hyper.viz import plot_joint_map, sanitize_token

matplotlib.use("Agg")

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


@dataclass(frozen=True, slots=True)
class SubjectAlphaCurve:
    """Per-subject alpha sweep summary for one predictor."""

    subject_id: str
    predictor_name: str
    alpha_values: np.ndarray
    mean_scores: np.ndarray
    se_scores: np.ndarray
    fold_scores: np.ndarray


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


def _load_subject_predictor_names(
    paths: ProjectPaths,
    *,
    subject_id: str,
    task: str,
    fallback_predictors: Sequence[str],
) -> tuple[str, ...]:
    """Return predictor names recorded for one fitted TRF subject, or a fallback list."""
    design_info_path = _trf_subject_dir(paths, subject_id=subject_id, task=task) / TRF_DESIGN_INFO_FILENAME
    if design_info_path.exists():
        design_info = json.loads(design_info_path.read_text(encoding="utf-8"))
        predictor_names = tuple(str(name) for name in design_info.get("predictors", []))
        if len(predictor_names) > 0:
            return predictor_names
    return tuple(str(name) for name in fallback_predictors)


def _single_predictor_cfg(cfg: ProjectConfig, predictor_name: str) -> ProjectConfig:
    """Clone the project config and replace TRF predictors with one self-consistent predictor."""
    raw = deepcopy(cfg.raw)
    trf_cfg = dict(raw.get("trf", {}))
    predictor = str(predictor_name)
    trf_cfg["predictors"] = [predictor]
    trf_cfg["qc_predictors"] = [predictor]
    # Alpha QC fits a standalone single-predictor model, so feature ablations do not apply.
    trf_cfg["ablation_targets"] = []
    raw["trf"] = trf_cfg
    return ProjectConfig(raw=raw)


def _standard_error(values: np.ndarray) -> np.ndarray:
    """Compute column-wise standard error with safe handling for one fold."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must have shape (n_folds, n_points)")
    if array.shape[0] <= 1:
        return np.zeros(array.shape[1], dtype=float)
    return np.nanstd(array, axis=0, ddof=1) / np.sqrt(float(array.shape[0]))


def _compute_subject_alpha_curve(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    subject_id: str,
    task: str,
    predictor_name: str,
) -> SubjectAlphaCurve | None:
    """Compute held-out outer-fold alpha scores for one subject and one predictor."""
    single_cfg = _single_predictor_cfg(cfg, predictor_name)
    trf_cfg = load_trf_config(single_cfg)
    run_ids = [str(run_id) for run_id in single_cfg.raw.get("runs", {}).get("include", {}).get(task, [])]
    run_inputs, _ = load_trf_run_inputs(
        cfg=single_cfg,
        paths=paths,
        subject_id=subject_id,
        task=task,
        run_ids=run_ids,
    )
    if len(run_inputs) == 0:
        return None

    lag_samples = compute_lag_samples(
        tmin_seconds=trf_cfg.lags.tmin_seconds,
        tmax_seconds=trf_cfg.lags.tmax_seconds,
        sampling_rate_hz=trf_cfg.target_sfreq,
    )
    segments = []
    for run_input in run_inputs:
        run_segments, _ = split_run_into_segments(run_input, trf_cfg)
        segments.extend(run_segments)
    if len(segments) == 0:
        return None

    lagged_designs = [build_lagged_segment_design(segment, lag_samples) for segment in segments]
    alpha_values = np.asarray(trf_cfg.alpha_values(), dtype=float)
    outer_splitter, outer_groups, _ = prepare_group_kfold(
        lagged_designs,
        requested_splits=trf_cfg.cv.outer.n_splits,
        group_by=trf_cfg.cv.outer.group_by,
        context=f"outer alpha QC for {subject_id} {predictor_name}",
    )
    segment_indices = np.arange(len(lagged_designs))
    score_rows: list[list[float]] = []
    for train_index, test_index in outer_splitter.split(segment_indices, groups=outer_groups):
        train_designs = [lagged_designs[index] for index in train_index]
        test_designs = [lagged_designs[index] for index in test_index]
        fold_scores = []
        for alpha in alpha_values:
            fold_score, _ = _fit_one_alpha(
                train_designs,
                test_designs,
                alpha=float(alpha),
                config=trf_cfg,
                lag_samples=lag_samples,
            )
            fold_scores.append(float(fold_score))
        score_rows.append(fold_scores)

    fold_scores_array = np.asarray(score_rows, dtype=float)
    return SubjectAlphaCurve(
        subject_id=subject_id,
        predictor_name=str(predictor_name),
        alpha_values=alpha_values,
        mean_scores=np.nanmean(fold_scores_array, axis=0),
        se_scores=_standard_error(fold_scores_array),
        fold_scores=fold_scores_array,
    )


def _plot_subject_alpha_curves(
    curves: Sequence[SubjectAlphaCurve],
    *,
    output_stem: Path,
    task: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    """Render one subject-level alpha sweep figure with one line per predictor."""
    if len(curves) == 0:
        raise ValueError("Expected at least one subject alpha curve to plot.")
    subject_id = curves[0].subject_id
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for curve in curves:
        axis.plot(curve.alpha_values, curve.mean_scores, linewidth=2.0, label=curve.predictor_name)
        axis.fill_between(
            curve.alpha_values,
            curve.mean_scores - curve.se_scores,
            curve.mean_scores + curve.se_scores,
            alpha=0.2,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Alpha")
    axis.set_ylabel("Channel-mean held-out score")
    axis.set_title(f"TRF alpha QC | {subject_id} | {task}")
    axis.legend(frameon=False)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for fmt in formats:
        output_path = output_stem.with_suffix(f".{fmt}")
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(output_path)
    plt.close(figure)
    return written_paths


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


def build_subject_alpha_qc_manifest(
    *,
    cfg: ProjectConfig,
    task: str,
    manifest_path: Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, object]:
    """Render one per-subject alpha-sweep QC figure with one line per predictor."""
    paths = ProjectPaths.from_config(cfg)
    fallback_predictors = tuple(str(name) for name in cfg.raw.get("trf", {}).get("predictors", []))
    plot_entries: list[dict[str, object]] = []
    for subject_id in _discover_subject_ids(cfg, paths):
        predictor_names = _load_subject_predictor_names(
            paths,
            subject_id=subject_id,
            task=task,
            fallback_predictors=fallback_predictors,
        )
        curves = []
        for predictor_name in predictor_names:
            curve = _compute_subject_alpha_curve(
                cfg,
                paths,
                subject_id=subject_id,
                task=task,
                predictor_name=predictor_name,
            )
            if curve is not None:
                curves.append(curve)
        if len(curves) == 0:
            continue
        output_stem = manifest_path.parent / "subjects" / sanitize_token(subject_id)
        written_paths = _plot_subject_alpha_curves(
            curves,
            output_stem=output_stem,
            task=task,
            formats=formats,
            dpi=dpi,
        )
        plot_entries.append(
            {
                "subject_id": subject_id,
                "task": task,
                "predictors": [curve.predictor_name for curve in curves],
                "alpha_values": curves[0].alpha_values.tolist(),
                "files": [str(path) for path in written_paths],
            }
        )

    manifest = {
        "status": "ok",
        "task": task,
        "plot_count": len(plot_entries),
        "plots": plot_entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
