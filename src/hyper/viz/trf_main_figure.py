"""Build a configurable multi-panel TRF joint-plot summary figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from hyper.config import ProjectConfig
from hyper.paths import ProjectPaths
from hyper.trf.qc import _discover_subject_ids, _load_subject_kernel_summary
from hyper.viz import plot_joint_map, sanitize_token

matplotlib.use("Agg")


@dataclass(frozen=True, slots=True)
class GroupAverageKernelData:
    """Group-average TRF kernels aligned across subjects."""

    predictor_names: tuple[str, ...]
    lag_seconds: np.ndarray
    channel_names: tuple[str, ...]
    group_mean_kernel_lag_feature_channel: np.ndarray
    subject_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PanelFeatureSpec:
    """One configured panel in the TRF main figure."""

    predictors: tuple[str, ...]
    label: str
    joint_times_seconds: tuple[float, ...] | None = None


def _trf_main_figure_cfg(cfg: ProjectConfig) -> dict:
    """Return TRF main-figure settings."""
    raw_cfg = cfg.raw.get("viz", {}).get("trf_main_figure", {})
    if not isinstance(raw_cfg, dict):
        raise ValueError("Config section `viz.trf_main_figure` must be a mapping.")
    return raw_cfg


def _task(cfg: ProjectConfig) -> str:
    """Return the task label for the TRF main figure."""
    return str(_trf_main_figure_cfg(cfg).get("task", "conversation"))


def _figure_size(cfg: ProjectConfig) -> tuple[float, float]:
    """Return output figure size."""
    figsize_cfg = _trf_main_figure_cfg(cfg).get("figsize", {})
    if not isinstance(figsize_cfg, dict):
        figsize_cfg = {}
    return (
        float(figsize_cfg.get("width", 18.0)),
        float(figsize_cfg.get("height", 18.0)),
    )


def _figure_dpi(cfg: ProjectConfig) -> int:
    """Return output figure DPI."""
    return int(_trf_main_figure_cfg(cfg).get("dpi", 300))


def _panel_dpi(cfg: ProjectConfig) -> int:
    """Return the temporary panel-render DPI."""
    return int(_trf_main_figure_cfg(cfg).get("panel_dpi", max(200, _figure_dpi(cfg))))


def _layout(cfg: ProjectConfig) -> tuple[int, int]:
    """Return panel grid shape."""
    layout_cfg = _trf_main_figure_cfg(cfg).get("layout", {})
    if not isinstance(layout_cfg, dict):
        layout_cfg = {}
    return (
        int(layout_cfg.get("rows", 3)),
        int(layout_cfg.get("cols", 3)),
    )


def _panel_features(cfg: ProjectConfig) -> list[PanelFeatureSpec]:
    """Return the ordered panel definitions to render."""
    raw_features = _trf_main_figure_cfg(cfg).get("features", [])
    if not isinstance(raw_features, list):
        raise ValueError("Config section `viz.trf_main_figure.features` must be a list.")
    resolved: list[PanelFeatureSpec] = []
    for item in raw_features:
        if isinstance(item, str):
            resolved.append(PanelFeatureSpec(predictors=(str(item),), label=str(item)))
            continue
        if not isinstance(item, dict):
            raise ValueError("Each `viz.trf_main_figure.features` entry must be a string or mapping.")
        predictors_value = item.get("predictors", item.get("predictor"))
        if isinstance(predictors_value, str):
            predictors = (str(predictors_value),)
        elif isinstance(predictors_value, list):
            predictors = tuple(str(name) for name in predictors_value)
        else:
            raise ValueError("Feature mappings must define `predictor` or `predictors`.")
        if len(predictors) == 0:
            raise ValueError("Feature mappings must define at least one predictor.")
        label = str(item.get("label", predictors[0]))
        joint_times_value = item.get("joint_times_seconds")
        joint_times_seconds = None
        if joint_times_value is not None:
            if not isinstance(joint_times_value, list) or len(joint_times_value) == 0:
                raise ValueError("Feature-specific `joint_times_seconds` must be a non-empty list when present.")
            joint_times_seconds = tuple(float(time_point) for time_point in joint_times_value)
        resolved.append(PanelFeatureSpec(predictors=predictors, label=label, joint_times_seconds=joint_times_seconds))
    if len(resolved) == 0:
        raise ValueError("Config section `viz.trf_main_figure.features` must list at least one predictor.")
    return resolved


def _feature_labels(cfg: ProjectConfig) -> dict[str, str]:
    """Return optional display labels for features."""
    labels = _trf_main_figure_cfg(cfg).get("feature_labels", {})
    if labels is None:
        return {}
    if not isinstance(labels, dict):
        raise ValueError("Config section `viz.trf_main_figure.feature_labels` must be a mapping when present.")
    return {str(key): str(value) for key, value in labels.items()}


def _joint_times(cfg: ProjectConfig) -> np.ndarray:
    """Return configured topomap times in seconds."""
    times = _trf_main_figure_cfg(cfg).get("joint_times_seconds", [])
    if not isinstance(times, list):
        raise ValueError("Config section `viz.trf_main_figure.joint_times_seconds` must be a list.")
    if len(times) == 0:
        raise ValueError("Config section `viz.trf_main_figure.joint_times_seconds` must list at least one time.")
    return np.asarray([float(time_point) for time_point in times], dtype=float)


def _line_width(cfg: ProjectConfig) -> float:
    """Return the joint-plot line width."""
    joint_plot_cfg = _trf_main_figure_cfg(cfg).get("joint_plot", {})
    if not isinstance(joint_plot_cfg, dict):
        joint_plot_cfg = {}
    return float(joint_plot_cfg.get("line_width", 2.8))


def _font_scale(cfg: ProjectConfig) -> float:
    """Return the joint-plot text scaling factor."""
    joint_plot_cfg = _trf_main_figure_cfg(cfg).get("joint_plot", {})
    if not isinstance(joint_plot_cfg, dict):
        joint_plot_cfg = {}
    return float(joint_plot_cfg.get("font_scale", 3.0))


def _ylabel(cfg: ProjectConfig) -> str | None:
    """Return the main timeseries y-axis label."""
    joint_plot_cfg = _trf_main_figure_cfg(cfg).get("joint_plot", {})
    if not isinstance(joint_plot_cfg, dict):
        joint_plot_cfg = {}
    value = joint_plot_cfg.get("ylabel", "A.U.")
    return None if value is None else str(value)


def _show_colorbar(cfg: ProjectConfig) -> bool:
    """Return whether joint topomaps should include a colorbar."""
    joint_plot_cfg = _trf_main_figure_cfg(cfg).get("joint_plot", {})
    if not isinstance(joint_plot_cfg, dict):
        joint_plot_cfg = {}
    return bool(joint_plot_cfg.get("show_colorbar", False))


def _compact_vertical(cfg: ProjectConfig) -> bool:
    """Return whether joint panels should minimize vertical whitespace."""
    joint_plot_cfg = _trf_main_figure_cfg(cfg).get("joint_plot", {})
    if not isinstance(joint_plot_cfg, dict):
        joint_plot_cfg = {}
    return bool(joint_plot_cfg.get("compact_vertical", True))


def _load_group_average_kernel_data(cfg: ProjectConfig, *, task: str) -> GroupAverageKernelData:
    """Load group-average kernels from subject-level TRF outputs."""
    paths = ProjectPaths.from_config(cfg)
    subject_summaries = []
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
    return GroupAverageKernelData(
        predictor_names=reference.predictor_names,
        lag_seconds=np.asarray(reference.lag_seconds, dtype=float),
        channel_names=reference.channel_names,
        group_mean_kernel_lag_feature_channel=np.nanmean(group_stack, axis=0),
        subject_ids=tuple(summary.subject_id for summary in subject_summaries),
    )


def build_trf_main_figure(
    *,
    cfg: ProjectConfig,
    output_path: Path,
) -> Path:
    """Render a grid of feature-specific TRF joint plots into one summary figure."""
    task = _task(cfg)
    data = _load_group_average_kernel_data(cfg, task=task)
    features = _panel_features(cfg)
    feature_labels = _feature_labels(cfg)
    joint_times = _joint_times(cfg)
    rows, cols = _layout(cfg)
    panel_count = rows * cols
    if len(features) > panel_count:
        raise ValueError(
            f"Configured {len(features)} features but the layout only has space for {panel_count} panels."
        )

    missing = [
        predictor_name
        for feature in features
        for predictor_name in feature.predictors
        if predictor_name not in data.predictor_names
    ]
    if missing:
        raise ValueError(
            "Configured TRF main-figure features were not found in fitted predictors: "
            f"missing={missing}, available={list(data.predictor_names)}."
        )

    with TemporaryDirectory(prefix="trf_main_figure_") as tmpdir:
        image_paths: list[Path] = []
        for feature in features:
            predictor_indices = [data.predictor_names.index(name) for name in feature.predictors]
            kernel_cube = np.asarray(
                data.group_mean_kernel_lag_feature_channel[:, np.asarray(predictor_indices, dtype=int), :],
                dtype=float,
            )
            kernel_map = np.nanmean(kernel_cube, axis=1).T
            title = feature_labels.get(feature.predictors[0], feature.label)
            feature_joint_times = (
                np.asarray(feature.joint_times_seconds, dtype=float)
                if feature.joint_times_seconds is not None
                else joint_times
            )
            output_stem = Path(tmpdir) / sanitize_token(title)
            written_paths = plot_joint_map(
                kernel_map,
                times=np.asarray(data.lag_seconds, dtype=float),
                channel_names=list(data.channel_names),
                output_stem=output_stem,
                title=None,
                formats=("png",),
                dpi=_panel_dpi(cfg),
                line_width=_line_width(cfg),
                joint_times=feature_joint_times,
                font_scale=_font_scale(cfg),
                ylabel=_ylabel(cfg),
                show_colorbar=_show_colorbar(cfg),
                compact_vertical=_compact_vertical(cfg),
            )
            image_paths.append(Path(written_paths[0]))

        figure, axes = plt.subplots(rows, cols, figsize=_figure_size(cfg), squeeze=False)
        flat_axes = list(axes.ravel())
        for axis, image_path, feature in zip(flat_axes, image_paths, features, strict=True):
            axis.imshow(plt.imread(image_path))
            axis.set_axis_off()
            axis.set_title(feature.label, y=1.14, fontsize=33, pad=0)
        for axis in flat_axes[len(image_paths):]:
            axis.set_axis_off()
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=-0.16)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=_figure_dpi(cfg), bbox_inches="tight")
        plt.close(figure)
    return output_path
