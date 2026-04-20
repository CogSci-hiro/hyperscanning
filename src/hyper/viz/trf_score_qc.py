"""TRF score QC violin figure helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

from hyper.config import ProjectConfig

matplotlib.use("Agg")

FONT_SCALE = 2.0
X_TICK_LABEL_SCALE = 0.7
JITTER_ALPHA = 1.0
JITTER_SIZE = 3.1 * 1.5
JITTER_LINEWIDTH = 0.45 * 1.5
STAR_Y_OFFSET_FRACTION = 0.04
STAR_AXES_Y = 0.9


@dataclass(frozen=True, slots=True)
class TrfScoreQcInputs:
    """Loaded QC score tables."""

    eeg_table: pd.DataFrame
    feature_table: pd.DataFrame


def _trf_score_cfg(cfg: ProjectConfig) -> dict:
    """Return TRF-score figure settings."""
    raw_cfg = cfg.raw.get("viz", {}).get("trf_score", {})
    if not isinstance(raw_cfg, dict):
        raise ValueError("Config section `viz.trf_score` must be a mapping.")
    return raw_cfg


def _figure_size(cfg: ProjectConfig) -> tuple[float, float]:
    """Return configured TRF-score figure size."""
    figsize_cfg = _trf_score_cfg(cfg).get("figsize", {})
    if not isinstance(figsize_cfg, dict):
        figsize_cfg = {}
    return (
        float(figsize_cfg.get("width", 13.0)),
        float(figsize_cfg.get("height", 4.5)),
    )


def _figure_dpi(cfg: ProjectConfig) -> int:
    """Return configured TRF-score figure DPI."""
    return int(_trf_score_cfg(cfg).get("dpi", 300))


def _x_tick_label_scale(cfg: ProjectConfig) -> float:
    """Return the x tick-label font scale for the TRF-score figure."""
    return float(_trf_score_cfg(cfg).get("x_tick_label_scale", X_TICK_LABEL_SCALE))


def _plotted_feature_labels(cfg: ProjectConfig) -> dict[str, str]:
    """Return optional feature-label aliases for the TRF score plot."""
    raw_mapping = _trf_score_cfg(cfg).get("plotted_features", {})
    if raw_mapping is None:
        return {}
    if not isinstance(raw_mapping, dict):
        raise ValueError("Config section `viz.trf_score.plotted_features` must be a mapping when present.")
    return {str(key): str(value) for key, value in raw_mapping.items()}


def _load_inputs(*, eeg_table_path: Path, feature_table_path: Path) -> TrfScoreQcInputs:
    """Load machine-readable TRF QC tables."""
    eeg_table = pd.read_csv(eeg_table_path, sep="\t")
    feature_table = pd.read_csv(feature_table_path, sep="\t")
    return TrfScoreQcInputs(eeg_table=eeg_table, feature_table=feature_table)


def _apply_publication_style() -> None:
    """Apply a clean Seaborn theme for publication-ready violins."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.15,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#303030",
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.7,
            "xtick.color": "#202020",
            "ytick.color": "#202020",
            "axes.labelcolor": "#202020",
            "axes.titleweight": "semibold",
        },
    )


def _pvalue_to_stars(pvalue: float) -> str:
    """Convert a p-value to conventional significance stars."""
    if pvalue <= 0.001:
        return "***"
    if pvalue <= 0.01:
        return "**"
    if pvalue <= 0.05:
        return "*"
    return ""


def _annotate_one_sample_stars(
    axis: plt.Axes,
    *,
    values_by_group: dict[str, np.ndarray],
    order: list[str],
    positions: dict[str, float] | None = None,
    axes_fraction_y: float | None = None,
) -> None:
    """Add one-sample t-test significance stars above requested groups."""
    if not hasattr(axis, "get_ylim") or not hasattr(axis, "text"):
        return
    ymin, ymax = axis.get_ylim()
    offset = (ymax - ymin) * STAR_Y_OFFSET_FRACTION
    for index, group in enumerate(order):
        values = np.asarray(values_by_group.get(group, np.array([], dtype=float)), dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        test = stats.ttest_1samp(values, 0.0, nan_policy="omit")
        pvalue = float(test.pvalue) if np.isfinite(test.pvalue) else 1.0
        stars = _pvalue_to_stars(pvalue)
        if not stars:
            continue
        x = float(positions[group]) if positions is not None and group in positions else float(index)
        if axes_fraction_y is not None and hasattr(axis, "get_xaxis_transform"):
            axis.text(x, axes_fraction_y, stars, ha="center", va="bottom", transform=axis.get_xaxis_transform())
            continue
        y = min(ymax - offset * 0.25, float(np.nanmax(values)) + offset)
        axis.text(x, y, stars, ha="center", va="bottom")


def _plot_eeg_quality(axis: plt.Axes, eeg_table: pd.DataFrame) -> None:
    """Render the EEG QC violin subplot."""
    plot_table = pd.DataFrame(
        {
            "class_label": ["Null R"] * len(eeg_table) + ["Real R"] * len(eeg_table) + [r"$\Delta R$"] * len(eeg_table),
            "pearson_r": pd.concat(
                [
                    eeg_table["null_score"].astype(float),
                    eeg_table["real_score"].astype(float),
                    eeg_table["delta"].astype(float),
                ],
                ignore_index=True,
            ),
        }
    )
    palette = {
        "Null R": "#4c78a8",
        "Real R": "#2a9d8f",
        r"$\Delta R$": "#e76f51",
    }
    order = ["Null R", "Real R", r"$\Delta R$"]
    sns.violinplot(
        data=plot_table,
        x="class_label",
        y="pearson_r",
        hue="class_label",
        order=order,
        palette=palette,
        inner="box",
        cut=0,
        density_norm="width",
        linewidth=1.1,
        saturation=0.95,
        legend=False,
        ax=axis,
    )
    sns.stripplot(
        data=plot_table,
        x="class_label",
        y="pearson_r",
        hue="class_label",
        order=order,
        palette=palette,
        dodge=False,
        size=JITTER_SIZE,
        alpha=JITTER_ALPHA,
        linewidth=JITTER_LINEWIDTH,
        edgecolor="white",
        legend=False,
        ax=axis,
    )
    _annotate_one_sample_stars(
        axis,
        values_by_group={r"$\Delta R$": eeg_table["delta"].astype(float).to_numpy()},
        order=[r"$\Delta R$"],
        positions={r"$\Delta R$": 2.0},
        axes_fraction_y=STAR_AXES_Y,
    )
    axis.set_ylabel("Pearson R")
    axis.set_title("EEG quality check")
    axis.set_xlabel("")
    axis.grid(axis="y", alpha=0.45, linewidth=0.7)
    axis.set_axisbelow(True)
    sns.despine(ax=axis, offset=6)


def _plot_feature_quality(axis: plt.Axes, feature_table: pd.DataFrame, *, plotted_feature_labels: dict[str, str]) -> None:
    """Render the feature QC violin subplot."""
    plot_table = feature_table.copy()
    plot_table["target"] = plot_table["target"].astype(str)
    if plotted_feature_labels:
        plot_table = plot_table.loc[plot_table["target"].isin(plotted_feature_labels)].copy()
    feature_names = [str(name) for name in plot_table["target"].tolist()]
    if len(feature_names) == 0:
        raise ValueError("Feature QC table is empty; cannot render violin plot.")
    categories = list(dict.fromkeys(feature_names))
    display_labels = [plotted_feature_labels.get(category, category) for category in categories]
    palette_values = sns.color_palette("Set2", n_colors=len(categories))
    palette = {feature_name: palette_values[index] for index, feature_name in enumerate(categories)}
    plot_table["delta"] = plot_table["delta"].astype(float)
    sns.violinplot(
        data=plot_table,
        x="target",
        y="delta",
        hue="target",
        order=categories,
        palette=palette,
        inner="box",
        cut=0,
        density_norm="width",
        linewidth=1.1,
        saturation=0.95,
        legend=False,
        ax=axis,
    )
    sns.stripplot(
        data=plot_table,
        x="target",
        y="delta",
        hue="target",
        order=categories,
        palette=palette,
        dodge=False,
        size=JITTER_SIZE,
        alpha=JITTER_ALPHA,
        linewidth=JITTER_LINEWIDTH,
        edgecolor="white",
        legend=False,
        ax=axis,
    )
    _annotate_one_sample_stars(
        axis,
        values_by_group={
            category: plot_table.loc[plot_table["target"] == category, "delta"].astype(float).to_numpy()
            for category in categories
        },
        order=categories,
        positions={category: float(index) for index, category in enumerate(categories)},
        axes_fraction_y=STAR_AXES_Y,
    )
    axis.set_xticks(range(len(categories)), display_labels)
    axis.set_ylabel(r"Delta R ($\Delta R$)")
    axis.set_title("Feature quality check")
    axis.set_xlabel("")
    axis.grid(axis="y", alpha=0.45, linewidth=0.7)
    axis.set_axisbelow(True)
    sns.despine(ax=axis, offset=6)


def build_trf_score_qc_figure(
    *,
    cfg: ProjectConfig,
    eeg_table_path: Path,
    feature_table_path: Path,
    output_path: Path,
) -> Path:
    """Build a 1x2 TRF score QC violin figure."""
    inputs = _load_inputs(eeg_table_path=eeg_table_path, feature_table_path=feature_table_path)
    if inputs.eeg_table.empty:
        raise ValueError("EEG QC table is empty; cannot render violin plot.")
    if inputs.feature_table.empty:
        raise ValueError("Feature QC table is empty; cannot render violin plot.")

    _apply_publication_style()
    plotted_feature_labels = _plotted_feature_labels(cfg)
    x_tick_label_scale = _x_tick_label_scale(cfg)
    figure, axes = plt.subplots(1, 2, figsize=_figure_size(cfg), gridspec_kw={"width_ratios": [1, 3]})
    _plot_eeg_quality(axes[0], inputs.eeg_table)
    _plot_feature_quality(axes[1], inputs.feature_table, plotted_feature_labels=plotted_feature_labels)
    if hasattr(figure, "findobj"):
        for text in figure.findobj(matplotlib.text.Text):
            fontsize = text.get_fontsize()
            if fontsize is not None:
                text.set_fontsize(float(fontsize) * FONT_SCALE)
    for axis in axes:
        if not hasattr(axis, "get_xticklabels"):
            continue
        for tick in axis.get_xticklabels():
            fontsize = tick.get_fontsize()
            if fontsize is not None:
                tick.set_fontsize(float(fontsize) * x_tick_label_scale)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=_figure_dpi(cfg), bbox_inches="tight")
    plt.close(figure)
    return output_path
