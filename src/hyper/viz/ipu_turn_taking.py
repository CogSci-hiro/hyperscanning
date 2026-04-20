"""IPU turn-taking summary figure helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from hyper.config import ProjectConfig

matplotlib.use("Agg")

IPU_FILENAME_PATTERN = re.compile(r"sub-(?P<subject>\d{3})_run-(?P<run>\d+)_ipu\.csv$", flags=re.IGNORECASE)
CATEGORY_ORDER: tuple[str, ...] = ("A", "B", "overlap", "silence")
CATEGORY_LABELS: dict[str, str] = {
    "A": "A alone",
    "B": "B alone",
    "overlap": "Overlap",
    "silence": "Silence",
}
CATEGORY_PALETTE: dict[str, str] = {
    "A": "#4c78a8",
    "B": "#f58518",
    "overlap": "#54a24b",
    "silence": "#9d9da1",
}
SPEAKER_PALETTE: dict[str, str] = {
    "A": "#4c78a8",
    "B": "#f58518",
}
FONT_SCALE = 2.0
TITLE_SIZE = 28
LABEL_SIZE = 24
TICK_SIZE = 20
LEGEND_SIZE = 18
SMALL_LEGEND_SIZE = LEGEND_SIZE * 0.8
SMALL_XTICK_SIZE = TICK_SIZE * 0.8


@dataclass(frozen=True, slots=True)
class DyadRun:
    """One conversation run for one dyad."""

    dyad_id: str
    run_id: str
    speaker_a_path: Path
    speaker_b_path: Path


@dataclass(frozen=True, slots=True)
class TurnSegment:
    """One constant-state segment across the dyad timeline."""

    start: float
    end: float
    category: str

    @property
    def duration(self) -> float:
        """Return segment duration in seconds."""
        return float(self.end - self.start)


def _turn_taking_cfg(cfg: ProjectConfig) -> dict:
    """Return the IPU turn-taking visualization settings."""
    raw_cfg = cfg.raw.get("viz", {}).get("ipu_turn_taking", {})
    if not isinstance(raw_cfg, dict):
        raise ValueError("Config section `viz.ipu_turn_taking` must be a mapping.")
    return raw_cfg


def _figure_size(cfg: ProjectConfig) -> tuple[float, float]:
    """Return configured figure size."""
    figsize_cfg = _turn_taking_cfg(cfg).get("figsize", {})
    if not isinstance(figsize_cfg, dict):
        figsize_cfg = {}
    return (
        float(figsize_cfg.get("width", 24.0)),
        float(figsize_cfg.get("height", 7.5)),
    )


def _figure_dpi(cfg: ProjectConfig) -> int:
    """Return configured output DPI."""
    return int(_turn_taking_cfg(cfg).get("dpi", 300))


def _annotation_root(cfg: ProjectConfig) -> Path:
    """Resolve the configured annotation root directory."""
    paths_cfg = cfg.raw.get("paths", {})
    if not isinstance(paths_cfg, dict):
        raise ValueError("Config missing required mapping: paths")
    annotation_root = paths_cfg.get("annotation_root")
    if annotation_root is None:
        raise ValueError("Config paths must define 'annotation_root' for IPU turn-taking figures.")
    ipu_version = cfg.raw.get("annotations", {}).get("ipu")
    if ipu_version is None:
        raise ValueError("Config annotations must define 'ipu' for IPU turn-taking figures.")
    return Path(str(annotation_root)) / str(ipu_version)


def _is_odd_subject(subject_id: str) -> bool:
    """Return True for odd-numbered subject IDs."""
    return int(subject_id) % 2 == 1


def _match_ipu_filename(path: Path) -> re.Match[str] | None:
    """Match an IPU CSV filename stem."""
    return IPU_FILENAME_PATTERN.match(path.name)


def _infer_partner_ipu_path(path: Path) -> Path:
    """Infer the paired speaker IPU CSV using odd/even subject numbering."""
    match = _match_ipu_filename(path)
    if match is None:
        raise ValueError(f"Could not parse subject/run from IPU path: {path}")
    subject_id = int(str(match.group("subject")))
    run_id = str(match.group("run"))
    partner_id = subject_id + 1 if subject_id % 2 == 1 else subject_id - 1
    return path.parent / f"sub-{partner_id:03d}_run-{run_id}_ipu.csv"


def _discover_dyad_runs(annotation_dir: Path) -> list[DyadRun]:
    """Return odd/even dyad run pairs from an IPU annotation directory."""
    dyad_runs: list[DyadRun] = []
    for path in sorted(annotation_dir.glob("sub-*_run-*_ipu.csv")):
        match = _match_ipu_filename(path)
        if match is None:
            continue
        subject_id = str(match.group("subject"))
        if not _is_odd_subject(subject_id):
            continue
        partner_path = _infer_partner_ipu_path(path)
        if not partner_path.exists():
            continue
        run_id = str(match.group("run"))
        dyad_index = (int(subject_id) + 1) // 2
        dyad_runs.append(
            DyadRun(
                dyad_id=f"dyad-{dyad_index:03d}",
                run_id=run_id,
                speaker_a_path=path,
                speaker_b_path=partner_path,
            )
        )
    if len(dyad_runs) == 0:
        raise ValueError(f"No odd/even IPU dyad runs found in {annotation_dir}.")
    return dyad_runs


def _load_ipu_table(path: Path) -> pd.DataFrame:
    """Load one IPU CSV and validate required columns."""
    table = pd.read_csv(path)
    required = {"start", "end", "duration"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"IPU table missing required columns {missing}: {path}")
    out = table.copy()
    out["start"] = out["start"].astype(float)
    out["end"] = out["end"].astype(float)
    out["duration"] = out["duration"].astype(float)
    out = out.loc[out["end"] > out["start"]].sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    return out


def _segments_from_ipus(a_table: pd.DataFrame, b_table: pd.DataFrame) -> list[TurnSegment]:
    """Split the shared run timeline into A/B/overlap/silence segments."""
    max_end = max(
        float(a_table["end"].max()) if len(a_table) else 0.0,
        float(b_table["end"].max()) if len(b_table) else 0.0,
    )
    if max_end <= 0:
        return []
    boundaries = np.unique(
        np.concatenate(
            [
                np.array([0.0, max_end], dtype=float),
                a_table[["start", "end"]].to_numpy(dtype=float).ravel() if len(a_table) else np.array([], dtype=float),
                b_table[["start", "end"]].to_numpy(dtype=float).ravel() if len(b_table) else np.array([], dtype=float),
            ]
        )
    )
    segments: list[TurnSegment] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
        if end <= start:
            continue
        a_active = bool(((a_table["start"] < end) & (a_table["end"] > start)).any())
        b_active = bool(((b_table["start"] < end) & (b_table["end"] > start)).any())
        if a_active and b_active:
            category = "overlap"
        elif a_active:
            category = "A"
        elif b_active:
            category = "B"
        else:
            category = "silence"
        segments.append(TurnSegment(start=float(start), end=float(end), category=category))
    return segments


def _cumulative_path(segments: list[TurnSegment]) -> tuple[np.ndarray, np.ndarray]:
    """Build cumulative speaking-time coordinates for one run."""
    x_values = [0.0]
    y_values = [0.0]
    x_pos = 0.0
    y_pos = 0.0
    for segment in segments:
        if segment.category == "silence":
            continue
        if segment.category in {"A", "overlap"}:
            x_pos += segment.duration
        if segment.category in {"B", "overlap"}:
            y_pos += segment.duration
        x_values.append(x_pos)
        y_values.append(y_pos)
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def _segments_to_summary_rows(dyad_run: DyadRun, segments: list[TurnSegment]) -> list[dict[str, object]]:
    """Return tidy rows summarizing segment durations."""
    rows: list[dict[str, object]] = []
    for segment in segments:
        rows.append(
            {
                "dyad": dyad_run.dyad_id,
                "run": dyad_run.run_id,
                "category": segment.category,
                "duration_s": segment.duration,
            }
        )
    return rows


def _load_turn_taking_inputs(annotation_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load per-IPU, per-run path, and per-dyad breakdown tables."""
    ipu_rows: list[dict[str, object]] = []
    cumulative_rows: list[dict[str, object]] = []
    breakdown_rows: list[dict[str, object]] = []
    for dyad_run in _discover_dyad_runs(annotation_dir):
        a_table = _load_ipu_table(dyad_run.speaker_a_path)
        b_table = _load_ipu_table(dyad_run.speaker_b_path)
        for speaker, table in (("A", a_table), ("B", b_table)):
            for duration in table["duration"].astype(float):
                ipu_rows.append(
                    {
                        "dyad": dyad_run.dyad_id,
                        "run": dyad_run.run_id,
                        "speaker": speaker,
                        "duration_s": float(duration),
                    }
                )
        segments = _segments_from_ipus(a_table, b_table)
        x_values, y_values = _cumulative_path(segments)
        for order, (x_value, y_value) in enumerate(zip(x_values, y_values, strict=False)):
            cumulative_rows.append(
                {
                    "dyad": dyad_run.dyad_id,
                    "run": dyad_run.run_id,
                    "order": order,
                    "speaker_a_cumulative_s": float(x_value),
                    "speaker_b_cumulative_s": float(y_value),
                }
            )
        breakdown_rows.extend(_segments_to_summary_rows(dyad_run, segments))

    ipu_table = pd.DataFrame(ipu_rows)
    cumulative_table = pd.DataFrame(cumulative_rows)
    breakdown_table = pd.DataFrame(breakdown_rows)
    if ipu_table.empty or cumulative_table.empty or breakdown_table.empty:
        raise ValueError(f"IPU turn-taking inputs are empty for {annotation_dir}.")
    return ipu_table, cumulative_table, breakdown_table


def _apply_publication_style() -> None:
    """Apply a clean report style."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=FONT_SCALE,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#2b2b2b",
            "axes.linewidth": 1.1,
            "grid.color": "#d7d7d7",
            "grid.linewidth": 0.85,
            "grid.alpha": 0.55,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "legend.title_fontsize": LEGEND_SIZE,
        },
    )


def _plot_ipu_duration_histogram(axis: plt.Axes, ipu_table: pd.DataFrame) -> None:
    """Render the IPU duration histogram."""
    bins = np.linspace(0.0, 10.0, 60)
    sns.histplot(
        data=ipu_table,
        x="duration_s",
        bins=bins,
        stat="count",
        color="#6fa8a3",
        edgecolor=None,
        linewidth=0.0,
        alpha=0.98,
        ax=axis,
    )
    axis.axvline(
        float(ipu_table["duration_s"].median()),
        linestyle="--",
        linewidth=2.0,
        color="#1f1f1f",
        alpha=0.8,
    )
    axis.set_title("Global IPU duration distribution")
    axis.set_xlabel("Duration (s)")
    axis.set_ylabel("Count")
    axis.set_xlim(0.0, 10.0)
    axis.grid(axis="y")


def _plot_cumulative_speaking_time(axis: plt.Axes, cumulative_table: pd.DataFrame) -> None:
    """Render cumulative speaking time traces for each run."""
    dyads = list(dict.fromkeys(cumulative_table["dyad"].astype(str).tolist()))
    palette_values = sns.color_palette("husl", n_colors=len(dyads))
    dyad_palette = {dyad: palette_values[index] for index, dyad in enumerate(dyads)}
    for (dyad, run), run_table in cumulative_table.groupby(["dyad", "run"], sort=True):
        run_sorted = run_table.sort_values("order", kind="stable")
        axis.plot(
            run_sorted["speaker_a_cumulative_s"],
            run_sorted["speaker_b_cumulative_s"],
            color=dyad_palette[str(dyad)],
            alpha=0.82,
            linewidth=2.2,
        )
    max_value = float(
        max(
            cumulative_table["speaker_a_cumulative_s"].max(),
            cumulative_table["speaker_b_cumulative_s"].max(),
        )
    )
    axis.plot([0.0, max_value], [0.0, max_value], linestyle="--", linewidth=1.6, color="#555555", alpha=0.7)
    axis.set_title("Cumulative speaking time")
    axis.set_xlabel("Speaker A cumulative time (s)")
    axis.set_ylabel("Speaker B cumulative time (s)")
    handles = [Line2D([0], [0], color=dyad_palette[dyad], linewidth=2.0, label=dyad) for dyad in dyads]
    axis.legend(
        handles=handles,
        title="Dyad",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=SMALL_LEGEND_SIZE,
        title_fontsize=SMALL_LEGEND_SIZE,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)


def _plot_breakdown(axis: plt.Axes, breakdown_table: pd.DataFrame) -> None:
    """Render the per-dyad stacked duration breakdown."""
    summary = (
        breakdown_table.groupby(["dyad", "category"], as_index=False)["duration_s"]
        .sum()
        .pivot(index="dyad", columns="category", values="duration_s")
        .fillna(0.0)
    )
    for category in CATEGORY_ORDER:
        if category not in summary.columns:
            summary[category] = 0.0
    summary = summary.loc[:, list(CATEGORY_ORDER)]
    totals = summary.sum(axis=1).replace(0.0, np.nan)
    proportions = summary.div(totals, axis=0).fillna(0.0)
    proportions = proportions.sort_values("silence", ascending=False, kind="stable")
    x_positions = np.arange(len(proportions), dtype=float)
    bottoms = np.zeros(len(proportions), dtype=float)
    for category in CATEGORY_ORDER:
        values = proportions[category].to_numpy(dtype=float)
        axis.bar(
            x_positions,
            values,
            bottom=bottoms,
            color=CATEGORY_PALETTE[category],
            edgecolor="white",
            linewidth=0.7,
            label=CATEGORY_LABELS[category],
        )
        bottoms += values
    axis.set_title("Turn-state breakdown")
    axis.set_xlabel("Dyad")
    axis.set_ylabel("Proportion of run time")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(proportions.index.tolist(), rotation=45, ha="right")
    axis.tick_params(axis="x", labelsize=SMALL_XTICK_SIZE)
    axis.legend(title="Category", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
    axis.grid(axis="y")


def build_ipu_turn_taking_figure(*, cfg: ProjectConfig, output_path: Path) -> Path:
    """Build the 3-panel IPU turn-taking summary figure."""
    annotation_dir = _annotation_root(cfg)
    ipu_table, cumulative_table, breakdown_table = _load_turn_taking_inputs(annotation_dir)
    _apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=_figure_size(cfg), dpi=_figure_dpi(cfg), constrained_layout=True)
    _plot_ipu_duration_histogram(axes[0], ipu_table)
    _plot_cumulative_speaking_time(axes[1], cumulative_table)
    _plot_breakdown(axes[2], breakdown_table)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.06, wspace=0.08, hspace=0.02)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
