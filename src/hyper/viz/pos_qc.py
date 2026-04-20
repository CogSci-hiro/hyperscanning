"""Lightweight QC summaries and plots for exported POS feature TSV files.

Expected exported POS TSV columns are inferred from the current Stanza POS
exporter in ``hyper.features.linguistic.pos``. At minimum, QC requires a token
text column resolved from ``("token", "word", "label")`` and the exported
``upos`` column. Optional identifiers such as ``run`` and ``speaker`` are used
when present, otherwise subject/run metadata can be supplied by the caller or
inferred from filenames like ``sub-001_task-conversation_run-1_desc-self_pos_features.tsv``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from hyper.features.linguistic.pos import PREFERRED_TEXT_COLUMNS

matplotlib.use("Agg")

# Constants

DEFAULT_DPI: Final[int] = 200
DEFAULT_FIGURE_WIDTH: Final[float] = 10.0
DEFAULT_FIGURE_HEIGHT: Final[float] = 5.0
HEATMAP_FIGURE_HEIGHT_PER_ROW: Final[float] = 0.45
HEATMAP_MIN_HEIGHT: Final[float] = 4.0
HEATMAP_MAX_HEIGHT: Final[float] = 14.0
BAR_LABEL_ROTATION: Final[int] = 45
MISSING_LABEL: Final[str] = "<MISSING>"
UNKNOWN_LABEL: Final[str] = "<UNKNOWN>"
FILE_ID_COLUMN: Final[str] = "source_file"
FILE_STEM_COLUMN: Final[str] = "source_file_stem"
SUBJECT_ID_COLUMN: Final[str] = "subject_id"
RUN_ID_COLUMN: Final[str] = "run_id"
RUN_UNIT_COLUMN: Final[str] = "run_unit"
TOKEN_TEXT_COLUMN: Final[str] = "token_text"
POS_TAG_COLUMN: Final[str] = "upos"
RUN_GROUPING_AUTO: Final[str] = "auto"
RUN_GROUPING_SUBJECT_RUN: Final[str] = "subject-run"
RUN_GROUPING_RUN: Final[str] = "run"
RUN_GROUPING_FILE: Final[str] = "file"
RUN_GROUPING_CHOICES: Final[tuple[str, ...]] = (
    RUN_GROUPING_AUTO,
    RUN_GROUPING_SUBJECT_RUN,
    RUN_GROUPING_RUN,
    RUN_GROUPING_FILE,
)
PROBLEM_METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("missing_pos_rate", "Missing POS"),
    ("pos_x_rate", "POS = X"),
    ("pos_sym_rate", "POS = SYM"),
    ("punctuation_rate", "Punctuation"),
    ("empty_token_rate", "Empty token"),
    ("non_alpha_rate", "Non-alphabetic"),
)
SUBJECT_PATTERN: Final[re.Pattern[str]] = re.compile(r"(sub-[A-Za-z0-9]+)")
RUN_PATTERN: Final[re.Pattern[str]] = re.compile(r"_run-([^_]+)")


@dataclass(frozen=True, slots=True)
class PosQcSchema:
    """Resolved POS QC schema for one or more exported tables."""

    token_text_column: str
    pos_tag_column: str
    subject_id_column: str
    run_id_column: str
    run_unit_column: str
    file_column: str


@dataclass(frozen=True, slots=True)
class PosQcDataset:
    """Normalized POS QC input plus its resolved schema."""

    data: pd.DataFrame
    schema: PosQcSchema


# Schema helpers


def resolve_pos_text_column(table: pd.DataFrame) -> str:
    """Resolve the released token text column using exporter conventions."""
    for column_name in PREFERRED_TEXT_COLUMNS:
        if column_name in table.columns:
            return str(column_name)
    available_columns = ", ".join(str(column_name) for column_name in table.columns)
    raise ValueError(
        "POS QC input must include one of the token text columns "
        f"{PREFERRED_TEXT_COLUMNS!r}; available columns: {available_columns}"
    )


def _first_non_empty(series: pd.Series) -> str | None:
    """Return the first non-empty string representation in a series."""
    for value in series.dropna():
        text = str(value).strip()
        if text:
            return text
    return None


def _infer_subject_id_from_path(path: Path) -> str | None:
    """Infer a subject identifier from the source filename when possible."""
    match = SUBJECT_PATTERN.search(path.name)
    if match is None:
        return None
    return str(match.group(1))


def _infer_run_id_from_path(path: Path) -> str | None:
    """Infer a run identifier from the source filename when possible."""
    match = RUN_PATTERN.search(path.name)
    if match is None:
        return None
    return str(match.group(1))


def _normalize_optional_identifier(value: object) -> str | None:
    """Normalize optional identifier values to stripped strings."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _build_run_unit(
    table: pd.DataFrame,
    *,
    grouping: str,
) -> pd.Series:
    """Build a deterministic run-unit label for within-run summaries."""
    if grouping == RUN_GROUPING_SUBJECT_RUN:
        return table[SUBJECT_ID_COLUMN].fillna(UNKNOWN_LABEL) + "_run-" + table[RUN_ID_COLUMN].fillna(UNKNOWN_LABEL)
    if grouping == RUN_GROUPING_RUN:
        return table[RUN_ID_COLUMN].fillna(UNKNOWN_LABEL)
    if grouping == RUN_GROUPING_FILE:
        return table[FILE_STEM_COLUMN].fillna(UNKNOWN_LABEL)
    raise ValueError(f"Unsupported run grouping: {grouping!r}")


def _resolve_grouping_mode(table: pd.DataFrame, requested_grouping: str) -> str:
    """Resolve automatic grouping to the best available identifier level."""
    if requested_grouping != RUN_GROUPING_AUTO:
        return requested_grouping

    has_subject = table[SUBJECT_ID_COLUMN].notna().any()
    has_run = table[RUN_ID_COLUMN].notna().any()
    if has_subject and has_run:
        return RUN_GROUPING_SUBJECT_RUN
    if has_run:
        return RUN_GROUPING_RUN
    return RUN_GROUPING_FILE


def load_pos_qc_dataset(
    input_paths: list[Path],
    *,
    subject_id: str | None = None,
    run_id: str | None = None,
    grouping: str = RUN_GROUPING_AUTO,
) -> PosQcDataset:
    """Load, validate, and normalize one or many exported POS TSV files.

    Parameters
    ----------
    input_paths
        POS feature TSV paths to concatenate into one QC dataset.
    subject_id
        Optional subject identifier override applied when an input file does
        not already carry or encode one.
    run_id
        Optional run identifier override applied when an input file does not
        already carry or encode one.
    grouping
        Run-unit grouping mode for per-run QC summaries.
    """
    if len(input_paths) == 0:
        raise ValueError("POS QC requires at least one input TSV path.")
    if grouping not in RUN_GROUPING_CHOICES:
        raise ValueError(f"Unsupported grouping {grouping!r}; expected one of {RUN_GROUPING_CHOICES!r}.")

    normalized_tables: list[pd.DataFrame] = []
    resolved_text_column: str | None = None
    for input_path in sorted(input_paths):
        table = pd.read_csv(input_path, sep="\t")
        text_column = resolve_pos_text_column(table)
        missing_required_columns = [column_name for column_name in (text_column, POS_TAG_COLUMN) if column_name not in table.columns]
        if missing_required_columns:
            missing_text = ", ".join(missing_required_columns)
            raise ValueError(f"{input_path} is missing required POS QC columns: {missing_text}")

        if resolved_text_column is None:
            resolved_text_column = text_column

        source_subject = _first_non_empty(table[SUBJECT_ID_COLUMN]) if SUBJECT_ID_COLUMN in table.columns else None
        source_run = _first_non_empty(table["run"]) if "run" in table.columns else None
        inferred_subject = _normalize_optional_identifier(source_subject) or _infer_subject_id_from_path(input_path) or _normalize_optional_identifier(subject_id)
        inferred_run = _normalize_optional_identifier(source_run) or _infer_run_id_from_path(input_path) or _normalize_optional_identifier(run_id)

        normalized = table.copy()
        normalized[TOKEN_TEXT_COLUMN] = normalized[text_column]
        normalized[FILE_ID_COLUMN] = str(input_path)
        normalized[FILE_STEM_COLUMN] = input_path.stem
        normalized[SUBJECT_ID_COLUMN] = inferred_subject if inferred_subject is not None else pd.NA
        normalized[RUN_ID_COLUMN] = inferred_run if inferred_run is not None else pd.NA
        normalized_tables.append(normalized)

    combined = pd.concat(normalized_tables, axis=0, ignore_index=True)
    active_grouping = _resolve_grouping_mode(combined, grouping)
    combined[RUN_UNIT_COLUMN] = _build_run_unit(combined, grouping=active_grouping)

    schema = PosQcSchema(
        token_text_column=resolved_text_column or TOKEN_TEXT_COLUMN,
        pos_tag_column=POS_TAG_COLUMN,
        subject_id_column=SUBJECT_ID_COLUMN,
        run_id_column=RUN_ID_COLUMN,
        run_unit_column=RUN_UNIT_COLUMN,
        file_column=FILE_ID_COLUMN,
    )
    return PosQcDataset(data=combined, schema=schema)


# Summary helpers


def _normalized_pos_series(table: pd.DataFrame) -> pd.Series:
    """Return POS tags with stable labels for missing values."""
    return table[POS_TAG_COLUMN].fillna(MISSING_LABEL).astype(str).replace({"": MISSING_LABEL})


def _token_text_series(table: pd.DataFrame) -> pd.Series:
    """Return token text as stripped strings with missing values preserved as empty."""
    return table[TOKEN_TEXT_COLUMN].fillna("").astype(str).map(str.strip)


def _is_non_alpha_token(token_text: str) -> bool:
    """Return whether a token lacks alphabetic characters entirely."""
    if token_text == "":
        return False
    return not any(character.isalpha() for character in token_text)


def _sort_pos_summary(table: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    """Sort POS tables deterministically by descending value then tag label."""
    return table.sort_values(by=[value_column, POS_TAG_COLUMN], ascending=[False, True]).reset_index(drop=True)


def compute_global_pos_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Compute global POS token counts."""
    counts = (
        _normalized_pos_series(table)
        .rename(POS_TAG_COLUMN)
        .value_counts(dropna=False)
        .rename("token_count")
        .reset_index()
        .rename(columns={"index": POS_TAG_COLUMN})
    )
    return _sort_pos_summary(counts, value_column="token_count")


def compute_global_pos_proportions(table: pd.DataFrame) -> pd.DataFrame:
    """Compute global POS token proportions."""
    counts = compute_global_pos_counts(table)
    total_tokens = float(counts["token_count"].sum())
    counts["token_proportion"] = counts["token_count"] / total_tokens if total_tokens > 0 else 0.0
    return counts[[POS_TAG_COLUMN, "token_proportion"]]


def compute_pos_proportions_by_run(table: pd.DataFrame) -> pd.DataFrame:
    """Compute within-run POS composition for each run unit."""
    working = table.copy()
    working[POS_TAG_COLUMN] = _normalized_pos_series(working)
    count_table = (
        working.groupby([RUN_UNIT_COLUMN, SUBJECT_ID_COLUMN, RUN_ID_COLUMN, POS_TAG_COLUMN], dropna=False)
        .size()
        .rename("token_count")
        .reset_index()
    )
    run_totals = count_table.groupby(RUN_UNIT_COLUMN, dropna=False)["token_count"].transform("sum")
    count_table["token_proportion"] = count_table["token_count"] / run_totals
    return count_table.sort_values(
        by=[RUN_UNIT_COLUMN, "token_proportion", POS_TAG_COLUMN],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def compute_problematic_metrics_by_run(table: pd.DataFrame) -> pd.DataFrame:
    """Compute per-run QC metrics for potentially problematic token rows."""
    working = table.copy()
    token_text = _token_text_series(working)
    pos_tags = _normalized_pos_series(working)

    working["is_missing_pos"] = pos_tags.eq(MISSING_LABEL)
    working["is_pos_x"] = pos_tags.eq("X")
    working["is_pos_sym"] = pos_tags.eq("SYM")
    working["is_punctuation"] = pos_tags.eq("PUNCT") | token_text.map(lambda value: value != "" and all(not character.isalnum() for character in value))
    working["is_empty_token"] = token_text.eq("")
    working["is_non_alpha"] = token_text.map(_is_non_alpha_token)

    grouped = working.groupby([RUN_UNIT_COLUMN, SUBJECT_ID_COLUMN, RUN_ID_COLUMN], dropna=False)
    summary = grouped.agg(
        token_count=(POS_TAG_COLUMN, "size"),
        missing_pos_count=("is_missing_pos", "sum"),
        pos_x_count=("is_pos_x", "sum"),
        pos_sym_count=("is_pos_sym", "sum"),
        punctuation_count=("is_punctuation", "sum"),
        empty_token_count=("is_empty_token", "sum"),
        non_alpha_count=("is_non_alpha", "sum"),
    ).reset_index()

    for prefix in ("missing_pos", "pos_x", "pos_sym", "punctuation", "empty_token", "non_alpha"):
        summary[f"{prefix}_rate"] = summary[f"{prefix}_count"] / summary["token_count"]

    return summary.sort_values(by=[RUN_UNIT_COLUMN]).reset_index(drop=True)


def build_pos_heatmap_table(table: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-run POS proportions into a heatmap-ready matrix."""
    by_run = compute_pos_proportions_by_run(table)
    column_order = compute_global_pos_counts(table)[POS_TAG_COLUMN].tolist()
    heatmap = (
        by_run.pivot(index=RUN_UNIT_COLUMN, columns=POS_TAG_COLUMN, values="token_proportion")
        .reindex(columns=column_order)
        .fillna(0.0)
    )
    return heatmap.sort_index()


# Plot helpers


def _apply_title(ax: plt.Axes, *, title_prefix: str | None, title: str) -> None:
    """Apply a consistent optional title prefix."""
    if title_prefix:
        ax.set_title(f"{title_prefix} - {title}")
        return
    ax.set_title(title)


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Save and close one figure deterministically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(figure)


def plot_global_pos_distribution(
    proportions: pd.DataFrame,
    output_path: Path,
    *,
    title_prefix: str | None = None,
) -> None:
    """Plot global POS proportions as a bar chart."""
    figure, ax = plt.subplots(figsize=(DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT))
    ax.bar(proportions[POS_TAG_COLUMN], proportions["token_proportion"])
    ax.set_xlabel("POS tag")
    ax.set_ylabel("Proportion")
    ax.tick_params(axis="x", rotation=BAR_LABEL_ROTATION)
    _apply_title(ax, title_prefix=title_prefix, title="Global POS Distribution")
    _save_figure(figure, output_path)


def plot_pos_heatmap_by_run(
    heatmap_table: pd.DataFrame,
    output_path: Path,
    *,
    title_prefix: str | None = None,
) -> None:
    """Plot per-run POS composition as a simple heatmap."""
    if heatmap_table.empty:
        raise ValueError("Cannot render POS heatmap because the per-run table is empty.")

    figure_height = float(
        min(
            HEATMAP_MAX_HEIGHT,
            max(HEATMAP_MIN_HEIGHT, HEATMAP_FIGURE_HEIGHT_PER_ROW * float(len(heatmap_table.index))),
        )
    )
    figure, ax = plt.subplots(figsize=(DEFAULT_FIGURE_WIDTH, figure_height))
    image = ax.imshow(heatmap_table.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(heatmap_table.columns)))
    ax.set_xticklabels(list(heatmap_table.columns), rotation=BAR_LABEL_ROTATION, ha="right")
    ax.set_yticks(np.arange(len(heatmap_table.index)))
    ax.set_yticklabels(list(heatmap_table.index))
    ax.set_xlabel("POS tag")
    ax.set_ylabel("Run unit")
    _apply_title(ax, title_prefix=title_prefix, title="POS Composition by Run")
    figure.colorbar(image, ax=ax, label="Proportion")
    _save_figure(figure, output_path)


def plot_problematic_token_summary(
    problematic_metrics: pd.DataFrame,
    output_path: Path,
    *,
    title_prefix: str | None = None,
) -> None:
    """Plot aggregate problematic-token rates across all run units."""
    metric_columns = [metric_name for metric_name, _metric_label in PROBLEM_METRICS if metric_name in problematic_metrics.columns]
    if len(metric_columns) == 0:
        raise ValueError("Cannot render problematic-token summary because no supported metrics were computed.")

    label_lookup = dict(PROBLEM_METRICS)
    mean_rates = problematic_metrics[metric_columns].mean(axis=0)
    plot_table = pd.DataFrame(
        {
            "metric_label": [label_lookup[column_name] for column_name in metric_columns],
            "metric_rate": [float(mean_rates[column_name]) for column_name in metric_columns],
        }
    )

    figure, ax = plt.subplots(figsize=(DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT))
    ax.bar(plot_table["metric_label"], plot_table["metric_rate"])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean proportion across run units")
    ax.tick_params(axis="x", rotation=BAR_LABEL_ROTATION)
    _apply_title(ax, title_prefix=title_prefix, title="Problematic Token Summary")
    _save_figure(figure, output_path)


# Output orchestration


def write_pos_qc_outputs(
    dataset: PosQcDataset,
    output_dir: Path,
    *,
    title_prefix: str | None = None,
) -> dict[str, Path]:
    """Write standard POS QC tables and plots to one output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = compute_global_pos_counts(dataset.data)
    proportions = compute_global_pos_proportions(dataset.data)
    by_run = compute_pos_proportions_by_run(dataset.data)
    problematic = compute_problematic_metrics_by_run(dataset.data)
    heatmap_table = build_pos_heatmap_table(dataset.data)

    output_paths = {
        "pos_distribution": output_dir / "pos_distribution.png",
        "pos_heatmap_by_run": output_dir / "pos_heatmap_by_run.png",
        "pos_problematic_tokens": output_dir / "pos_problematic_tokens.png",
        "pos_counts": output_dir / "pos_counts.tsv",
        "pos_proportions": output_dir / "pos_proportions.tsv",
        "pos_proportions_by_run": output_dir / "pos_proportions_by_run.tsv",
        "pos_problematic_metrics_by_run": output_dir / "pos_problematic_metrics_by_run.tsv",
    }

    counts.to_csv(output_paths["pos_counts"], sep="\t", index=False)
    proportions.to_csv(output_paths["pos_proportions"], sep="\t", index=False)
    by_run.to_csv(output_paths["pos_proportions_by_run"], sep="\t", index=False)
    problematic.to_csv(output_paths["pos_problematic_metrics_by_run"], sep="\t", index=False)

    plot_global_pos_distribution(
        proportions,
        output_paths["pos_distribution"],
        title_prefix=title_prefix,
    )
    plot_pos_heatmap_by_run(
        heatmap_table,
        output_paths["pos_heatmap_by_run"],
        title_prefix=title_prefix,
    )
    plot_problematic_token_summary(
        problematic,
        output_paths["pos_problematic_tokens"],
        title_prefix=title_prefix,
    )

    return output_paths
