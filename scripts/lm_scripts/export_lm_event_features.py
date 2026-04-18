#!/usr/bin/env python3
"""Export dyad-level LM token uncertainty tables as event-like feature derivatives."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


JSON_INDENT_SPACES = 2
DEFAULT_TASK = "conversation"
DEFAULT_SESSION = "ses-01"
MISSING_VALUE = "n/a"
PRIMARY_INPUT_NAME = "token_uncertainty.csv"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Configuration for one exported LM scalar derivative."""

    name: str
    column_name: str
    source_column: str
    descriptor: str
    output_dirname: str
    description: str


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="surprisal",
        column_name="surprisal",
        source_column="token_surprisal_nats",
        descriptor="lmSurprisal",
        output_dirname="lm_surprisal",
        description="Token-level language-model surprisal derived from per-dyad token_uncertainty.csv.",
    ),
    MetricSpec(
        name="entropy",
        column_name="entropy",
        source_column="token_shannon_entropy_nats",
        descriptor="lmShannonEntropy",
        output_dirname="lm_shannon_entropy",
        description="Token-level Shannon entropy derived from per-dyad token_uncertainty.csv.",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert Runpod dyad token uncertainty outputs into event-style TSV+JSON feature derivatives."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Root directory containing dyad-* subdirectories with token_uncertainty.csv files.",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Derived-data output root. Event features are written under out_dir/features/events/...",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help=f"Task label used in output filenames. Default: {DEFAULT_TASK}.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=DEFAULT_SESSION,
        help=f"Session label used in output filenames. Default: {DEFAULT_SESSION}.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=JSON_INDENT_SPACES, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _coerce_run_label(value: object) -> str:
    """Return a stable run label for filenames and filtering."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _parse_dyad_index(dyad_dir: Path) -> int:
    """Parse integer dyad index from a dyad-### directory name."""
    prefix = "dyad-"
    if not dyad_dir.name.startswith(prefix):
        raise ValueError(f"Dyad directory must start with {prefix!r}: {dyad_dir}")
    return int(dyad_dir.name.removeprefix(prefix))


def _subject_ids_for_dyad(dyad_index: int) -> dict[str, str]:
    """Map speaker labels to canonical subject identifiers for one dyad."""
    base = (dyad_index - 1) * 2 + 1
    return {
        "A": f"sub-{base:03d}",
        "B": f"sub-{base + 1:03d}",
    }


def _normalize_text(value: object) -> str:
    """Return a stripped string unless the value is missing."""
    if pd.isna(value):
        return MISSING_VALUE
    text = str(value).strip()
    return text if text else MISSING_VALUE


def _numeric_or_na(value: object) -> float | str:
    """Return a float when possible, otherwise the TSV missing-value marker."""
    if pd.isna(value):
        return MISSING_VALUE
    return float(value)


def _round_series(series: pd.Series, *, digits: int = 6) -> pd.Series:
    """Round numeric values for cleaner event-table exports."""
    return pd.to_numeric(series, errors="coerce").round(digits)


def _build_event_rows(token_df: pd.DataFrame, metric: MetricSpec, *, dyad_id: str) -> pd.DataFrame:
    """Transform token_uncertainty rows into an event-style feature table."""
    required_columns = {
        "run",
        "speaker",
        "start",
        "end",
        "token",
        "rendered_text",
        "annotation_index",
        metric.source_column,
    }
    missing_columns = sorted(required_columns.difference(token_df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Input table is missing required columns for {metric.name}: {missing_text}")

    working = token_df.copy()
    working["run"] = working["run"].map(_coerce_run_label)
    working["speaker"] = working["speaker"].astype(str).str.strip()
    working["start"] = pd.to_numeric(working["start"], errors="coerce")
    working["end"] = pd.to_numeric(working["end"], errors="coerce")
    working = working.loc[working["speaker"].isin(("A", "B"))].copy()
    working = working.loc[working["start"].notna() & working["end"].notna()].copy()
    working["duration"] = working["end"] - working["start"]
    working = working.sort_values(
        by=["run", "speaker", "start", "end", "annotation_index"],
        kind="stable",
    ).reset_index(drop=True)
    working["word_index"] = working.groupby(["run", "speaker"]).cumcount() + 1

    metric_values = pd.to_numeric(working[metric.source_column], errors="coerce")
    event_table = pd.DataFrame(
        {
            "onset": _round_series(working["start"]),
            "duration": _round_series(working["duration"]),
            "speaker": working["speaker"],
            "word": working["token"].map(_normalize_text),
            "normalized_word": working["rendered_text"].map(_normalize_text),
            "word_index": working["word_index"].astype(int),
            metric.column_name: metric_values.map(_numeric_or_na),
            "run": working["run"],
            "dyad_id": dyad_id,
            "lm_token_id": [
                f"{dyad_id}_run-{run}_speaker-{speaker}_ann-{annotation_index}"
                for run, speaker, annotation_index in zip(
                    working["run"],
                    working["speaker"],
                    working["annotation_index"],
                    strict=True,
                )
            ],
            "source_interval_id": [
                f"{dyad_id}_run-{run}_ann-{annotation_index}"
                for run, annotation_index in zip(
                    working["run"],
                    working["annotation_index"],
                    strict=True,
                )
            ],
            "alignment_status": [
                "ok" if value != MISSING_VALUE else "unmatched"
                for value in metric_values.map(_numeric_or_na)
            ],
        }
    )
    return event_table


def _sidecar_payload(
    event_table: pd.DataFrame,
    metric: MetricSpec,
    *,
    feature_file_path: str,
    input_token_path: Path,
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build an event-feature JSON sidecar."""
    missing_value_count = int((event_table == MISSING_VALUE).sum().sum())
    status_counts = {
        str(key): int(value)
        for key, value in event_table["alignment_status"].value_counts(dropna=False).items()
    }
    speaker_levels = sorted({str(value) for value in event_table["speaker"].dropna().unique()})

    model_name = run_summary.get("model_name")
    renyi_alphas = run_summary.get("renyi_alphas")
    notes = [
        "Rows originate from token_uncertainty.csv; each row represents one original SPPAS token row.",
        f"The exported {metric.column_name} values are stored in nats.",
    ]
    if isinstance(renyi_alphas, list) and renyi_alphas:
        notes.append(f"Available precomputed Renyi alphas in the source run summary: {renyi_alphas}.")

    return {
        "FeatureName": f"{metric.descriptor}_features",
        "FeatureLabel": f"LM {metric.name.title()} features",
        "FeatureType": "event",
        "Description": metric.description,
        "TimeBase": "seconds_from_run_onset",
        "Delimiter": "tab",
        "RowDefinition": "One row represents one original SPPAS token row aligned to the run timeline.",
        "Columns": {
            "onset": {
                "Description": "Token onset time relative to run onset.",
                "Units": "s",
                "DType": "float",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            },
            "duration": {
                "Description": "Token duration.",
                "Units": "s",
                "DType": "float",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            },
            "speaker": {
                "Description": "Original speaker identifier for the token row.",
                "Units": None,
                "DType": "string",
                "Levels": speaker_levels,
                "MissingValues": MISSING_VALUE,
            },
            "word": {
                "Description": "Original token text from token_uncertainty.csv.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            },
            "normalized_word": {
                "Description": "LM-facing normalized token text from rendered_text.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            },
            "word_index": {
                "Description": "One-based token index within each run and speaker stream after onset sorting.",
                "Units": None,
                "DType": "integer",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            },
            metric.column_name: {
                "Description": f"Token-level LM {metric.name} aggregated from model-token pieces.",
                "Units": "nats",
                "DType": "float",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            },
            "run": {
                "Description": "Conversation run identifier from the source dyad export.",
                "Units": None,
                "DType": "string",
                "Levels": sorted({str(value) for value in event_table["run"].dropna().unique()}),
                "MissingValues": MISSING_VALUE,
            },
            "dyad_id": {
                "Description": "Dyad directory identifier from the Runpod export root.",
                "Units": None,
                "DType": "string",
                "Levels": sorted({str(value) for value in event_table["dyad_id"].dropna().unique()}),
                "MissingValues": MISSING_VALUE,
            },
            "lm_token_id": {
                "Description": "Stable token identifier derived from dyad, run, speaker, and annotation index.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            },
            "source_interval_id": {
                "Description": "Source annotation identifier derived from the dyad annotation index.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            },
            "alignment_status": {
                "Description": "Whether the token row has a non-missing LM scalar value in the source table.",
                "Units": None,
                "DType": "string",
                "Levels": ["ok", "unmatched"],
                "MissingValues": MISSING_VALUE,
            },
        },
        "Source": {
            "Modality": "annotation",
            "InputFiles": [str(input_token_path)],
            "InputTimeBase": "seconds",
            "AnnotationSource": "Runpod LM token uncertainty export",
        },
        "Generation": {
            "SoftwareName": "hyperscanning",
            "SoftwareVersion": "0.1.0",
            "CodeURL": None,
            "ExtractionLibrary": None,
            "ExtractionLibraryVersion": None,
            "ComputationDate": datetime.now(timezone.utc).isoformat(),
        },
        "Method": {
            "Algorithm": "Direct event-table export from token_uncertainty.csv",
            "AlgorithmReference": None,
            "Parameters": {
                "MetricColumn": metric.source_column,
                "MetricUnits": "nats",
                "PrimaryInputTable": PRIMARY_INPUT_NAME,
                "ModelName": model_name,
            },
            "Preprocessing": [
                "Rows with missing onset or offset were excluded.",
                "Rows were sorted by run, speaker, onset, end, and annotation index.",
            ],
            "Postprocessing": [
                "One-based word_index values were assigned within each run and speaker stream.",
            ],
            "FilteringRules": [
                "Only speakers A and B are exported.",
            ],
            "AggregationRules": [
                "Each output row corresponds to one original SPPAS token row from token_uncertainty.csv.",
                f"{metric.column_name} is copied from {metric.source_column} without further aggregation.",
            ],
        },
        "QualityControl": {
            "NumRows": int(event_table.shape[0]),
            "NumMissingValues": missing_value_count,
            "ExcludedCount": 0,
            "CategoryCounts": {
                "alignment_status": status_counts,
            },
            "TimingSorted": bool(event_table.sort_values(["onset", "duration"]).index.is_monotonic_increasing),
        },
        "FeatureFile": {
            "Path": feature_file_path,
            "Format": "TSV",
        },
        "Notes": notes,
    }


def _write_metric_outputs(
    event_table: pd.DataFrame,
    metric: MetricSpec,
    *,
    out_dir: Path,
    task: str,
    session: str,
    input_token_path: Path,
    run_summary: dict[str, Any],
    subject_id: str,
    run_id: str,
) -> tuple[Path, Path]:
    """Write one TSV+JSON pair for a subject/run/metric slice."""
    run_stem = f"{subject_id}_{session}_task-{task}_run-{run_id}"
    output_root = out_dir / "features" / "events" / metric.output_dirname
    tsv_path = output_root / f"{run_stem}_desc-{metric.descriptor}_features.tsv"
    json_path = output_root / f"{run_stem}_desc-{metric.descriptor}_features.json"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    event_table.to_csv(tsv_path, sep="\t", index=False, na_rep=MISSING_VALUE)
    sidecar = _sidecar_payload(
        event_table,
        metric,
        feature_file_path=tsv_path.name,
        input_token_path=input_token_path,
        run_summary=run_summary,
    )
    _write_json(json_path, sidecar)
    return tsv_path, json_path


def export_dyad_root(input_root: Path, out_dir: Path, *, task: str, session: str) -> list[tuple[Path, Path]]:
    """Export all dyad directories found under the input root."""
    written_paths: list[tuple[Path, Path]] = []
    dyad_dirs = sorted(path for path in input_root.glob("dyad-*") if path.is_dir())
    if not dyad_dirs:
        raise FileNotFoundError(f"No dyad-* directories found under {input_root}")

    for dyad_dir in dyad_dirs:
        token_path = dyad_dir / PRIMARY_INPUT_NAME
        if not token_path.exists():
            raise FileNotFoundError(f"Required input file not found: {token_path}")

        run_summary_path = dyad_dir / "run_summary.json"
        run_summary = {}
        if run_summary_path.exists():
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))

        dyad_index = _parse_dyad_index(dyad_dir)
        subject_map = _subject_ids_for_dyad(dyad_index)
        token_df = pd.read_csv(token_path)

        for metric in METRIC_SPECS:
            event_table = _build_event_rows(token_df, metric, dyad_id=dyad_dir.name)
            if event_table.empty:
                continue

            for (run_id, speaker), subset in event_table.groupby(["run", "speaker"], sort=True):
                resolved_subject = subject_map.get(str(speaker))
                if resolved_subject is None:
                    continue
                written_paths.append(
                    _write_metric_outputs(
                        subset.reset_index(drop=True),
                        metric,
                        out_dir=out_dir,
                        task=task,
                        session=session,
                        input_token_path=token_path,
                        run_summary=run_summary,
                        subject_id=resolved_subject,
                        run_id=str(run_id),
                    )
                )

    return written_paths


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    written_paths = export_dyad_root(
        input_root=args.input_root,
        out_dir=args.out_dir,
        task=str(args.task),
        session=str(args.session),
    )
    print(f"Wrote {len(written_paths)} feature file pairs.")


if __name__ == "__main__":
    main()
