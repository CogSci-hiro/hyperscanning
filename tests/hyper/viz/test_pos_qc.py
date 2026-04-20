"""Tests for lightweight POS QC summaries and figure writing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hyper.viz.pos_qc import (
    RUN_GROUPING_SUBJECT_RUN,
    compute_global_pos_counts,
    compute_pos_proportions_by_run,
    compute_problematic_metrics_by_run,
    load_pos_qc_dataset,
    write_pos_qc_outputs,
)


def _write_pos_tsv(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a tiny POS TSV fixture."""
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_load_pos_qc_dataset_accepts_export_like_tsv_and_infers_file_metadata(tmp_path: Path) -> None:
    """Valid POS exports should load and infer subject/run identifiers from filename."""
    input_path = _write_pos_tsv(
        tmp_path / "sub-001_task-conversation_run-1_desc-self_pos_features.tsv",
        [
            {"run": "1", "speaker": "A", "token": "Bonjour", "upos": "INTJ", "mapping_status": "exact"},
            {"run": "1", "speaker": "A", "token": "monde", "upos": "NOUN", "mapping_status": "exact"},
        ],
    )

    dataset = load_pos_qc_dataset([input_path])

    assert dataset.schema.token_text_column == "token"
    assert dataset.data["subject_id"].tolist() == ["sub-001", "sub-001"]
    assert dataset.data["run_id"].tolist() == ["1", "1"]
    assert dataset.data["run_unit"].tolist() == ["sub-001_run-1", "sub-001_run-1"]


def test_load_pos_qc_dataset_fails_helpfully_when_required_columns_are_missing(tmp_path: Path) -> None:
    """Missing exported POS columns should raise a clear error."""
    input_path = _write_pos_tsv(
        tmp_path / "broken.tsv",
        [
            {"run": "1", "speaker": "A", "token": "Bonjour"},
        ],
    )

    with pytest.raises(ValueError, match="missing required POS QC columns: upos"):
        load_pos_qc_dataset([input_path])


def test_summary_tables_include_expected_shapes_and_columns(tmp_path: Path) -> None:
    """Summary helpers should retain run identity and emit stable key columns."""
    input_a = _write_pos_tsv(
        tmp_path / "sub-001_task-conversation_run-1_desc-self_pos_features.tsv",
        [
            {"run": "1", "token": "Bonjour", "upos": "INTJ"},
            {"run": "1", "token": "monde", "upos": "NOUN"},
            {"run": "1", "token": "!", "upos": "PUNCT"},
        ],
    )
    input_b = _write_pos_tsv(
        tmp_path / "sub-002_task-conversation_run-2_desc-self_pos_features.tsv",
        [
            {"run": "2", "token": "", "upos": None},
            {"run": "2", "token": "123", "upos": "SYM"},
            {"run": "2", "token": "euh", "upos": "X"},
        ],
    )

    dataset = load_pos_qc_dataset([input_a, input_b], grouping=RUN_GROUPING_SUBJECT_RUN)
    counts = compute_global_pos_counts(dataset.data)
    by_run = compute_pos_proportions_by_run(dataset.data)
    problematic = compute_problematic_metrics_by_run(dataset.data)

    assert set(counts.columns) == {"upos", "token_count"}
    assert set(by_run.columns) == {"run_unit", "subject_id", "run_id", "upos", "token_count", "token_proportion"}
    assert "missing_pos_rate" in problematic.columns
    assert "punctuation_rate" in problematic.columns
    assert problematic.shape[0] == 2


def test_write_pos_qc_outputs_creates_expected_files(tmp_path: Path) -> None:
    """Writing QC outputs should create the standard PNG and TSV artifacts."""
    input_path = _write_pos_tsv(
        tmp_path / "sub-001_task-conversation_run-1_desc-self_pos_features.tsv",
        [
            {"run": "1", "token": "Bonjour", "upos": "INTJ"},
            {"run": "1", "token": "monde", "upos": "NOUN"},
            {"run": "1", "token": "!", "upos": "PUNCT"},
        ],
    )
    dataset = load_pos_qc_dataset([input_path])

    output_paths = write_pos_qc_outputs(dataset, tmp_path / "qc" / "pos", title_prefix="POS QC")

    expected_names = {
        "pos_distribution",
        "pos_heatmap_by_run",
        "pos_problematic_tokens",
        "pos_counts",
        "pos_proportions",
        "pos_proportions_by_run",
        "pos_problematic_metrics_by_run",
    }
    assert set(output_paths) == expected_names
    for output_path in output_paths.values():
        assert output_path.exists()
