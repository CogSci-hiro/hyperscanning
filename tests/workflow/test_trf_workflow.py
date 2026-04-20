"""Smoke tests for TRF Snakemake wiring."""

from __future__ import annotations

from pathlib import Path


def test_trf_qc_score_table_rule_is_wired_into_features_workflow() -> None:
    """The workflow should expose an explicit rule for TRF QC score tables."""
    text = Path("workflow/rules/features.smk").read_text(encoding="utf-8")

    assert "rule trf_qc_score_tables:" in text
    assert 'out_path("trf_qc", "task-{task}", "eeg_scores.tsv")' in text
    assert 'out_path("trf_qc", "task-{task}", "feature_scores.tsv")' in text
    assert "{HYPER_MODULE_CMD} trf-score-qc \\" in text


def test_trf_qc_score_table_targets_are_explicit() -> None:
    """targets.smk should list the new QC score tables explicitly."""
    text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "rule qc_trf_score_tables_all:" in text
    assert 'out_path("trf_qc", f"task-{task}", "eeg_scores.tsv")' in text
    assert 'out_path("trf_qc", f"task-{task}", "feature_scores.tsv")' in text
