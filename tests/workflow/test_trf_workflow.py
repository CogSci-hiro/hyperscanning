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


def test_speech_envelope_rule_uses_ipu_masking_inputs() -> None:
    """Envelope extraction should pass subject-specific IPU CSVs into the CLI."""
    text = Path("workflow/rules/features.smk").read_text(encoding="utf-8")

    assert 'self_ipu=annotation_path(config["annotations"]["ipu"], "{subject}_run-{run}_ipu.csv")' in text
    assert "--ipu {input.self_ipu} \\" in text
    assert "--ipu {input.other_ipu} \\" in text
