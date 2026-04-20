"""Smoke tests for POS QC Snakemake wiring."""

from __future__ import annotations

from pathlib import Path


def test_pos_qc_rule_is_wired_into_features_workflow() -> None:
    """The workflow should expose an explicit rule for POS QC outputs."""
    text = Path("workflow/rules/features.smk").read_text(encoding="utf-8")

    assert "rule pos_qc:" in text
    assert 'out_path("qc", "pos", "pos_distribution.png")' in text
    assert 'out_path("qc", "pos", "pos_problematic_metrics_by_run.tsv")' in text
    assert "{HYPER_MODULE_CMD} pos-qc \\" in text


def test_pos_qc_targets_are_explicit() -> None:
    """targets.smk should list the POS QC outputs explicitly."""
    text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "rule qc_pos_all:" in text
    assert 'out_path("qc", "pos", "pos_distribution.png")' in text
    assert 'out_path("qc", "pos", "pos_problematic_metrics_by_run.tsv")' in text
