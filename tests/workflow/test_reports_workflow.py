"""Smoke tests for report-figure Snakemake wiring."""

from __future__ import annotations

from pathlib import Path


def test_speech_artefact_rule_is_wired_into_reports_workflow() -> None:
    """The reports workflow should expose the speech artefact summary figure rule."""
    text = Path("workflow/rules/reports.smk").read_text(encoding="utf-8")

    assert "rule speech_artefact_figure:" in text
    assert '{HYPER_MODULE_CMD} speech-artefact-qc \\' in text
    assert 'reports_path("figures", str(VIZ.get("speech_artefact", {}).get("filename", "speech_artefact_summary.png")))' in text


def test_trf_score_qc_figure_rule_is_wired_into_reports_workflow() -> None:
    """The reports workflow should expose the TRF score QC figure rule."""
    text = Path("workflow/rules/reports.smk").read_text(encoding="utf-8")

    assert "rule trf_score_qc_figure:" in text
    assert '{HYPER_MODULE_CMD} trf-score-qc-figure \\' in text
    assert 'out_path("trf_qc", f"task-{_trf_score_task()}", "eeg_scores.tsv")' in text
    assert 'reports_path("figures", str(VIZ.get("trf_score", {}).get("filename", "trf_score_summary.png")))' in text


def test_ipu_turn_taking_figure_rule_is_wired_into_reports_workflow() -> None:
    """The reports workflow should expose the IPU turn-taking figure rule."""
    text = Path("workflow/rules/reports.smk").read_text(encoding="utf-8")

    assert "rule ipu_turn_taking_figure:" in text
    assert '{HYPER_MODULE_CMD} ipu-turn-taking-figure \\' in text
    assert 'reports_path("figures", str(VIZ.get("ipu_turn_taking", {}).get("filename", "ipu_turn_taking_summary.png")))' in text


def test_trf_main_figure_rule_is_wired_into_reports_workflow() -> None:
    """The reports workflow should expose the TRF main figure rule."""
    text = Path("workflow/rules/reports.smk").read_text(encoding="utf-8")

    assert "rule trf_main_figure:" in text
    assert '{HYPER_MODULE_CMD} trf-main-figure \\' in text
    assert 'fig=reports_path("figures", _trf_main_figure_filename())' in text


def test_main_figures_target_is_exposed() -> None:
    """targets.smk should expose the aggregate main figures target."""
    text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "rule main_figures_all:" in text
    assert 'reports_path("figures", str(VIZ.get("speech_artefact", {}).get("filename", "speech_artefact_summary.png")))' in text
    assert 'reports_path("figures", str(VIZ.get("trf_score", {}).get("filename", "trf_score_summary.png")))' in text
    assert 'reports_path("figures", str(VIZ.get("trf_main_figure", {}).get("filename", "trf_main_figure_summary.png")))' in text
    assert 'reports_path("figures", str(VIZ.get("ipu_turn_taking", {}).get("filename", "ipu_turn_taking_summary.png")))' in text
