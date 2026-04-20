"""Smoke tests for the no-ICA preprocessing branch."""

from __future__ import annotations

from pathlib import Path


def test_noica_rules_are_wired_into_preprocessing_workflow() -> None:
    """The workflow should expose explicit no-ICA preprocessing rules."""
    text = Path("workflow/rules/preprocessing.smk").read_text(encoding="utf-8")

    assert "rule interpolate_noica:" in text
    assert "rule filter_raw_noica:" in text
    assert "rule metadata_noica:" in text
    assert "rule epoch_noica:" in text
    assert 'out_path("eeg", "filtered_noica", "{subject}_task-{task}_run-{run}_raw_filt_noica.fif")' in text


def test_noica_targets_are_exposed() -> None:
    """targets.smk should expose aggregate and canary no-ICA targets."""
    text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "rule filter_noica_all:" in text
    assert "rule metadata_noica_all:" in text
    assert "rule epoch_noica_all:" in text
    assert "rule canary_preprocessing_noica:" in text
    assert 'out_path("canary", "all_noica.done")' in text
