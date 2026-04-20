"""Smoke tests for persistent pre-bandpass preprocessing outputs."""

from __future__ import annotations

from pathlib import Path


def test_interpolated_outputs_are_persistent_prebandpass_endpoints() -> None:
    """Interpolated FIFs should no longer be marked temporary."""
    text = Path("workflow/rules/preprocessing.smk").read_text(encoding="utf-8")

    assert 'raw_interp=out_path("eeg", "interpolated", "{subject}_task-{task}_run-{run}_raw_interp.fif")' in text
    assert 'raw_interp=out_path("eeg", "interpolated_noica", "{subject}_task-{task}_run-{run}_raw_interp_noica.fif")' in text
    assert 'raw_interp=maybe_temp(out_path("eeg", "interpolated", "{subject}_task-{task}_run-{run}_raw_interp.fif"))' not in text
    assert 'raw_interp=maybe_temp(out_path("eeg", "interpolated_noica", "{subject}_task-{task}_run-{run}_raw_interp_noica.fif"))' not in text


def test_preprocessing_workflow_documents_bandpass_as_downstream_step() -> None:
    """Workflow comments should describe pre-bandpass data as the persistent endpoint."""
    text = Path("workflow/rules/preprocessing.smk").read_text(encoding="utf-8")

    assert "persistent pre-bandpass endpoint" in text
    assert "additional band-pass step for TRF/downstream modeling" in text
