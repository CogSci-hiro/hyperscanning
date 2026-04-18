"""Wiring tests for `hyper.cli.commands.features.pitch`."""

from __future__ import annotations

import argparse
from pathlib import Path

from hyper.cli.commands.features import pitch as mod


def test_run_forwards_expected_pitch_arguments(monkeypatch, tmp_path: Path) -> None:
    """Pitch CLI should forward the expected paths and numeric settings."""
    captured = {}
    monkeypatch.setattr(mod, "run_pitch_pipeline", lambda **kwargs: captured.update(kwargs))

    args = argparse.Namespace(
        audio=tmp_path / "audio.wav",
        eeg_sfreq=512.0,
        eeg_samples=1024,
        out=tmp_path / "f0.npy",
        out_sidecar=tmp_path / "f0.json",
        fmin=75.0,
        fmax=300.0,
        frame_length=0.04,
        frame_step=0.01,
        fill_strategy="linear",
    )

    mod.run(args, cfg={})

    assert captured["audio_path"] == args.audio
    assert captured["eeg_sample_count"] == 1024
    assert captured["config"].fill_strategy == "linear"


def test_add_subparser_registers_pitch_command() -> None:
    """Pitch CLI subparser registration should succeed."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    mod.add_subparser(subparsers)
