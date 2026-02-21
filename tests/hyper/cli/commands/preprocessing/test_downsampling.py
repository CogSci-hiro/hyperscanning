"""Wiring tests for `hyper.cli.commands.preprocessing.downsampling`."""

import argparse
from pathlib import Path

from hyper.cli.commands.preprocessing import downsampling as mod


def test_run_forwards_arguments(monkeypatch, tmp_path: Path) -> None:
    """Downsample runner should pass expected fields to preprocessing API."""
    captured = {}
    monkeypatch.setattr(mod, "downsample_edf_to_fif", lambda **k: captured.update(k))

    args = argparse.Namespace(
        in_edf=tmp_path / "in.edf",
        channels=tmp_path / "channels.tsv",
        target_sfreq=256.0,
        out=tmp_path / "out" / "raw.fif",
        preload=False,
    )
    mod.run(args, cfg={})

    assert captured["input_edf_path"] == args.in_edf
    assert captured["target_sfreq_hz"] == 256.0


def test_add_subparser_registers_command() -> None:
    """Subparser registration should succeed with argparse registry."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mod.add_subparser(subparsers)
