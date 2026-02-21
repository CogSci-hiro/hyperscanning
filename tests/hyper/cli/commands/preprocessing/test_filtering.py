"""Wiring tests for `hyper.cli.commands.preprocessing.filtering`."""

import argparse
from pathlib import Path

from hyper.cli.commands.preprocessing import filtering as mod


def test_run_forwards_arguments(monkeypatch, tmp_path: Path) -> None:
    """Filter runner should map cutoff frequencies as floats."""
    captured = {}
    monkeypatch.setattr(mod, "bandpass_filter_fif_to_fif", lambda **k: captured.update(k))

    args = argparse.Namespace(
        in_fif=tmp_path / "in.fif",
        l_freq=1.0,
        h_freq=30.0,
        out=tmp_path / "out" / "out.fif",
        preload=True,
    )
    mod.run(args, cfg={})

    assert captured["l_freq_hz"] == 1.0
    assert captured["h_freq_hz"] == 30.0


def test_add_subparser_registers_command() -> None:
    """Subparser registration should succeed with argparse registry."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mod.add_subparser(subparsers)
