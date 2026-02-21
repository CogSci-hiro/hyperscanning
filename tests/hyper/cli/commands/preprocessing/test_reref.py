"""Wiring tests for `hyper.cli.commands.preprocessing.reref`."""

import argparse
from pathlib import Path

from hyper.cli.commands.preprocessing import reref as mod


def test_run_forwards_arguments(monkeypatch, tmp_path: Path) -> None:
    """Reref runner should map CLI args to preprocessing API arguments."""
    captured = {}
    monkeypatch.setattr(mod, "rereference_fif_to_fif", lambda **k: captured.update(k))

    args = argparse.Namespace(
        in_fif=tmp_path / "in.fif",
        channels=tmp_path / "channels.tsv",
        out=tmp_path / "out" / "out.fif",
        preload=True,
    )
    mod.run(args, cfg={})

    assert captured["input_fif_path"] == args.in_fif
    assert captured["channels_tsv_path"] == args.channels


def test_add_subparser_registers_command() -> None:
    """Subparser registration should succeed with argparse registry."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mod.add_subparser(subparsers)
