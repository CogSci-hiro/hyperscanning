"""Wiring tests for `hyper.cli.commands.preprocessing.interpolation`."""

import argparse
from pathlib import Path

from hyper.cli.commands.preprocessing import interpolation as mod


def test_run_forwards_arguments(monkeypatch, tmp_path: Path) -> None:
    """Interpolation runner should include method and preload flags."""
    captured = {}
    monkeypatch.setattr(mod, "interpolate_bads_fif_to_fif", lambda **k: captured.update(k))

    args = argparse.Namespace(
        in_fif=tmp_path / "in.fif",
        channels=tmp_path / "channels.tsv",
        method="nearest",
        out=tmp_path / "out" / "out.fif",
        preload=False,
    )
    mod.run(args, cfg={})

    assert captured["method"] == "nearest"
    assert captured["preload"] is False


def test_add_subparser_registers_command() -> None:
    """Subparser registration should succeed with argparse registry."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mod.add_subparser(subparsers)
