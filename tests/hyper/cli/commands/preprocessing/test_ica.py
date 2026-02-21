"""Wiring tests for `hyper.cli.commands.preprocessing.ica`."""

import argparse
from pathlib import Path

from hyper.cli.commands.preprocessing import ica as mod


def test_run_forwards_arguments(monkeypatch, tmp_path: Path) -> None:
    """ICA runner should pass raw/ica/output paths through unchanged."""
    captured = {}
    monkeypatch.setattr(mod, "apply_ica_fif_to_fif", lambda **k: captured.update(k))

    args = argparse.Namespace(
        in_fif=tmp_path / "in.fif",
        ica=tmp_path / "ica.fif",
        out=tmp_path / "out" / "out.fif",
        preload=True,
    )
    mod.run(args, cfg={})

    assert captured["ica_path"] == args.ica
    assert captured["output_fif_path"] == args.out


def test_add_subparser_registers_command() -> None:
    """Subparser registration should succeed with argparse registry."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mod.add_subparser(subparsers)
