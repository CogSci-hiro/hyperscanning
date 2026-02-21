"""Wiring tests for `hyper.cli.commands.preprocessing.metadata`."""

import argparse
from pathlib import Path

from hyper.cli.commands.preprocessing import metadata as mod


def test_run_forwards_arguments(monkeypatch, tmp_path: Path) -> None:
    """Metadata runner should pass lock/anchor/margin controls to core API."""
    captured = {}
    monkeypatch.setattr(mod, "make_metadata_and_events_from_self_ipu", lambda **k: captured.update(k))

    args = argparse.Namespace(
        ipu=tmp_path / "sub-001_run-1_ipu.csv",
        raw=tmp_path / "raw.fif",
        out_tsv=tmp_path / "out" / "meta.tsv",
        out_events=tmp_path / "out" / "events.npy",
        margin=0.75,
        time_lock="offset",
        anchor="other",
    )
    mod.run(args, cfg={})

    assert captured["margin_s"] == 0.75
    assert captured["time_lock"] == "offset"
    assert captured["anchor"] == "other"


def test_add_subparser_registers_command() -> None:
    """Subparser registration should succeed with argparse registry."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mod.add_subparser(subparsers)
