"""Wiring tests for `hyper.cli.commands.preprocessing.epoching`."""

import argparse
from pathlib import Path

from hyper.cli.commands.preprocessing import epoching as mod


def test_run_converts_baseline_and_detrend(monkeypatch, tmp_path: Path) -> None:
    """Epoch runner should convert CLI flags into API baseline/detrend values."""
    captured = {}
    monkeypatch.setattr(mod, "make_epochs_fif_to_fif", lambda **k: captured.update(k))

    args = argparse.Namespace(
        raw=tmp_path / "raw.fif",
        events=tmp_path / "events.npy",
        metadata=tmp_path / "meta.tsv",
        out=tmp_path / "out" / "epo.fif",
        tmin=-0.2,
        tmax=0.8,
        baseline=False,
        baseline_start=-0.2,
        baseline_end=0.0,
        detrend=-1,
        preload_raw=False,
    )
    mod.run(args, cfg={})

    assert captured["baseline"] is None
    assert captured["detrend"] is None


def test_add_subparser_registers_command() -> None:
    """Subparser registration should succeed with argparse registry."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mod.add_subparser(subparsers)
