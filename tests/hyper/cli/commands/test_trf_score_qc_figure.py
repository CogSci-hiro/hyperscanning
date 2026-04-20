"""Wiring tests for `hyper.cli.commands.trf_score_qc_figure`."""

import argparse
from pathlib import Path

from hyper.cli.commands import trf_score_qc_figure as mod


def test_run_forwards_arguments(monkeypatch, tmp_path: Path) -> None:
    """TRF-score figure runner should pass inputs and output through unchanged."""
    captured = {}
    monkeypatch.setattr(mod, "build_trf_score_qc_figure", lambda **k: captured.update(k))

    args = argparse.Namespace(
        eeg_table=tmp_path / "eeg_scores.tsv",
        feature_table=tmp_path / "feature_scores.tsv",
        out_fig=tmp_path / "trf_score_summary.png",
    )
    mod.run(args, cfg={"viz": {}})

    assert captured["eeg_table_path"] == args.eeg_table
    assert captured["feature_table_path"] == args.feature_table
    assert captured["output_path"] == args.out_fig


def test_add_subparser_registers_command() -> None:
    """Subparser registration should succeed with argparse registry."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mod.add_subparser(subparsers)
