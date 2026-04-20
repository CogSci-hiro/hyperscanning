"""Wiring tests for `hyper.cli.commands.speech_artefact_qc`."""

import argparse
from pathlib import Path

from hyper.cli.commands import speech_artefact_qc as mod


def test_run_forwards_arguments(monkeypatch, tmp_path: Path) -> None:
    """Speech artefact runner should pass inputs and output through unchanged."""
    captured = {}
    monkeypatch.setattr(mod, "build_speech_artefact_summary_figure", lambda **k: captured.update(k))

    args = argparse.Namespace(
        filtered_noica_inputs=[tmp_path / "noica_a.fif"],
        filtered_inputs=[tmp_path / "ica_a.fif"],
        ica_inputs=[tmp_path / "ica_solution.fif"],
        out_fig=tmp_path / "reports" / "speech_artefact_summary.png",
    )
    mod.run(args, cfg={"viz": {}})

    assert captured["filtered_noica_paths"] == [tmp_path / "noica_a.fif"]
    assert captured["filtered_paths"] == [tmp_path / "ica_a.fif"]
    assert captured["ica_paths"] == [tmp_path / "ica_solution.fif"]
    assert captured["output_path"] == args.out_fig


def test_add_subparser_registers_command() -> None:
    """Subparser registration should succeed with argparse registry."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mod.add_subparser(subparsers)
