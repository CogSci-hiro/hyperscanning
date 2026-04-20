"""Wiring tests for `hyper.cli.commands.pos_qc`."""

from __future__ import annotations

import argparse
from pathlib import Path

from hyper.cli.commands import pos_qc as mod
from hyper.config import ProjectConfig


def test_run_forwards_expected_pos_qc_arguments(tmp_path: Path, monkeypatch) -> None:
    """POS QC CLI should resolve inputs and forward normalized arguments."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(mod, "load_pos_qc_dataset", lambda *args, **kwargs: captured.setdefault("dataset", (args, kwargs)))
    monkeypatch.setattr(mod, "write_pos_qc_outputs", lambda *args, **kwargs: captured.setdefault("write", (args, kwargs)))

    input_path = tmp_path / "sub-001_task-conversation_run-1_desc-self_pos_features.tsv"
    input_path.write_text("token\tupos\nBonjour\tINTJ\n", encoding="utf-8")
    args = argparse.Namespace(
        config=tmp_path / "config.yaml",
        inputs=[input_path],
        globs=None,
        out_dir=tmp_path / "qc" / "pos",
        title_prefix="QC",
        subject_id=None,
        run_id=None,
        group_by="auto",
    )

    mod.run(args, ProjectConfig(raw={}))

    dataset_args, dataset_kwargs = captured["dataset"]
    write_args, write_kwargs = captured["write"]
    assert list(dataset_args[0]) == [input_path.resolve()]
    assert dataset_kwargs["grouping"] == "auto"
    assert write_args[1] == args.out_dir
    assert write_kwargs["title_prefix"] == "QC"


def test_add_subparser_registers_pos_qc_command() -> None:
    """POS QC CLI subparser registration should succeed."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    mod.add_subparser(subparsers)
