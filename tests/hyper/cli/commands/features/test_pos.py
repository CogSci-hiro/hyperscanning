"""Wiring tests for `hyper.cli.commands.features.pos`."""

from __future__ import annotations

import argparse
from pathlib import Path

from hyper.cli.commands.features import pos as mod
from hyper.config import ProjectConfig


def test_run_forwards_expected_pos_arguments(tmp_path: Path, monkeypatch) -> None:
    """POS CLI should resolve config defaults and forward paths and flags."""
    captured = {}
    monkeypatch.setattr(mod, "run_token_pos_pipeline", lambda **kwargs: captured.update(kwargs))

    args = argparse.Namespace(
        tokens=tmp_path / "tokens.csv",
        subject="sub-001",
        run="1",
        out_tsv=tmp_path / "pos.tsv",
        out_sidecar=tmp_path / "pos.json",
        feature_name="self_pos",
        source_subject="sub-001",
        source_role="self",
        language=None,
        processors=None,
        resources_dir=None,
        allow_download=None,
        fail_on_mapping_error=False,
        exclude_label=["#", "*"],
        show_progress=None,
    )
    cfg = ProjectConfig(
        raw={
            "features": {
                "stanza_pos": {
                    "language": "fr",
                    "processors": "tokenize,pos,lemma",
                    "allow_download": False,
                }
            }
        }
    )

    mod.run(args, cfg)

    assert captured["tokens_path"] == args.tokens
    assert captured["feature_name"] == "self_pos"
    assert captured["exclude_labels"] == ("#", "*")
    assert captured["config"].language == "fr"
    assert captured["config"].allow_download is False
    assert captured["show_progress"] is True


def test_add_subparser_registers_pos_command() -> None:
    """POS CLI subparser registration should succeed."""
    parser = argparse.ArgumentParser(prog="hyper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    mod.add_subparser(subparsers)
