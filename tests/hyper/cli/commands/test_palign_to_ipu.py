"""Tests for palign-to-ipu CLI command wrapper and helpers."""

import argparse
from pathlib import Path

import pytest

from hyper.annotations.palign_core import Interval
from hyper.cli.commands import palign_to_ipu as mod


def test_parse_bool_accepts_standard_true_false_values() -> None:
    """Boolean parser should support common string variants."""
    assert mod._parse_bool("true")
    assert mod._parse_bool("YES")
    assert not mod._parse_bool("0")


def test_parse_bool_rejects_unknown_values() -> None:
    """Invalid values should fail clearly for user feedback."""
    with pytest.raises(ValueError, match="Invalid boolean"):
        mod._parse_bool("maybe")


def test_parse_args_requires_existing_input(tmp_path: Path) -> None:
    """Input TextGrid must exist before command execution."""
    args = argparse.Namespace(
        in_textgrid=tmp_path / "missing.TextGrid",
        out_textgrid=tmp_path / "out.TextGrid",
        tokens_tier="TokensAlign",
        include_laughter=False,
        include_noise=False,
        include_filled_pause="true",
        min_ipu=0.01,
        min_silence=0.2,
    )

    with pytest.raises(FileNotFoundError, match="Input TextGrid not found"):
        mod._parse_args(args)


def test_run_executes_pipeline_steps(monkeypatch, tmp_path: Path) -> None:
    """Run should call read -> segment -> render -> write in order."""
    in_tg = tmp_path / "in.TextGrid"
    out_tg = tmp_path / "out.TextGrid"
    in_tg.write_text("dummy", encoding="utf-8")

    calls = []

    monkeypatch.setattr(mod, "_read_tokensalign_intervals", lambda **k: (0.0, 1.0, [Interval(0.0, 0.2, "a")]))
    monkeypatch.setattr(mod, "build_ipu_segments_from_tokens", lambda *a, **k: [(0.0, 0.2)])
    monkeypatch.setattr(
        mod,
        "apply_min_ipu_and_render_full_tier",
        lambda **k: [Interval(0.0, 0.2, "IPU"), Interval(0.2, 1.0, "#")],
    )
    monkeypatch.setattr(mod, "_write_ipu_textgrid", lambda **k: calls.append(("write", k["out_path"])))

    args = argparse.Namespace(
        in_textgrid=in_tg,
        out_textgrid=out_tg,
        tokens_tier="TokensAlign",
        include_laughter=False,
        include_noise=False,
        include_filled_pause="true",
        min_ipu=0.01,
        min_silence=0.2,
    )

    rc = mod.run(args)

    assert rc == 0
    assert calls == [("write", out_tg.resolve())]
