"""Tests for top-level CLI parser and command dispatch logic."""

from pathlib import Path

import pytest

from hyper.cli import main as mod


class _FakeCommand:
    """Simple command module replacement for dispatch tests."""

    def __init__(self) -> None:
        self.ran = False

    def add_subparser(self, subparsers) -> None:  # noqa: ANN001
        p = subparsers.add_parser("fake")
        p.add_argument("--config", type=Path, required=True)

    def run(self, args, cfg) -> None:  # noqa: ANN001
        self.ran = True
        self.args = args
        self.cfg = cfg


def test_build_arg_parser_contains_registered_subcommands() -> None:
    """Core parser should include all keys declared in command registry."""
    parser = mod.build_arg_parser()

    # Parse one valid command to verify parser wiring quickly.
    args = parser.parse_args([
        "filter",
        "--config",
        "config/config.yaml",
        "--in-fif",
        "a.fif",
        "--l-freq",
        "1",
        "--h-freq",
        "40",
        "--out",
        "b.fif",
    ])

    assert args.command == "filter"


def test_main_dispatches_to_selected_command(monkeypatch, tmp_path: Path) -> None:
    """`main(...)` should load config once and call module.run(args, cfg)."""
    fake = _FakeCommand()
    monkeypatch.setattr(mod, "_COMMANDS", {"fake": fake})
    monkeypatch.setattr(mod, "load_project_config", lambda p: {"loaded_from": p})

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("x: 1\n", encoding="utf-8")

    mod.main(["fake", "--config", str(cfg_path)])

    assert fake.ran is True
    assert fake.cfg["loaded_from"] == cfg_path


def test_build_arg_parser_rejects_missing_add_subparser(monkeypatch) -> None:
    """Registry entries without add_subparser should fail early."""
    class _Broken:
        pass

    monkeypatch.setattr(mod, "_COMMANDS", {"oops": _Broken()})

    with pytest.raises(RuntimeError, match="missing add_subparser"):
        mod.build_arg_parser()
