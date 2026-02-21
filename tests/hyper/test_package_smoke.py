"""Import smoke tests for lightweight modules and package entrypoints.

The goal here is coverage for very small modules where behavior is mostly
presence/shape rather than runtime logic.
"""

from hyper import annotations, cli, preprocessing, viz
from hyper.cli.types import CliCommand
from hyper.viz.style import Style


def test_packages_import() -> None:
    """Top-level package namespaces should import without side effects."""
    assert annotations is not None
    assert cli is not None
    assert preprocessing is not None
    assert viz is not None


def test_style_defaults() -> None:
    """Style dataclass should expose documented defaults."""
    style = Style()
    assert style.fontsize == 12


def test_cli_protocol_exposes_expected_methods() -> None:
    """Protocol must define the command contract used by CLI registry."""
    assert hasattr(CliCommand, "add_subparser")
    assert hasattr(CliCommand, "run")
