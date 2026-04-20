"""CLI command for IPU turn-taking summary figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.viz.ipu_turn_taking import build_ipu_turn_taking_figure


def add_subparser(subparsers: Any) -> None:
    """Register the `ipu-turn-taking-figure` subcommand."""
    parser = subparsers.add_parser(
        "ipu-turn-taking-figure",
        help="Render IPU turn-taking summary panels from dyad annotation CSVs.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--out-fig", type=Path, required=True, help="Output figure path.")


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `ipu-turn-taking-figure` command."""
    build_ipu_turn_taking_figure(
        cfg=cfg,
        output_path=Path(args.out_fig),
    )
