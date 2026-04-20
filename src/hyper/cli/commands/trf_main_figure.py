"""CLI command for the TRF main joint-panel figure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.viz.trf_main_figure import build_trf_main_figure


def add_subparser(subparsers: Any) -> None:
    """Register the `trf-main-figure` subcommand."""
    parser = subparsers.add_parser(
        "trf-main-figure",
        help="Render a multi-panel TRF joint summary figure from group-average kernels.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--out-fig", type=Path, required=True, help="Output figure path.")


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `trf-main-figure` command."""
    build_trf_main_figure(
        cfg=cfg,
        output_path=Path(args.out_fig),
    )
