"""CLI command for TRF kernel QC figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.trf.qc import build_group_average_trf_kernel_manifest


def add_subparser(subparsers: Any) -> None:
    """Register the `trf-kernel-qc` subcommand."""
    parser = subparsers.add_parser(
        "trf-kernel-qc",
        help="Render group-average TRF kernel joint plots per predictor.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--task", type=str, default="conversation", help="Task label. Defaults to conversation.")
    parser.add_argument("--manifest", type=Path, required=True, help="Output JSON manifest path.")


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `trf-kernel-qc` command."""
    build_group_average_trf_kernel_manifest(
        cfg=cfg,
        task=str(args.task),
        manifest_path=Path(args.manifest),
    )
