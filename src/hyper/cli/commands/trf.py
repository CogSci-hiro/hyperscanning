"""CLI command for subject-level TRF benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.trf.pipeline import run_trf_pipeline


def add_subparser(subparsers: Any) -> None:
    """Register the `trf` subcommand."""
    parser = subparsers.add_parser(
        "trf",
        help="Run the subject-level TRF benchmark pipeline.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--subject", type=str, required=True, help="Subject identifier, e.g. sub-006.")
    parser.add_argument("--task", type=str, default="conversation", help="Task label. Defaults to conversation.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to results/trf/<subject>/task-<task>/.",
    )


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `trf` command."""
    run_trf_pipeline(
        cfg=cfg,
        subject_id=str(args.subject),
        task=str(args.task),
        out_dir=args.out_dir,
    )
