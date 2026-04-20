"""CLI command for subject-level TRF alpha QC figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.trf.qc import build_subject_alpha_qc_manifest


def add_subparser(subparsers: Any) -> None:
    """Register the `trf-alpha-qc` subcommand."""
    parser = subparsers.add_parser(
        "trf-alpha-qc",
        help="Render per-subject alpha-sweep TRF QC figures.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--task", type=str, default="conversation", help="Task label. Defaults to conversation.")
    parser.add_argument("--manifest", type=Path, required=True, help="Output JSON manifest path.")


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `trf-alpha-qc` command."""
    build_subject_alpha_qc_manifest(
        cfg=cfg,
        task=str(args.task),
        manifest_path=Path(args.manifest),
    )
