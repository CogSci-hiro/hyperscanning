"""CLI command for subject-level TRF QC score tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.trf.pipeline import run_trf_qc_score_tables


def add_subparser(subparsers: Any) -> None:
    """Register the `trf-score-qc` subcommand."""
    parser = subparsers.add_parser(
        "trf-score-qc",
        help="Compute subject-level EEG and feature TRF QC score tables.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--task", type=str, default="conversation", help="Task label. Defaults to conversation.")
    parser.add_argument("--eeg-out", type=Path, required=True, help="Output TSV path for EEG QC scores.")
    parser.add_argument("--feature-out", type=Path, required=True, help="Output TSV path for feature QC scores.")


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `trf-score-qc` command."""
    run_trf_qc_score_tables(
        cfg=cfg,
        task=str(args.task),
        eeg_output_path=Path(args.eeg_out),
        feature_output_path=Path(args.feature_out),
    )
