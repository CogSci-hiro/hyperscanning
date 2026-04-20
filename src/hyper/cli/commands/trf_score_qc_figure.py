"""CLI command for TRF QC score violin figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.viz.trf_score_qc import build_trf_score_qc_figure


def add_subparser(subparsers: Any) -> None:
    """Register the `trf-score-qc-figure` subcommand."""
    parser = subparsers.add_parser(
        "trf-score-qc-figure",
        help="Render violin plots from TRF EEG and feature QC score tables.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--eeg-table", type=Path, required=True, help="Input TSV path for EEG QC scores.")
    parser.add_argument("--feature-table", type=Path, required=True, help="Input TSV path for feature QC scores.")
    parser.add_argument("--out-fig", type=Path, required=True, help="Output figure path.")


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `trf-score-qc-figure` command."""
    build_trf_score_qc_figure(
        cfg=cfg,
        eeg_table_path=Path(args.eeg_table),
        feature_table_path=Path(args.feature_table),
        output_path=Path(args.out_fig),
    )
