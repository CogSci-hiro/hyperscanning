"""CLI command for the speech artefact summary figure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.viz.speech_artefact_qc import build_speech_artefact_summary_figure


def add_subparser(subparsers: Any) -> None:
    """Register the `speech-artefact-qc` subcommand."""
    parser = subparsers.add_parser(
        "speech-artefact-qc",
        help="Render a speech artefact summary figure across filtered and ICA inputs.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument(
        "--filtered-noica",
        dest="filtered_noica_inputs",
        action="append",
        type=Path,
        default=None,
        help="Filtered no-ICA FIF input. Can be passed multiple times.",
    )
    parser.add_argument(
        "--filtered",
        dest="filtered_inputs",
        action="append",
        type=Path,
        default=None,
        help="Filtered post-ICA FIF input. Can be passed multiple times.",
    )
    parser.add_argument(
        "--ica",
        dest="ica_inputs",
        action="append",
        type=Path,
        default=None,
        help="Precomputed ICA FIF input. Can be passed multiple times.",
    )
    parser.add_argument("--out-fig", type=Path, required=True, help="Output figure path.")


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `speech-artefact-qc` command."""
    build_speech_artefact_summary_figure(
        cfg=cfg,
        filtered_noica_paths=[Path(path) for path in args.filtered_noica_inputs or []],
        filtered_paths=[Path(path) for path in args.filtered_inputs or []],
        ica_paths=[Path(path) for path in args.ica_inputs or []],
        output_path=Path(args.out_fig),
    )
