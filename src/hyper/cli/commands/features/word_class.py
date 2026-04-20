"""CLI command for word-class onset events derived from token-level POS tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.features.pipelines.linguistic import run_word_class_event_pipeline


def add_subparser(subparsers: Any) -> None:
    """Register the `word-class-events` subcommand."""
    parser = subparsers.add_parser(
        "word-class-events",
        help="Export binary onset-coded function/content word events from token-level POS tables.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--pos-features", type=Path, required=True, help="Token-level POS feature TSV.")
    parser.add_argument("--subject", type=str, required=True, help="Subject identifier, e.g. sub-015.")
    parser.add_argument("--run", type=str, required=True, help="Conversation run number.")
    parser.add_argument(
        "--word-class",
        type=str,
        choices=("function", "content"),
        required=True,
        help="Word class to export.",
    )
    parser.add_argument("--out-tsv", type=Path, required=True, help="Output TSV event table path.")
    parser.add_argument("--out-sidecar", type=Path, required=True, help="Output JSON sidecar path.")
    parser.add_argument("--feature-name", type=str, required=True, help="Feature name stored in metadata.")
    parser.add_argument(
        "--source-subject",
        type=str,
        default=None,
        help="Optional source subject stored in metadata.",
    )
    parser.add_argument("--source-role", type=str, default=None, help="Optional source role stored in metadata.")


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `word-class-events` command."""
    del cfg
    run_word_class_event_pipeline(
        pos_features_path=args.pos_features,
        subject=str(args.subject),
        run=str(args.run),
        word_class=str(args.word_class),
        output_tsv_path=args.out_tsv,
        output_sidecar_path=args.out_sidecar,
        feature_name=str(args.feature_name),
        source_subject=args.source_subject,
        source_role=args.source_role,
    )
