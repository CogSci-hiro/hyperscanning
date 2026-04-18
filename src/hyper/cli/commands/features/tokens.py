"""CLI command for exporting subject-specific token onset event tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.features.pipelines.acoustics import run_token_event_pipeline


def add_subparser(subparsers: Any) -> None:
    """Register the `token-events` subcommand."""
    parser = subparsers.add_parser(
        "token-events",
        help="Export subject-specific token onsets from a dyad token table.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--tokens", type=Path, required=True, help="Dyad-level token CSV.")
    parser.add_argument("--subject", type=str, required=True, help="Subject identifier, e.g. sub-015.")
    parser.add_argument("--run", type=str, required=True, help="Conversation run number.")
    parser.add_argument("--out-tsv", type=Path, required=True, help="Output TSV event table path.")
    parser.add_argument("--out-sidecar", type=Path, required=True, help="Output JSON sidecar path.")
    parser.add_argument("--feature-name", type=str, default="tokens", help="Feature name stored in metadata.")
    parser.add_argument("--source-subject", type=str, default=None, help="Optional source subject stored in metadata.")
    parser.add_argument("--source-role", type=str, default=None, help="Optional source role stored in metadata.")
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=None,
        help="Token label to exclude. Can be provided multiple times.",
    )


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `token-events` command."""
    del cfg
    run_token_event_pipeline(
        tokens_path=args.tokens,
        subject=str(args.subject),
        run=str(args.run),
        output_tsv_path=args.out_tsv,
        output_sidecar_path=args.out_sidecar,
        feature_name=str(args.feature_name),
        exclude_labels=tuple(args.exclude_label or ()),
        source_subject=args.source_subject,
        source_role=args.source_role,
    )
