"""CLI command for Stanza-backed POS extraction on aligned token rows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.features.linguistic.pos import DEFAULT_FEATURE_NAME, StanzaPosConfig
from hyper.features.pipelines.linguistic import run_token_pos_pipeline


def add_subparser(subparsers: Any) -> None:
    """Register the `pos-tags` subcommand."""
    parser = subparsers.add_parser(
        "pos-tags",
        help="Attach Stanza POS annotations to the original aligned token rows for one subject/run.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--tokens", type=Path, required=True, help="Dyad-level aligned token CSV.")
    parser.add_argument("--subject", type=str, required=True, help="Subject identifier, e.g. sub-015.")
    parser.add_argument("--run", type=str, required=True, help="Conversation run number.")
    parser.add_argument("--out-tsv", type=Path, required=True, help="Output TSV feature table path.")
    parser.add_argument("--out-sidecar", type=Path, required=True, help="Output JSON sidecar path.")
    parser.add_argument(
        "--feature-name",
        type=str,
        default=DEFAULT_FEATURE_NAME,
        help="Feature name stored in metadata.",
    )
    parser.add_argument(
        "--source-subject",
        type=str,
        default=None,
        help="Optional source subject stored in metadata.",
    )
    parser.add_argument("--source-role", type=str, default=None, help="Optional source role stored in metadata.")
    parser.add_argument("--language", type=str, default=None, help="Optional Stanza language override.")
    parser.add_argument(
        "--processors",
        type=str,
        default=None,
        help="Optional comma-separated Stanza processors override.",
    )
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=None,
        help="Optional local Stanza resources directory.",
    )
    parser.add_argument(
        "--allow-download",
        dest="allow_download",
        action="store_true",
        default=None,
        help="Allow Stanza to download missing resources automatically.",
    )
    parser.add_argument(
        "--no-allow-download",
        dest="allow_download",
        action="store_false",
        help="Disable automatic Stanza resource downloads and require a populated resources directory.",
    )
    parser.add_argument(
        "--fail-on-mapping-error",
        action="store_true",
        help="Raise an error if any token row cannot be mapped exactly to a Stanza word.",
    )
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=None,
        help="Token label to exclude. Can be provided multiple times.",
    )
    parser.add_argument(
        "--progress",
        dest="show_progress",
        action="store_true",
        default=None,
        help="Show a staged progress bar during POS extraction.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable the POS extraction progress bar.",
    )


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `pos-tags` command."""
    features_cfg = getattr(cfg, "raw", {}).get("features", {})
    pos_cfg = features_cfg.get("stanza_pos", {})
    defaults = StanzaPosConfig()
    show_progress = pos_cfg.get("show_progress", True)
    if args.show_progress is not None:
        show_progress = bool(args.show_progress)

    allow_download = pos_cfg.get("allow_download", defaults.allow_download)
    if args.allow_download is not None:
        allow_download = bool(args.allow_download)

    fail_on_mapping_error = bool(
        args.fail_on_mapping_error or pos_cfg.get("fail_on_mapping_error", defaults.fail_on_mapping_error)
    )

    config = StanzaPosConfig(
        enabled=bool(pos_cfg.get("enabled", defaults.enabled)),
        language=str(
            args.language if args.language is not None else pos_cfg.get("language", defaults.language)
        ),
        processors=str(
            args.processors if args.processors is not None else pos_cfg.get("processors", defaults.processors)
        ),
        resources_dir=args.resources_dir
        if args.resources_dir is not None
        else (
            Path(str(pos_cfg["resources_dir"])).expanduser()
            if pos_cfg.get("resources_dir") not in (None, "")
            else defaults.resources_dir
        ),
        allow_download=bool(allow_download),
        preserve_unmapped_rows=bool(
            pos_cfg.get("preserve_unmapped_rows", defaults.preserve_unmapped_rows)
        ),
        fail_on_mapping_error=fail_on_mapping_error,
    )

    run_token_pos_pipeline(
        tokens_path=args.tokens,
        subject=str(args.subject),
        run=str(args.run),
        output_tsv_path=args.out_tsv,
        output_sidecar_path=args.out_sidecar,
        config=config,
        feature_name=str(args.feature_name),
        exclude_labels=tuple(args.exclude_label or ()),
        source_subject=args.source_subject,
        source_role=args.source_role,
        show_progress=bool(show_progress),
    )
