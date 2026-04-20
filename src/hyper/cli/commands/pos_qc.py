"""CLI command for lightweight POS annotation QC figures and tables."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

from hyper.viz.pos_qc import RUN_GROUPING_CHOICES, load_pos_qc_dataset, write_pos_qc_outputs


def add_subparser(subparsers: Any) -> None:
    """Register the `pos-qc` subcommand."""
    parser = subparsers.add_parser(
        "pos-qc",
        help="Generate lightweight QC plots and tables from exported POS TSV files.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        default=None,
        help="POS feature TSV input. Can be passed multiple times.",
    )
    parser.add_argument(
        "--glob",
        dest="globs",
        action="append",
        default=None,
        help="Glob pattern expanded to POS TSV inputs. Can be passed multiple times.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for QC tables and figures.")
    parser.add_argument("--title-prefix", type=str, default=None, help="Optional title prefix applied to figures.")
    parser.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Optional subject identifier override when not present in-file or filename.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier override when not present in-file or filename.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=RUN_GROUPING_CHOICES,
        default="auto",
        help="Grouping used to define per-run QC units.",
    )


def _resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve explicit input paths plus any requested glob patterns."""
    input_paths: list[Path] = []
    for input_path in args.inputs or []:
        input_paths.append(Path(input_path))

    for pattern in args.globs or []:
        matched_paths = sorted(glob.glob(str(pattern)))
        input_paths.extend(Path(path) for path in matched_paths)

    deduplicated_paths = sorted({path.resolve() for path in input_paths})
    if len(deduplicated_paths) == 0:
        raise ValueError("Provide at least one --input or --glob for `hyper pos-qc`.")
    return deduplicated_paths


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `pos-qc` command."""
    del cfg
    input_paths = _resolve_input_paths(args)
    dataset = load_pos_qc_dataset(
        input_paths,
        subject_id=args.subject_id,
        run_id=args.run_id,
        grouping=str(args.group_by),
    )
    write_pos_qc_outputs(
        dataset,
        Path(args.out_dir),
        title_prefix=args.title_prefix,
    )
