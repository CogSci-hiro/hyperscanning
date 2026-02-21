# ==================================================================================================
#                              CLI: metadata
# ==================================================================================================
#
# Command handler for: `conv metadata ...`
#
# Stable interface for Snakemake:
#   conv metadata --config ... --ipu ... --raw ... --time-lock ... --anchor ... --margin ...
#       --out-tsv ... --out-events ...
#
# No scientific logic belongs here.
#

# ==================================================================================================
# Imports
# ==================================================================================================

import argparse
from pathlib import Path
from typing import Any

from hyper.preprocessing.metadata import make_metadata_and_events_from_self_ipu

# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_TIME_LOCK: str = "onset"
DEFAULT_ANCHOR: str = "self"
DEFAULT_MARGIN_S: float = 1.0


# ==================================================================================================
# Subparser
# ==================================================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register the `metadata` subcommand.

    Parameters
    ----------
    subparsers
        Subparser registry from the top-level CLI.

    Usage example
    -------------
        # called internally by conv.cli.main.build_arg_parser()
        add_subparser(subparsers)
    """
    parser = subparsers.add_parser(
        "metadata",
        help="Create metadata TSV + events NPY from self IPU CSV (partner inferred)",
    )

    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--ipu", type=Path, required=True, help="Self IPU CSV (partner inferred).")
    parser.add_argument("--raw", type=Path, required=True, help="Raw file readable by MNE (e.g., raw_filt.fif).")

    parser.add_argument(
        "--time-lock",
        type=str,
        choices=["onset", "offset"],
        default=DEFAULT_TIME_LOCK,
        help="Whether to time-lock to onset (start) or offset (end).",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        choices=["self", "other"],
        default=DEFAULT_ANCHOR,
        help="Which speaker defines the event rows.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN_S,
        help="Temporal margin in seconds for event window around the anchor.",
    )

    parser.add_argument("--out-tsv", type=Path, required=True, help="Output TSV file.")
    parser.add_argument("--out-events", type=Path, required=True, help="Output events .npy file.")


# ==================================================================================================
# Runner
# ==================================================================================================

def run(args: argparse.Namespace, cfg) -> None:
    """
    Execute the `metadata` command.

    Parameters
    ----------
    args
        Parsed argparse namespace for this subcommand.
    cfg
        Project config (already loaded once in conv.cli.main).

    Usage example
    -------------
        # called internally by conv.cli.main.main()
        run(args, cfg)
    """
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_events.parent.mkdir(parents=True, exist_ok=True)

    make_metadata_and_events_from_self_ipu(
        self_ipu_csv_path=args.ipu,
        raw_path=args.raw,
        output_tsv_path=args.out_tsv,
        output_events_npy_path=args.out_events,
        config=cfg,
        margin_s=float(args.margin),
        time_lock=str(args.time_lock),
        anchor=str(args.anchor),
    )
