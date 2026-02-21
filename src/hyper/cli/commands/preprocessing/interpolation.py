# ==================================================================================================
#                              CLI: interpolate
# ==================================================================================================
#
# Command handler for: `conv interpolate ...`
#
# Responsibilities
# ----------------
# - define subcommand arguments (add_subparser)
# - run the command given parsed args + loaded config (run)
#
# No scientific logic belongs here.
#

# ==================================================================================================
# Imports
# ==================================================================================================

import argparse
from pathlib import Path
from typing import Any

from hyper.preprocessing.interpolation import interpolate_bads_fif_to_fif


# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_PRELOAD: bool = True
DEFAULT_METHOD: str = "spline"


# ==================================================================================================
# Subparser
# ==================================================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register the `interpolate` subcommand.

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
        "interpolate",
        help="Interpolate bad channels using channels.tsv",
    )

    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--in-fif", type=Path, required=True, help="Input raw FIF file.")
    parser.add_argument("--channels", type=Path, required=True, help="Input channels.tsv file.")
    parser.add_argument(
        "--method",
        type=str,
        default=DEFAULT_METHOD,
        help=f"Interpolation method passed to MNE (default: {DEFAULT_METHOD}).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output raw FIF file.")
    parser.add_argument(
        "--preload",
        action="store_true",
        default=DEFAULT_PRELOAD,
        help="Preload FIF into memory before interpolation.",
    )


# ==================================================================================================
# Runner
# ==================================================================================================

def run(args: argparse.Namespace, cfg) -> None:
    """
    Execute the `interpolate` command.

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
    args.out.parent.mkdir(parents=True, exist_ok=True)

    interpolate_bads_fif_to_fif(
        input_fif_path=args.in_fif,
        channels_tsv_path=args.channels,
        output_fif_path=args.out,
        config=cfg,
        method=str(args.method),
        preload=bool(args.preload),
    )
