# ==================================================================================================
#                               CLI: filter
# ==================================================================================================
#
# Command handler for: `conv filter ...`
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

from hyper.preprocessing.filtering import bandpass_filter_fif_to_fif

# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_PRELOAD: bool = True


# ==================================================================================================
# Subparser
# ==================================================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register the `filter` subcommand.

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
        "filter",
        help="Apply band-pass filter to raw FIF data",
    )

    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--in-fif", type=Path, required=True, help="Input raw FIF file.")
    parser.add_argument("--l-freq", type=float, required=True, help="High-pass cutoff frequency in Hz.")
    parser.add_argument("--h-freq", type=float, required=True, help="Low-pass cutoff frequency in Hz.")
    parser.add_argument("--out", type=Path, required=True, help="Output raw FIF file.")
    parser.add_argument(
        "--preload",
        action="store_true",
        default=DEFAULT_PRELOAD,
        help="Preload FIF into memory before filtering.",
    )


# ==================================================================================================
# Runner
# ==================================================================================================

def run(args: argparse.Namespace, cfg) -> None:
    """
    Execute the `filter` command.

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

    bandpass_filter_fif_to_fif(
        input_fif_path=args.in_fif,
        output_fif_path=args.out,
        config=cfg,
        l_freq_hz=float(args.l_freq),
        h_freq_hz=float(args.h_freq),
        preload=bool(args.preload),
    )
