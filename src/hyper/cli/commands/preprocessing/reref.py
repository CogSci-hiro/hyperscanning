# ==================================================================================================
#                                CLI: reref
# ==================================================================================================
#
# Command handler for: `conv reref ...`
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

from hyper.preprocessing.reref import rereference_fif_to_fif


# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_PRELOAD: bool = False


# ==================================================================================================
# Subparser
# ==================================================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register the `reref` subcommand.

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
        "reref",
        help="Rereference EEG (average reference) and set montage",
    )

    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--in-fif", type=Path, required=True, help="Input raw FIF file (e.g., raw_ds.fif).")
    parser.add_argument("--channels", type=Path, required=True, help="Input channels.tsv file.")
    parser.add_argument("--out", type=Path, required=True, help="Output FIF file (e.g., raw_reref.fif).")
    parser.add_argument(
        "--preload",
        action="store_true",
        default=DEFAULT_PRELOAD,
        help="Preload FIF into memory before rereferencing (can be memory-heavy).",
    )


# ==================================================================================================
# Runner
# ==================================================================================================

def run(args: argparse.Namespace, cfg) -> None:
    """
    Execute the `reref` command.

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

    rereference_fif_to_fif(
        input_fif_path=args.in_fif,
        channels_tsv_path=args.channels,
        output_fif_path=args.out,
        config=cfg,
        preload=bool(args.preload),
    )
