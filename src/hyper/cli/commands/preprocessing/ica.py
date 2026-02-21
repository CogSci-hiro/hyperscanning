# ==================================================================================================
#                                CLI: ica-apply
# ==================================================================================================
#
# Command handler for: `conv ica-apply ...`
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

from hyper.preprocessing.ica import apply_ica_fif_to_fif


# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_PRELOAD: bool = True


# ==================================================================================================
# Subparser
# ==================================================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register the `ica-apply` subcommand.

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
        "ica-apply",
        help="Apply a precomputed ICA solution (manual curation done elsewhere).",
    )

    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--in-fif", type=Path, required=True, help="Input raw FIF file (pre-ICA).")
    parser.add_argument("--ica", type=Path, required=True, help="Precomputed ICA object (ica.fif).")
    parser.add_argument("--out", type=Path, required=True, help="Output raw FIF file (post-ICA).")
    parser.add_argument(
        "--preload",
        action="store_true",
        default=DEFAULT_PRELOAD,
        help="Preload FIF into memory before applying ICA.",
    )


# ==================================================================================================
# Runner
# ==================================================================================================

def run(args: argparse.Namespace, cfg) -> None:
    """
    Execute the `ica-apply` command.

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

    apply_ica_fif_to_fif(
        input_fif_path=args.in_fif,
        ica_path=args.ica,
        output_fif_path=args.out,
        config=cfg,
        preload=bool(args.preload),
    )
