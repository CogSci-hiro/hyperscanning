# ==================================================================================================
#                              CLI: downsample
# ==================================================================================================
#
# Command handler for: `conv downsample ...`
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

from hyper.preprocessing.downsampling import downsample_edf_to_fif


# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_PRELOAD: bool = False


# ==================================================================================================
# Subparser
# ==================================================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register the `downsample` subcommand.

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
        "downsample",
        help="Convert EDF to FIF and optionally downsample",
    )

    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--in-edf", type=Path, required=True, help="Input EDF file.")
    parser.add_argument("--channels", type=Path, required=True, help="Input channels.tsv file.")

    # Kept for compatibility with existing Snakemake calls; not used by core logic.
    parser.add_argument(
        "--sfreq",
        type=float,
        required=False,
        default=0.0,
        help="Original sampling frequency (Hz). For compatibility/logging only.",
    )

    parser.add_argument("--target-sfreq", type=float, required=True, help="Target sampling frequency (Hz).")
    parser.add_argument("--out", type=Path, required=True, help="Output FIF file.")
    parser.add_argument(
        "--preload",
        action="store_true",
        default=DEFAULT_PRELOAD,
        help="Preload EDF into memory before resampling (can be memory-heavy).",
    )


# ==================================================================================================
# Runner
# ==================================================================================================

def run(args: argparse.Namespace, cfg) -> None:
    """
    Execute the `downsample` command.

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

    downsample_edf_to_fif(
        input_edf_path=args.in_edf,
        channels_tsv_path=args.channels,
        output_fif_path=args.out,
        config=cfg,
        target_sfreq_hz=float(args.target_sfreq),
        preload=bool(args.preload),
    )
