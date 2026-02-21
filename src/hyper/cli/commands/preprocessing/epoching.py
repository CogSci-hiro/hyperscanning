# ==================================================================================================
#                               CLI: epoch
# ==================================================================================================
#
# Command handler for: `conv epoch ...`
#
# Stable interface for Snakemake:
#   conv epoch --config ... --raw ... --events ... --metadata ... --tmin ... --tmax ...
#       --baseline --baseline-start ... --baseline-end ... --detrend ... --preload-raw --out ...
#
# No scientific logic belongs here.
#
# ==================================================================================================
# Imports
# ==================================================================================================

import argparse
from pathlib import Path
from typing import Any, Optional, Tuple

from hyper.preprocessing.epoching import make_epochs_fif_to_fif


# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_BASELINE: bool = True
DEFAULT_BASELINE_START_S: float = -0.2
DEFAULT_BASELINE_END_S: float = 0.0
DEFAULT_DETREND: int = -1  # < 1 means "None"
DEFAULT_PRELOAD_RAW: bool = False


# ==================================================================================================
# Subparser
# ==================================================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register the `epoch` subcommand.

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
        "epoch",
        help="Create MNE Epochs from raw FIF + events + metadata",
    )

    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--raw", type=Path, required=True, help="Input raw FIF file.")
    parser.add_argument("--events", type=Path, required=True, help="Input events .npy file (n_events, 3).")
    parser.add_argument("--metadata", type=Path, required=True, help="Input metadata TSV file.")
    parser.add_argument("--out", type=Path, required=True, help="Output epochs FIF file (e.g., epochs-epo.fif).")

    parser.add_argument("--tmin", type=float, required=True, help="Epoch start (seconds).")
    parser.add_argument("--tmax", type=float, required=True, help="Epoch end (seconds).")

    parser.add_argument(
        "--baseline",
        action="store_true",
        default=DEFAULT_BASELINE,
        help="Enable baseline correction (default: enabled).",
    )
    parser.add_argument("--baseline-start", type=float, default=DEFAULT_BASELINE_START_S, help="Baseline start (s).")
    parser.add_argument("--baseline-end", type=float, default=DEFAULT_BASELINE_END_S, help="Baseline end (s).")

    parser.add_argument(
        "--detrend",
        type=int,
        default=DEFAULT_DETREND,
        help="Detrend order (0 or 1). Use < 1 to disable (default).",
    )
    parser.add_argument(
        "--preload-raw",
        action="store_true",
        default=DEFAULT_PRELOAD_RAW,
        help="Preload raw into memory before epoching (usually not needed).",
    )


# ==================================================================================================
# Runner
# ==================================================================================================

def run(args: argparse.Namespace, cfg) -> None:
    """
    Execute the `epoch` command.

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

    baseline: Optional[Tuple[float, float]]
    if bool(args.baseline):
        baseline = (float(args.baseline_start), float(args.baseline_end))
    else:
        baseline = None

    detrend: Optional[int] = None if int(args.detrend) < 1 else int(args.detrend)

    make_epochs_fif_to_fif(
        raw_fif_path=args.raw,
        events_npy_path=args.events,
        metadata_tsv_path=args.metadata,
        output_epochs_path=args.out,
        config=cfg,
        tmin_s=float(args.tmin),
        tmax_s=float(args.tmax),
        baseline=baseline,
        detrend=detrend,
        preload_raw=bool(args.preload_raw),
        reject_by_annotation=False,
    )
