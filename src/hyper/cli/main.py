# ==================================================================================================
#                                   CLI
# ==================================================================================================
#
# Entry point for the `hyper` command-line interface.
#
# This module is a thin dispatcher:
# - parse global + subcommand arguments
# - load project config once
# - call a single library function per subcommand
#
# Scientific logic must live in `hyper.*` (preprocessing/analysis/stats), not here.
#
# ==================================================================================================
# Imports
# ==================================================================================================

import argparse
from pathlib import Path
from typing import Dict, Sequence

from hyper.config import load_project_config
from hyper.cli.types import CliCommand

# Command handlers (thin; no scientific logic here either)
from hyper.cli.commands.preprocessing import (  # noqa: F401
    downsampling as cmd_downsample,
    epoching as cmd_epoch,
    filtering as cmd_filter,
    ica as cmd_ica_apply,
    interpolation as cmd_interpolate,
    metadata as cmd_metadata,
    reref as cmd_reref,
)


# ==================================================================================================
# Command registry
# ==================================================================================================

_COMMANDS: Dict[str, CliCommand] = {
    "downsample": cmd_downsample,
    "reref": cmd_reref,
    "ica-apply": cmd_ica_apply,
    "interpolate": cmd_interpolate,
    "filter": cmd_filter,
    "metadata": cmd_metadata,
    "epoch": cmd_epoch
}


# ==================================================================================================
# Argument parsing
# ==================================================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser with subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.

    Usage example
    -------------
        hyper downsample --config config/config.yaml --in-edf sub-01.edf --channels sub-01_channels.tsv \
          --sfreq 2048 --target-sfreq 512 --out derived/raw_ds.fif

        hyper erp-generate --config config/config.yaml --epoch-dir derived/epochs --subject-list config/subjects.txt \
          --out-dir derived/erp --split-col self_rate --min-latency 0 --max-latency 2 --min-response-duration 0.2 \
          --min-self-n-syllables 3 --min-other-n-syllables 3 --overwrite
    """
    parser = argparse.ArgumentParser(
        prog="hyper",
        description="Speech-rate convergence EEG analysis pipeline",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    for name, module in _COMMANDS.items():
        if not hasattr(module, "add_subparser"):
            raise RuntimeError(f"CLI command module for '{name}' is missing add_subparser().")
        module.add_subparser(subparsers)

    return parser


# ==================================================================================================
# Entry point
# ==================================================================================================

def main(argv: Sequence[str] | None = None) -> None:
    """
    CLI entry point.

    Parameters
    ----------
    argv
        Optional argv for testing. If None, reads from sys.argv.

    Returns
    -------
    None

    Usage example
    -------------
        main([
            "downsample",
            "--config", "config/config.yaml",
            "--in-edf", "sub-01_eeg.edf",
            "--channels", "sub-01_channels.tsv",
            "--sfreq", "2048",
            "--target-sfreq", "512",
            "--out", "derived/raw_ds.fif",
        ])
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)  # noqa

    # Every subcommand requires --config (enforced by handlers)
    if not hasattr(args, "config"):
        raise RuntimeError("Internal error: subcommand args missing --config.")

    cfg = load_project_config(Path(args.config))

    command_name = str(args.command)
    module = _COMMANDS.get(command_name)
    if module is None:
        raise RuntimeError(f"Unknown command: {command_name}")

    if not hasattr(module, "run"):
        raise RuntimeError(f"CLI command module for '{command_name}' is missing run().")

    module.run(args, cfg)


if __name__ == "__main__":
    main()
