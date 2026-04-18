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
import importlib
import os
import sys
from pathlib import Path
from collections.abc import Sequence

from hyper.config import load_project_config
from hyper.cli.types import CliCommand

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


# ==================================================================================================
# Command registry
# ==================================================================================================

_COMMANDS: dict[str, str | CliCommand] = {
    "downsample": "hyper.cli.commands.preprocessing.downsampling",
    "reref": "hyper.cli.commands.preprocessing.reref",
    "ica-apply": "hyper.cli.commands.preprocessing.ica",
    "interpolate": "hyper.cli.commands.preprocessing.interpolation",
    "filter": "hyper.cli.commands.preprocessing.filtering",
    "metadata": "hyper.cli.commands.preprocessing.metadata",
    "epoch": "hyper.cli.commands.preprocessing.epoching",
    "acoustic-envelope": "hyper.cli.commands.features.envelope",
    "acoustic-pitch": "hyper.cli.commands.features.pitch",
    "acoustic-formants": "hyper.cli.commands.features.formants",
    "alignment-events": "hyper.cli.commands.features.alignment_events",
    "token-events": "hyper.cli.commands.features.tokens",
    "pos-tags": "hyper.cli.commands.features.pos",
    "trf": "hyper.cli.commands.trf",
    "trf-kernel-qc": "hyper.cli.commands.trf_qc",
    "trf-alpha-qc": "hyper.cli.commands.trf_alpha_qc",
}


def _resolve_command_module(command: str, module_or_path: str | CliCommand) -> CliCommand:
    """Resolve a command registry entry into a command module."""
    if isinstance(module_or_path, str):
        return importlib.import_module(module_or_path)
    return module_or_path


# ==================================================================================================
# Argument parsing
# ==================================================================================================

def build_arg_parser(selected_command: str | None = None) -> argparse.ArgumentParser:
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

    if selected_command is None:
        for name in _COMMANDS:
            subparsers.add_parser(name)
        return parser

    module_or_path = _COMMANDS.get(selected_command)
    if module_or_path is None:
        raise RuntimeError(f"Unknown command: {selected_command}")

    module = _resolve_command_module(selected_command, module_or_path)
    if not hasattr(module, "add_subparser"):
        raise RuntimeError(f"CLI command module for '{selected_command}' is missing add_subparser().")
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
    argv_list = list(sys.argv[1:] if argv is None else argv)

    if not argv_list or argv_list[0] in {"-h", "--help"}:
        parser = build_arg_parser()
        parser.print_help()
        return

    command_name = str(argv_list[0])
    parser = build_arg_parser(command_name)
    args = parser.parse_args(argv_list)  # noqa

    # Every subcommand requires --config (enforced by handlers)
    if not hasattr(args, "config"):
        raise RuntimeError("Internal error: subcommand args missing --config.")

    cfg = load_project_config(Path(args.config))

    command_name = str(args.command)
    module_or_path = _COMMANDS.get(command_name)
    if module_or_path is None:
        raise RuntimeError(f"Unknown command: {command_name}")
    module = _resolve_command_module(command_name, module_or_path)

    if not hasattr(module, "run"):
        raise RuntimeError(f"CLI command module for '{command_name}' is missing run().")

    module.run(args, cfg)


if __name__ == "__main__":
    main()
