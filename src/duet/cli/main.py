# ==================================================================================================
#                                   CLI
# ==================================================================================================
#
# Entry point for the `duet` command-line interface.
#
# This module is a thin dispatcher:
# - parse global + subcommand arguments
# - load project config once
# - call a single library function per subcommand
#
# Scientific logic must live in `duet.*` (preprocessing/analysis/stats), not here.
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

from duet.config import load_project_config
from duet.cli.types import CliCommand

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


# ==================================================================================================
# Command registry
# ==================================================================================================

_COMMANDS: dict[str, str | CliCommand] = {
    "downsample": "duet.cli.commands.preprocessing.downsampling",
    "reref": "duet.cli.commands.preprocessing.reref",
    "ica-apply": "duet.cli.commands.preprocessing.ica",
    "interpolate": "duet.cli.commands.preprocessing.interpolation",
    "filter": "duet.cli.commands.preprocessing.filtering",
    "metadata": "duet.cli.commands.preprocessing.metadata",
    "epoch": "duet.cli.commands.preprocessing.epoching",
    "acoustic-envelope": "duet.cli.commands.features.envelope",
    "acoustic-pitch": "duet.cli.commands.features.pitch",
    "acoustic-formants": "duet.cli.commands.features.formants",
    "alignment-events": "duet.cli.commands.features.alignment_events",
    "token-events": "duet.cli.commands.features.tokens",
    "pos-tags": "duet.cli.commands.features.pos",
    "word-class-events": "duet.cli.commands.features.word_class",
    "pos-qc": "duet.cli.commands.pos_qc",
    "trf": "duet.cli.commands.trf",
    "trf-kernel-qc": "duet.cli.commands.trf_qc",
    "trf-alpha-qc": "duet.cli.commands.trf_alpha_qc",
    "trf-score-qc": "duet.cli.commands.trf_score_qc",
    "trf-score-qc-figure": "duet.cli.commands.trf_score_qc_figure",
    "trf-main-figure": "duet.cli.commands.trf_main_figure",
    "speech-artefact-qc": "duet.cli.commands.speech_artefact_qc",
    "ipu-turn-taking-figure": "duet.cli.commands.ipu_turn_taking_figure",
}

_COMMAND_CONFIG_SECTIONS: dict[str, tuple[str, ...]] = {
    "downsample": ("preprocessing",),
    "reref": ("preprocessing",),
    "ica-apply": ("preprocessing",),
    "interpolate": ("preprocessing",),
    "filter": ("preprocessing",),
    "metadata": (),
    "epoch": (),
    "acoustic-envelope": ("features",),
    "acoustic-pitch": ("features",),
    "acoustic-formants": ("features",),
    "alignment-events": ("features",),
    "token-events": ("features",),
    "pos-tags": ("features",),
    "word-class-events": ("features",),
    "pos-qc": ("features",),
    "trf": ("paths", "trf"),
    "trf-kernel-qc": ("paths", "trf"),
    "trf-alpha-qc": ("paths", "trf"),
    "trf-score-qc": ("paths", "trf"),
    "trf-score-qc-figure": ("paths", "trf", "viz"),
    "trf-main-figure": ("paths", "trf", "viz"),
    "speech-artefact-qc": ("paths", "preprocessing", "viz"),
    "ipu-turn-taking-figure": ("paths", "viz"),
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
        duet downsample --config config/config.yaml --in-edf sub-01.edf --channels sub-01_channels.tsv \
          --sfreq 2048 --target-sfreq 512 --out derived/raw_ds.fif

        duet erp-generate --config config/config.yaml --epoch-dir derived/epochs --subject-list config/subjects.txt \
          --out-dir derived/erp --split-col self_rate --min-latency 0 --max-latency 2 --min-response-duration 0.2 \
          --min-self-n-syllables 3 --min-other-n-syllables 3 --overwrite
    """
    parser = argparse.ArgumentParser(
        prog="duet",
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

    command_name = str(args.command)
    cfg = load_project_config(
        Path(args.config),
        sections=_COMMAND_CONFIG_SECTIONS.get(command_name, ()),
    )
    module_or_path = _COMMANDS.get(command_name)
    if module_or_path is None:
        raise RuntimeError(f"Unknown command: {command_name}")
    module = _resolve_command_module(command_name, module_or_path)

    if not hasattr(module, "run"):
        raise RuntimeError(f"CLI command module for '{command_name}' is missing run().")

    module.run(args, cfg)


if __name__ == "__main__":
    main()
