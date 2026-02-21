"""
Shared CLI typing contracts.

This module defines protocol interfaces used by the CLI dispatcher to type
check command modules without importing concrete implementations at type time.
"""

import argparse
from typing import Any, Protocol

# ==================================================================================================
#                                   TYPES
# ==================================================================================================

class CliCommand(Protocol):
    """
    Structural interface for CLI subcommand modules.

    Any module registered in `hyper.cli.main._COMMANDS` should implement this
    protocol:
    - `add_subparser(...)` registers CLI arguments.
    - `run(...)` executes command logic after argument parsing + config load.
    """

    def add_subparser(self, subparsers: Any) -> None: ...
    def run(self, args: argparse.Namespace, cfg) -> None: ...
