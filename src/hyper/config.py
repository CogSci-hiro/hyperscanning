# ==================================================================================================
#                               Config loading
# ==================================================================================================
#
# This module defines the *single, canonical entry point* for reading project
# configuration from disk.
#
# Why this exists
# ---------------
# The pipeline relies on a YAML config file to define:
#   - dataset scope (subjects, runs, tasks)
#   - scientific parameters (filters, epochs, thresholds, quantiles, etc.)
#   - filesystem layout (raw / derived / results / reports roots)
#
# Instead of letting every script and Snakemake rule load YAML independently,
# we centralize config parsing here to:
#   - guarantee consistent behavior across CLI tools, library code, and
#     Snakemake wrappers
#   - validate basic assumptions about the config structure once, at the
#     boundary between "user input" and "pipeline logic"
#   - provide a stable, typed container (`ProjectConfig`) that can be passed
#     through the codebase without re-reading files from disk
#
# Design principles
# -----------------
# - This module performs *no scientific logic*.
# - It does not interpret or transform configuration values.
# - It only loads, validates, and packages raw config data.
#
# Any domain-specific meaning of config fields (e.g., what "epochs.tmin_s"
# actually does) belongs in the analysis or workflow layers, not here.
#
# This separation makes the pipeline easier to test, refactor, and reproduce.
#

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


# ==================================================================================================
#                                   TYPES
# ==================================================================================================

@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """
    Parsed project configuration.

    Parameters
    ----------
    raw
        Raw config dictionary loaded from YAML.

    Usage example
    -------------
        cfg = load_project_config(Path("config/config.yaml"))
        raw_root = cfg.raw["paths"]["raw_root"]
    """

    raw: Dict[str, Any]


# ==================================================================================================
#                               IO / PLOTTING
# ==================================================================================================

def load_project_config(config_path: Path) -> ProjectConfig:
    """
    Load YAML config into a ProjectConfig object.

    Parameters
    ----------
    config_path
        Path to YAML config file.

    Returns
    -------
    ProjectConfig
        Loaded configuration.

    Usage example
    -------------
        cfg = load_project_config(Path("config/config.yaml"))
        print(cfg.raw["project"]["name"])
    """
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping at top-level, got: {type(data)}")

    return ProjectConfig(raw=dict(data))
