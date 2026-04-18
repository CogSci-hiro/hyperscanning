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

def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """Load one YAML file and require a top-level mapping."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping at top-level, got: {type(data)}")
    return dict(data)


def _merge_path_sections(external: Mapping[str, Any], base: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge `paths` mappings with base config taking precedence."""
    merged: Dict[str, Any] = dict(external)
    external_paths = external.get("paths", {})
    base_paths = base.get("paths", {})
    if isinstance(external_paths, Mapping) or isinstance(base_paths, Mapping):
        merged["paths"] = {
            **(dict(external_paths) if isinstance(external_paths, Mapping) else {}),
            **(dict(base_paths) if isinstance(base_paths, Mapping) else {}),
        }
    for key, value in base.items():
        if key == "paths" and "paths" in merged:
            continue
        merged[key] = value
    return merged


def _merge_named_sections(external: Mapping[str, Any], base: Mapping[str, Any], section_names: tuple[str, ...]) -> Dict[str, Any]:
    """Merge selected top-level mapping sections, with base values taking precedence."""
    merged: Dict[str, Any] = dict(external)
    for section_name in section_names:
        external_section = external.get(section_name, {})
        base_section = base.get(section_name, {})
        if isinstance(external_section, Mapping) or isinstance(base_section, Mapping):
            merged[section_name] = {
                **(dict(external_section) if isinstance(external_section, Mapping) else {}),
                **(dict(base_section) if isinstance(base_section, Mapping) else {}),
            }
    for key, value in base.items():
        if key in section_names and key in merged:
            continue
        merged[key] = value
    return merged


def load_raw_project_config(config_path: Path) -> Dict[str, Any]:
    """Load config YAML and merge optional sibling config fragments."""
    config_path = Path(config_path)
    base = _load_yaml_mapping(config_path)
    merged = dict(base)

    preprocessing_path = config_path.with_name("preprocessing.yaml")
    if preprocessing_path.exists():
        external = _load_yaml_mapping(preprocessing_path)
        merged = _merge_named_sections(external, merged, ("preprocessing",))

    features_path = config_path.with_name("features.yaml")
    if features_path.exists():
        external = _load_yaml_mapping(features_path)
        merged = _merge_named_sections(external, merged, ("features",))

    paths_path = config_path.with_name("paths.yaml")
    if paths_path.exists():
        external = _load_yaml_mapping(paths_path)
        merged = _merge_path_sections(external, merged)
    return merged

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
    return ProjectConfig(raw=load_raw_project_config(Path(config_path)))
