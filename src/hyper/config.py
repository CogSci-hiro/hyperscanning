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
from typing import Any, Dict, Mapping, Sequence

import yaml


SECTION_FILE_NAMES: dict[str, str] = {
    "paths": "paths.yaml",
    "preprocessing": "preprocessing.yaml",
    "features": "features.yaml",
    "trf": "trf.yaml",
}


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


def _normalized_sections(sections: Sequence[str]) -> tuple[str, ...]:
    """Validate and normalize the requested sibling config sections."""
    unknown = sorted({str(section) for section in sections if str(section) not in SECTION_FILE_NAMES})
    if unknown:
        raise ValueError(f"Unknown config sections requested: {', '.join(unknown)}")
    return tuple(dict.fromkeys(str(section) for section in sections))


def _load_optional_section(
    *,
    config_path: Path,
    base: Mapping[str, Any],
    section_name: str,
) -> Dict[str, Any]:
    """Load one optional sibling section and merge it with inline overrides."""
    base_section = base.get(section_name, {})
    if base_section is None:
        base_section = {}
    if not isinstance(base_section, Mapping):
        raise ValueError(f"Config section '{section_name}' must be a mapping when present.")

    sibling_path = config_path.with_name(SECTION_FILE_NAMES[section_name])
    sibling_section: Mapping[str, Any] = {}
    if sibling_path.exists():
        sibling_data = _load_yaml_mapping(sibling_path)
        raw_section = sibling_data.get(section_name, {})
        if raw_section is None:
            raw_section = {}
        if not isinstance(raw_section, Mapping):
            raise ValueError(f"Config section '{section_name}' in {sibling_path} must be a mapping.")
        sibling_section = raw_section

    if section_name == "trf":
        merged = {**dict(base_section), **dict(sibling_section)}
    else:
        merged = {**dict(sibling_section), **dict(base_section)}
    return merged


def load_raw_project_config(config_path: Path, *, sections: Sequence[str] = ()) -> Dict[str, Any]:
    """Load config YAML and optionally attach only the requested sibling sections."""
    config_path = Path(config_path)
    base = _load_yaml_mapping(config_path)
    merged = dict(base)
    for section_name in _normalized_sections(sections):
        merged[section_name] = _load_optional_section(
            config_path=config_path,
            base=base,
            section_name=section_name,
        )
    return merged

def load_project_config(config_path: Path, *, sections: Sequence[str] = ()) -> ProjectConfig:
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
    return ProjectConfig(raw=load_raw_project_config(Path(config_path), sections=sections))
