# ==================================================================================================
#                               Project paths
# ==================================================================================================
#
# This module converts loosely typed config entries into explicit Path objects
# used throughout the codebase. The goal is to resolve path roots once and then
# pass around a strongly named container (`ProjectPaths`) instead of raw dicts.

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .config import ProjectConfig

# ==================================================================================================
#                                   TYPES
# ==================================================================================================

@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """
    Resolved project paths.

    Parameters
    ----------
    raw_root
        Root directory for raw input data.
    out_dir
        Root directory for writable pipeline outputs and intermediate data.
    results_root
        Root directory for results (stats tables, cluster results, etc.).
    reports_root
        Root directory for report artifacts (figures, tables).

    Usage example
    -------------
        cfg = load_project_config(Path("config/config.yaml"))
        paths = ProjectPaths.from_config(cfg)
        print(paths.results_root)
    """

    raw_root: Path
    out_dir: Path
    results_root: Path
    reports_root: Path

    @staticmethod
    def from_config(cfg: ProjectConfig) -> "ProjectPaths":
        """
        Construct ProjectPaths from config.

        Parameters
        ----------
        cfg
            Project configuration.

        Returns
        -------
        ProjectPaths
            Resolved paths.

        Usage example
        -------------
            paths = ProjectPaths.from_config(cfg)
        """
        # Configuration is expected to define top-level path roots under
        # "paths". We validate this shape up-front to fail early with a clear
        # error, rather than surfacing KeyError later in unrelated code.
        paths_cfg = cfg.raw.get("paths", None)
        if not isinstance(paths_cfg, Mapping):
            raise ValueError("Config missing required mapping: paths")

        raw_root_value = paths_cfg.get("raw_root", paths_cfg.get("bids_root"))
        if raw_root_value is None:
            raise ValueError("Config paths must define either 'raw_root' or 'bids_root'")

        # Cast values via str() first so YAML scalar types are handled
        # consistently even if users quote/unquote fields differently.
        out_dir_value = paths_cfg.get("out_dir", paths_cfg.get("derived_root"))
        if out_dir_value is None:
            raise ValueError("Config paths must define 'out_dir'")

        results_root_value = paths_cfg.get("results_root", out_dir_value)

        return ProjectPaths(
            raw_root=Path(str(raw_root_value)),
            out_dir=Path(str(out_dir_value)),
            results_root=Path(str(results_root_value)),
            reports_root=Path(str(paths_cfg["reports_root"])),
        )
