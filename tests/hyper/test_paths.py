"""Tests for conversion from raw config mappings into typed project paths."""

from pathlib import Path

import pytest

from hyper.config import ProjectConfig
from hyper.paths import ProjectPaths


def test_project_paths_from_config_builds_path_objects(sample_config_dict: dict) -> None:
    """All path fields should be converted to `pathlib.Path` instances."""
    cfg = ProjectConfig(raw=sample_config_dict)

    paths = ProjectPaths.from_config(cfg)

    assert paths.raw_root == Path("raw")
    assert paths.derived_root == Path("derived")
    assert paths.results_root == Path("results")
    assert paths.reports_root == Path("reports")


def test_project_paths_from_config_requires_paths_mapping() -> None:
    """Missing `paths` should fail early with a clear error."""
    cfg = ProjectConfig(raw={})

    with pytest.raises(ValueError, match="missing required mapping: paths"):
        ProjectPaths.from_config(cfg)
