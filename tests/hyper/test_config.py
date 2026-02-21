"""Tests for configuration loading behavior.

These tests guard the single config ingestion boundary used by both CLI and
library layers.
"""

from pathlib import Path

import pytest

from hyper.config import ProjectConfig, load_project_config


def test_load_project_config_returns_project_config(tmp_path: Path) -> None:
    """A valid YAML mapping should be returned as `ProjectConfig.raw`."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("project:\n  name: demo\n", encoding="utf-8")

    cfg = load_project_config(cfg_path)

    assert isinstance(cfg, ProjectConfig)
    assert cfg.raw["project"]["name"] == "demo"


def test_load_project_config_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    """Top-level YAML must be a mapping, not a list/scalar."""
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Config must be a mapping"):
        load_project_config(cfg_path)
