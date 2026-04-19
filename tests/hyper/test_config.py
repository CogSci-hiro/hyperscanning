"""Tests for configuration loading behavior."""

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


def test_load_project_config_does_not_merge_sibling_sections_by_default(tmp_path: Path) -> None:
    """Only the explicitly requested sibling fragments should be attached."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("project:\n  name: demo\n", encoding="utf-8")
    (tmp_path / "paths.yaml").write_text("paths:\n  bids_root: /bids\n", encoding="utf-8")

    cfg = load_project_config(cfg_path)

    assert "paths" not in cfg.raw


def test_load_project_config_loads_requested_paths_section(tmp_path: Path) -> None:
    """Requested path fragments should be loaded and merged with inline overrides."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "paths:\n  derived_root: /override\n  precomputed_ica_root: /override-ica\n",
        encoding="utf-8",
    )
    (tmp_path / "paths.yaml").write_text(
        "paths:\n  bids_root: /bids\n  derived_root: /derived\n  lm_feature_root: /lm\n  precomputed_ica_root: /ica\n",
        encoding="utf-8",
    )

    cfg = load_project_config(cfg_path, sections=("paths",))

    assert cfg.raw["paths"]["bids_root"] == "/bids"
    assert cfg.raw["paths"]["derived_root"] == "/override"
    assert cfg.raw["paths"]["precomputed_ica_root"] == "/override-ica"


def test_load_project_config_loads_requested_preprocessing_section(tmp_path: Path) -> None:
    """Requested preprocessing fragments should be loaded and allow inline overrides."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("preprocessing:\n  output:\n    save_intermediates: true\n", encoding="utf-8")
    (tmp_path / "preprocessing.yaml").write_text(
        "preprocessing:\n  output:\n    save_intermediates: false\n  downsample:\n    sfreq_hz: 512\n",
        encoding="utf-8",
    )

    cfg = load_project_config(cfg_path, sections=("preprocessing",))

    assert cfg.raw["preprocessing"]["output"]["save_intermediates"] is True
    assert cfg.raw["preprocessing"]["downsample"]["sfreq_hz"] == 512


def test_load_project_config_loads_requested_trf_section_with_sibling_precedence(tmp_path: Path) -> None:
    """Dedicated `trf.yaml` should remain the source of truth for TRF defaults."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "trf:\n"
        "  enabled: true\n"
        "  predictors:\n"
        "    - self_speech_envelope\n"
        "    - other_speech_envelope\n",
        encoding="utf-8",
    )
    (tmp_path / "trf.yaml").write_text(
        "trf:\n"
        "  predictors:\n"
        "    - self_speech_envelope\n"
        "    - other_speech_envelope\n"
        "    - self_f0\n"
        "    - other_f0\n",
        encoding="utf-8",
    )

    cfg = load_project_config(cfg_path, sections=("trf",))

    assert cfg.raw["trf"]["enabled"] is True
    assert cfg.raw["trf"]["predictors"] == [
        "self_speech_envelope",
        "other_speech_envelope",
        "self_f0",
        "other_f0",
    ]


def test_load_project_config_rejects_unknown_requested_section(tmp_path: Path) -> None:
    """Unknown section names should fail fast."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("project:\n  name: demo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config sections requested"):
        load_project_config(cfg_path, sections=("bogus",))


def test_load_project_config_loads_requested_viz_section(tmp_path: Path) -> None:
    """Requested visualization fragments should be merged like other standard sections."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("viz:\n  figure_defaults:\n    dpi: 150\n", encoding="utf-8")
    (tmp_path / "viz.yaml").write_text(
        "viz:\n  speech_artefact:\n    dpi: 300\n    task: conversation\n",
        encoding="utf-8",
    )

    cfg = load_project_config(cfg_path, sections=("viz",))

    assert cfg.raw["viz"]["figure_defaults"]["dpi"] == 150
    assert cfg.raw["viz"]["speech_artefact"]["task"] == "conversation"
