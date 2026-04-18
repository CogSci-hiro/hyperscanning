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


def test_load_project_config_merges_sibling_paths_yaml(tmp_path: Path) -> None:
    """Sibling `paths.yaml` should supply machine-local path settings."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("project:\n  name: demo\n", encoding="utf-8")
    (tmp_path / "paths.yaml").write_text(
        "paths:\n  bids_root: /bids\n  derived_root: /derived\n  lm_feature_root: /lm\n  precomputed_ica_root: /ica\n  annotation_root: /ann\n",
        encoding="utf-8",
    )

    cfg = load_project_config(cfg_path)

    assert cfg.raw["paths"]["bids_root"] == "/bids"
    assert cfg.raw["paths"]["lm_feature_root"] == "/lm"
    assert cfg.raw["paths"]["precomputed_ica_root"] == "/ica"


def test_load_project_config_prefers_inline_paths_over_sibling_paths_yaml(tmp_path: Path) -> None:
    """Temporary or test configs should be able to override the shared paths file."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "paths:\n  derived_root: /override\n  precomputed_ica_root: /override-ica\n",
        encoding="utf-8",
    )
    (tmp_path / "paths.yaml").write_text(
        "paths:\n  bids_root: /bids\n  derived_root: /derived\n  lm_feature_root: /lm\n  precomputed_ica_root: /ica\n  annotation_root: /ann\n",
        encoding="utf-8",
    )

    cfg = load_project_config(cfg_path)

    assert cfg.raw["paths"]["bids_root"] == "/bids"
    assert cfg.raw["paths"]["derived_root"] == "/override"
    assert cfg.raw["paths"]["precomputed_ica_root"] == "/override-ica"


def test_load_project_config_merges_sibling_preprocessing_yaml(tmp_path: Path) -> None:
    """Sibling `preprocessing.yaml` should populate nested preprocessing settings."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("project:\n  name: demo\n", encoding="utf-8")
    (tmp_path / "preprocessing.yaml").write_text(
        "preprocessing:\n  downsample:\n    sfreq_hz: 512\n  pipeline_order:\n    - downsample\n",
        encoding="utf-8",
    )

    cfg = load_project_config(cfg_path)

    assert cfg.raw["preprocessing"]["downsample"]["sfreq_hz"] == 512
    assert cfg.raw["preprocessing"]["pipeline_order"] == ["downsample"]


def test_load_project_config_prefers_inline_preprocessing_over_sibling_file(tmp_path: Path) -> None:
    """Temporary configs should still be able to override shared preprocessing settings."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("preprocessing:\n  output:\n    save_intermediates: true\n", encoding="utf-8")
    (tmp_path / "preprocessing.yaml").write_text(
        "preprocessing:\n  output:\n    save_intermediates: false\n  downsample:\n    sfreq_hz: 512\n",
        encoding="utf-8",
    )

    cfg = load_project_config(cfg_path)

    assert cfg.raw["preprocessing"]["output"]["save_intermediates"] is True
    assert cfg.raw["preprocessing"]["downsample"]["sfreq_hz"] == 512


def test_load_project_config_merges_sibling_trf_yaml(tmp_path: Path) -> None:
    """Sibling `trf.yaml` should populate the dedicated TRF block."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("project:\n  name: demo\n", encoding="utf-8")
    (tmp_path / "trf.yaml").write_text(
        "trf:\n  enabled: true\n  target_sfreq: 64\n  predictors:\n    - self_speech_envelope\n    - other_speech_envelope\n",
        encoding="utf-8",
    )

    cfg = load_project_config(cfg_path)

    assert cfg.raw["trf"]["enabled"] is True
    assert cfg.raw["trf"]["target_sfreq"] == 64
    assert cfg.raw["trf"]["predictors"] == ["self_speech_envelope", "other_speech_envelope"]


def test_load_project_config_prefers_sibling_trf_over_inline_trf(tmp_path: Path) -> None:
    """Dedicated `trf.yaml` should override stale inline TRF defaults."""
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

    cfg = load_project_config(cfg_path)

    assert cfg.raw["trf"]["enabled"] is True
    assert cfg.raw["trf"]["predictors"] == [
        "self_speech_envelope",
        "other_speech_envelope",
        "self_f0",
        "other_f0",
    ]
