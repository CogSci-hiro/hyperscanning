"""Tests for TRF configuration validation."""

from __future__ import annotations

import pytest

from hyper.trf.config import TrfConfig


def _base_trf_mapping() -> dict:
    return {
        "enabled": True,
        "predictors": ["self_speech_envelope", "other_speech_envelope"],
        "qc_predictors": ["self_speech_envelope"],
        "ablation_targets": ["other_speech_envelope"],
        "target_sfreq": 64.0,
        "lags": {"tmin_seconds": -0.2, "tmax_seconds": 0.6},
        "conversation": {"duration_seconds": 240.0, "start_source": "metadata"},
        "segmentation": {
            "method": "blockwise_within_run",
            "n_blocks_per_run": 4,
            "drop_remainder": False,
            "min_block_duration_seconds": 10.0,
        },
        "cv": {
            "outer": {"splitter": "group_kfold", "n_splits": 5, "group_by": "run_id"},
            "inner": {"splitter": "group_kfold", "n_splits": 4, "group_by": "segment_id"},
        },
        "hyperparameters": {"alpha": {"scale": "logspace", "start_exp": -1, "stop_exp": 8, "num": 10}},
        "model": {"estimator": "ridge", "fit_intercept": False, "standardize_x": True, "standardize_y": False},
        "scoring": {"primary": "pearsonr"},
        "outputs": {},
    }


def test_trf_config_rejects_ablation_targets_outside_predictors() -> None:
    """Feature ablation targets should refer to configured full-model predictors only."""
    mapping = _base_trf_mapping()
    mapping["ablation_targets"] = ["self_surprisal"]

    with pytest.raises(ValueError, match="ablation_targets must resolve to members drawn from trf.predictors"):
        TrfConfig.from_mapping(mapping)


def test_trf_config_rejects_empty_qc_predictors() -> None:
    """EEG QC needs a non-empty predictor list."""
    mapping = _base_trf_mapping()
    mapping["qc_predictors"] = []

    with pytest.raises(ValueError, match="qc_predictors must contain at least one predictor"):
        TrfConfig.from_mapping(mapping)


def test_trf_config_rejects_unknown_qc_predictors() -> None:
    """QC predictors should use the same supported predictor vocabulary as the main TRF model."""
    mapping = _base_trf_mapping()
    mapping["qc_predictors"] = ["does_not_exist"]

    with pytest.raises(ValueError, match="qc_predictors must be drawn from the supported predictor set"):
        TrfConfig.from_mapping(mapping)


def test_trf_config_accepts_grouped_word_class_ablation_targets() -> None:
    """Grouped word-class ablations should validate when all member predictors are present."""
    mapping = _base_trf_mapping()
    mapping["predictors"] = [
        "other_speech_envelope",
        "other_function_word_onsets",
        "other_content_word_onsets",
    ]
    mapping["qc_predictors"] = ["other_speech_envelope"]
    mapping["ablation_targets"] = ["other_word_class_onsets"]

    cfg = TrfConfig.from_mapping(mapping)

    assert cfg.ablation_targets == ("other_word_class_onsets",)


def test_trf_config_accepts_grouped_word_class_predictor() -> None:
    """Grouped word-class predictors should be accepted as regular TRF predictors."""
    mapping = _base_trf_mapping()
    mapping["predictors"] = ["other_speech_envelope", "other_word_class_onsets"]
    mapping["qc_predictors"] = ["other_word_class_onsets"]
    mapping["ablation_targets"] = ["other_word_class_onsets"]

    cfg = TrfConfig.from_mapping(mapping)

    assert cfg.predictors == ("other_speech_envelope", "other_word_class_onsets")
    assert cfg.qc_predictors == ("other_word_class_onsets",)
