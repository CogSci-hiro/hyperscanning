"""Configuration models for the TRF benchmark pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from hyper.config import ProjectConfig

DEFAULT_TRF_TASK: str = "conversation"
SUPPORTED_PRIMARY_SCORE: str = "pearsonr"
SUPPORTED_TRF_PREDICTORS: frozenset[str] = frozenset(
    {
        "speech_envelope",
        "envelope",
        "self_speech_envelope",
        "other_speech_envelope",
        "self_envelope",
        "other_envelope",
        "f0",
        "self_f0",
        "other_f0",
        "self_f1_f2",
        "other_f1_f2",
        "self_f1",
        "other_f1",
        "self_f2",
        "other_f2",
        "self_phoneme_onsets",
        "other_phoneme_onsets",
        "self_syllable_onsets",
        "other_syllable_onsets",
        "self_token_onsets",
        "other_token_onsets",
        "self_entropy",
        "other_entropy",
        "self_surprisal",
        "other_surprisal",
    }
)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Require a mapping-like config section."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for {name}, got {type(value)!r}")
    return value


def _as_bool(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    """Read a bool config field with a default."""
    return bool(mapping.get(key, default))


def _as_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    """Read a float config field with a default."""
    return float(mapping.get(key, default))


def _as_int(mapping: Mapping[str, Any], key: str, default: int) -> int:
    """Read an integer config field with a default."""
    return int(mapping.get(key, default))


@dataclass(frozen=True, slots=True)
class TrfLagConfig:
    """Lag window for TRF fitting."""

    tmin_seconds: float = -0.20
    tmax_seconds: float = 0.60

    def validate(self) -> None:
        """Validate lag settings."""
        if self.tmin_seconds >= self.tmax_seconds:
            raise ValueError("TRF lag window requires tmin_seconds < tmax_seconds.")


@dataclass(frozen=True, slots=True)
class TrfConversationConfig:
    """Conversation cropping settings."""

    duration_seconds: float = 240.0
    start_source: str = "metadata"

    def validate(self) -> None:
        """Validate conversation-window settings."""
        if self.duration_seconds <= 0:
            raise ValueError("TRF conversation duration_seconds must be positive.")
        if self.start_source != "metadata":
            raise ValueError("TRF currently supports conversation.start_source='metadata' only.")


@dataclass(frozen=True, slots=True)
class TrfSegmentationConfig:
    """Blockwise segmentation settings."""

    method: str = "blockwise_within_run"
    n_blocks_per_run: int = 4
    drop_remainder: bool = False
    min_block_duration_seconds: float = 10.0

    def validate(self) -> None:
        """Validate segmentation settings."""
        if self.method != "blockwise_within_run":
            raise ValueError("TRF currently supports segmentation.method='blockwise_within_run' only.")
        if self.n_blocks_per_run <= 0:
            raise ValueError("TRF segmentation.n_blocks_per_run must be positive.")
        if self.min_block_duration_seconds <= 0:
            raise ValueError("TRF segmentation.min_block_duration_seconds must be positive.")


@dataclass(frozen=True, slots=True)
class TrfCvSplitConfig:
    """One level of grouped CV configuration."""

    splitter: str
    n_splits: int
    group_by: str

    def validate(self, *, level_name: str) -> None:
        """Validate grouped CV options."""
        if self.splitter != "group_kfold":
            raise ValueError(f"TRF {level_name}.splitter must be 'group_kfold'.")
        if self.n_splits <= 1:
            raise ValueError(f"TRF {level_name}.n_splits must be greater than 1.")
        if self.group_by not in {"run_id", "segment_id"}:
            raise ValueError(f"TRF {level_name}.group_by must be 'run_id' or 'segment_id'.")


@dataclass(frozen=True, slots=True)
class TrfCvConfig:
    """Nested CV settings."""

    outer: TrfCvSplitConfig
    inner: TrfCvSplitConfig

    def validate(self) -> None:
        """Validate nested CV settings."""
        self.outer.validate(level_name="cv.outer")
        self.inner.validate(level_name="cv.inner")


@dataclass(frozen=True, slots=True)
class TrfAlphaGridConfig:
    """Ridge alpha grid configuration."""

    scale: str = "logspace"
    start_exp: float = -1.0
    stop_exp: float = 8.0
    num: int = 10

    def validate(self) -> None:
        """Validate alpha grid options."""
        if self.scale != "logspace":
            raise ValueError("TRF hyperparameters.alpha.scale must be 'logspace'.")
        if self.num <= 0:
            raise ValueError("TRF hyperparameters.alpha.num must be positive.")

    def values(self) -> np.ndarray:
        """Return the resolved alpha grid."""
        self.validate()
        return np.logspace(self.start_exp, self.stop_exp, self.num, dtype=float)


@dataclass(frozen=True, slots=True)
class TrfModelConfig:
    """Model-fitting options."""

    estimator: str = "ridge"
    fit_intercept: bool = False
    standardize_x: bool = True
    standardize_y: bool = False

    def validate(self) -> None:
        """Validate model settings."""
        if self.estimator != "ridge":
            raise ValueError("TRF model.estimator must be 'ridge'.")
        if self.fit_intercept:
            raise ValueError("TRF currently supports model.fit_intercept=false only.")


@dataclass(frozen=True, slots=True)
class TrfScoringConfig:
    """Scoring configuration."""

    primary: str = SUPPORTED_PRIMARY_SCORE

    def validate(self) -> None:
        """Validate scoring settings."""
        if self.primary != SUPPORTED_PRIMARY_SCORE:
            raise ValueError("TRF scoring.primary must be 'pearsonr'.")


@dataclass(frozen=True, slots=True)
class TrfOutputConfig:
    """Output toggles for persisted TRF artifacts."""

    save_fold_scores: bool = True
    save_selected_alpha_per_fold: bool = True
    save_coefficients: bool = True
    save_design_info: bool = True


@dataclass(frozen=True, slots=True)
class TrfConfig:
    """Resolved TRF configuration.

    Usage example
    -------------
        trf_cfg = load_trf_config(project_cfg)
        if trf_cfg.enabled:
            print(trf_cfg.alpha_values())
    """

    enabled: bool = False
    predictors: tuple[str, ...] = ("self_speech_envelope",)
    qc_predictors: tuple[str, ...] = ("self_speech_envelope",)
    ablation_targets: tuple[str, ...] = ()
    target_sfreq: float = 64.0
    lags: TrfLagConfig = TrfLagConfig()
    conversation: TrfConversationConfig = TrfConversationConfig()
    segmentation: TrfSegmentationConfig = TrfSegmentationConfig()
    cv: TrfCvConfig = TrfCvConfig(
        outer=TrfCvSplitConfig(splitter="group_kfold", n_splits=5, group_by="run_id"),
        inner=TrfCvSplitConfig(splitter="group_kfold", n_splits=4, group_by="segment_id"),
    )
    alpha: TrfAlphaGridConfig = TrfAlphaGridConfig()
    model: TrfModelConfig = TrfModelConfig()
    scoring: TrfScoringConfig = TrfScoringConfig()
    random_seed: int = 42
    outputs: TrfOutputConfig = TrfOutputConfig()

    def validate(self) -> None:
        """Validate the full TRF configuration."""
        if len(self.predictors) == 0:
            raise ValueError("TRF requires at least one predictor.")
        unsupported_predictors = [name for name in self.predictors if name not in SUPPORTED_TRF_PREDICTORS]
        if unsupported_predictors:
            raise ValueError(
                "TRF predictors must be drawn from the supported continuous set "
                f"{sorted(SUPPORTED_TRF_PREDICTORS)!r}; got unsupported predictors {unsupported_predictors!r}."
            )
        if len(self.qc_predictors) == 0:
            raise ValueError("TRF qc_predictors must contain at least one predictor.")
        unsupported_qc_predictors = [name for name in self.qc_predictors if name not in SUPPORTED_TRF_PREDICTORS]
        if unsupported_qc_predictors:
            raise ValueError(
                "TRF qc_predictors must be drawn from the supported predictor set "
                f"{sorted(SUPPORTED_TRF_PREDICTORS)!r}; got unsupported predictors {unsupported_qc_predictors!r}."
            )
        missing_ablation_targets = [name for name in self.ablation_targets if name not in self.predictors]
        if missing_ablation_targets:
            raise ValueError(
                "TRF ablation_targets must be drawn from trf.predictors; "
                f"got invalid targets {missing_ablation_targets!r} for predictors {list(self.predictors)!r}."
            )
        empty_reduced_targets = [name for name in self.ablation_targets if len(self.predictors) == 1 and name in self.predictors]
        if empty_reduced_targets:
            raise ValueError(
                "TRF ablation_targets cannot remove the only predictor from the full model; "
                f"got {empty_reduced_targets!r}."
            )
        if self.target_sfreq <= 0:
            raise ValueError("TRF target_sfreq must be positive.")
        self.lags.validate()
        self.conversation.validate()
        self.segmentation.validate()
        self.cv.validate()
        self.alpha.validate()
        self.model.validate()
        self.scoring.validate()

    def alpha_values(self) -> np.ndarray:
        """Return the resolved ridge alpha grid."""
        return self.alpha.values()

    @staticmethod
    def from_mapping(mapping: Mapping[str, Any] | None) -> TrfConfig:
        """Build a validated TRF config from a raw mapping."""
        if mapping is None:
            cfg = TrfConfig()
            cfg.validate()
            return cfg

        root = _require_mapping(mapping, name="trf")
        lags_cfg = _require_mapping(root.get("lags", {}), name="trf.lags")
        conversation_cfg = _require_mapping(root.get("conversation", {}), name="trf.conversation")
        segmentation_cfg = _require_mapping(root.get("segmentation", {}), name="trf.segmentation")
        cv_cfg = _require_mapping(root.get("cv", {}), name="trf.cv")
        outer_cfg = _require_mapping(cv_cfg.get("outer", {}), name="trf.cv.outer")
        inner_cfg = _require_mapping(cv_cfg.get("inner", {}), name="trf.cv.inner")
        hyper_cfg = _require_mapping(root.get("hyperparameters", {}), name="trf.hyperparameters")
        alpha_cfg = _require_mapping(hyper_cfg.get("alpha", {}), name="trf.hyperparameters.alpha")
        model_cfg = _require_mapping(root.get("model", {}), name="trf.model")
        scoring_cfg = _require_mapping(root.get("scoring", {}), name="trf.scoring")
        outputs_cfg = _require_mapping(root.get("outputs", {}), name="trf.outputs")

        predictors_raw = root.get("predictors", ("self_speech_envelope",))
        predictors = tuple(str(value) for value in predictors_raw)
        qc_predictors_raw = root.get("qc_predictors", predictors_raw)
        qc_predictors = tuple(str(value) for value in qc_predictors_raw)
        ablation_targets_raw = root.get("ablation_targets", ())
        ablation_targets = tuple(str(value) for value in ablation_targets_raw)

        cfg = TrfConfig(
            enabled=_as_bool(root, "enabled", False),
            predictors=predictors,
            qc_predictors=qc_predictors,
            ablation_targets=ablation_targets,
            target_sfreq=_as_float(root, "target_sfreq", 64.0),
            lags=TrfLagConfig(
                tmin_seconds=_as_float(lags_cfg, "tmin_seconds", -0.20),
                tmax_seconds=_as_float(lags_cfg, "tmax_seconds", 0.60),
            ),
            conversation=TrfConversationConfig(
                duration_seconds=_as_float(conversation_cfg, "duration_seconds", 240.0),
                start_source=str(conversation_cfg.get("start_source", "metadata")),
            ),
            segmentation=TrfSegmentationConfig(
                method=str(segmentation_cfg.get("method", "blockwise_within_run")),
                n_blocks_per_run=_as_int(segmentation_cfg, "n_blocks_per_run", 4),
                drop_remainder=_as_bool(segmentation_cfg, "drop_remainder", False),
                min_block_duration_seconds=_as_float(segmentation_cfg, "min_block_duration_seconds", 10.0),
            ),
            cv=TrfCvConfig(
                outer=TrfCvSplitConfig(
                    splitter=str(outer_cfg.get("splitter", "group_kfold")),
                    n_splits=_as_int(outer_cfg, "n_splits", 5),
                    group_by=str(outer_cfg.get("group_by", "run_id")),
                ),
                inner=TrfCvSplitConfig(
                    splitter=str(inner_cfg.get("splitter", "group_kfold")),
                    n_splits=_as_int(inner_cfg, "n_splits", 4),
                    group_by=str(inner_cfg.get("group_by", "segment_id")),
                ),
            ),
            alpha=TrfAlphaGridConfig(
                scale=str(alpha_cfg.get("scale", "logspace")),
                start_exp=_as_float(alpha_cfg, "start_exp", -1.0),
                stop_exp=_as_float(alpha_cfg, "stop_exp", 8.0),
                num=_as_int(alpha_cfg, "num", 10),
            ),
            model=TrfModelConfig(
                estimator=str(model_cfg.get("estimator", "ridge")),
                fit_intercept=_as_bool(model_cfg, "fit_intercept", False),
                standardize_x=_as_bool(model_cfg, "standardize_x", True),
                standardize_y=_as_bool(model_cfg, "standardize_y", False),
            ),
            scoring=TrfScoringConfig(primary=str(scoring_cfg.get("primary", SUPPORTED_PRIMARY_SCORE))),
            random_seed=_as_int(root, "random_seed", 42),
            outputs=TrfOutputConfig(
                save_fold_scores=_as_bool(outputs_cfg, "save_fold_scores", True),
                save_selected_alpha_per_fold=_as_bool(outputs_cfg, "save_selected_alpha_per_fold", True),
                save_coefficients=_as_bool(outputs_cfg, "save_coefficients", True),
                save_design_info=_as_bool(outputs_cfg, "save_design_info", True),
            ),
        )
        cfg.validate()
        return cfg


def load_trf_config(cfg: ProjectConfig | Mapping[str, Any]) -> TrfConfig:
    """Load the TRF config from a `ProjectConfig` or raw mapping."""
    raw = cfg.raw if isinstance(cfg, ProjectConfig) else cfg
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected project config mapping, got {type(raw)!r}")
    return TrfConfig.from_mapping(raw.get("trf"))
