"""TRF benchmark pipeline built around run-wise continuous data."""

from .config import TrfConfig, load_trf_config
from .pipeline import (
    TrfRunInput,
    TrfSegment,
    build_lagged_segment_design,
    crop_run_to_conversation_window,
    fit_nested_trf,
    load_trf_run_inputs,
    prepare_group_kfold,
    run_trf_pipeline,
    split_run_into_segments,
)

__all__ = [
    "TrfConfig",
    "TrfRunInput",
    "TrfSegment",
    "build_lagged_segment_design",
    "crop_run_to_conversation_window",
    "fit_nested_trf",
    "load_trf_config",
    "load_trf_run_inputs",
    "prepare_group_kfold",
    "run_trf_pipeline",
    "split_run_into_segments",
]
