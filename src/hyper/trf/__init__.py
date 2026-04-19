"""TRF benchmark pipeline built around run-wise continuous data."""

from .config import TrfConfig, load_trf_config
from .pipeline import (
    TrfRunInput,
    TrfQcScoreSummary,
    TrfSegment,
    build_lagged_segment_design,
    build_circular_shifted_eeg_qc_null_run_inputs,
    build_reduced_predictor_list,
    compute_score_delta,
    crop_target_to_conversation_window,
    fit_nested_trf,
    fit_subject_trf_score,
    load_trf_run_inputs,
    prepare_group_kfold,
    prepare_trf_segment_designs,
    run_trf_pipeline,
    run_trf_qc_score_tables,
    split_run_into_segments,
)

__all__ = [
    "TrfConfig",
    "TrfQcScoreSummary",
    "TrfRunInput",
    "TrfSegment",
    "build_lagged_segment_design",
    "build_circular_shifted_eeg_qc_null_run_inputs",
    "build_reduced_predictor_list",
    "compute_score_delta",
    "crop_target_to_conversation_window",
    "fit_nested_trf",
    "fit_subject_trf_score",
    "load_trf_config",
    "load_trf_run_inputs",
    "prepare_group_kfold",
    "prepare_trf_segment_designs",
    "run_trf_pipeline",
    "run_trf_qc_score_tables",
    "split_run_into_segments",
]
