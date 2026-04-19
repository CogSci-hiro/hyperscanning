"""Visualization helpers and shared style definitions."""

from .joint import contiguous_true_spans, infer_sfreq, pick_peak_indices, plot_joint_map, resolve_joint_times, sanitize_token
from .speech_artefact_qc import build_speech_artefact_summary_figure
from .style import Style
from .trf_score_qc import build_trf_score_qc_figure

__all__ = [
    "Style",
    "build_speech_artefact_summary_figure",
    "build_trf_score_qc_figure",
    "contiguous_true_spans",
    "infer_sfreq",
    "pick_peak_indices",
    "plot_joint_map",
    "resolve_joint_times",
    "sanitize_token",
]
