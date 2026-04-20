"""Visualization helpers and shared style definitions."""

from .joint import contiguous_true_spans, infer_sfreq, pick_peak_indices, plot_joint_map, resolve_joint_times, sanitize_token
from .style import Style


def build_speech_artefact_summary_figure(*args, **kwargs):
    """Lazily import the speech artefact figure builder."""
    from .speech_artefact_qc import build_speech_artefact_summary_figure as _build

    return _build(*args, **kwargs)


def build_trf_score_qc_figure(*args, **kwargs):
    """Lazily import the TRF-score figure builder."""
    from .trf_score_qc import build_trf_score_qc_figure as _build

    return _build(*args, **kwargs)


def build_trf_main_figure(*args, **kwargs):
    """Lazily import the TRF main-figure builder."""
    from .trf_main_figure import build_trf_main_figure as _build

    return _build(*args, **kwargs)

__all__ = [
    "Style",
    "build_trf_main_figure",
    "build_speech_artefact_summary_figure",
    "build_trf_score_qc_figure",
    "contiguous_true_spans",
    "infer_sfreq",
    "pick_peak_indices",
    "plot_joint_map",
    "resolve_joint_times",
    "sanitize_token",
]
