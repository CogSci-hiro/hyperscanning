"""Joint-plot helpers for channel-by-time EEG maps."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Agg")


def infer_sfreq(times: np.ndarray) -> float:
    """Infer sampling frequency from a 1D time vector."""
    time_array = np.asarray(times, dtype=float)
    if time_array.ndim != 1:
        raise ValueError("times must be a 1D array")
    if time_array.size < 2:
        return 1.0
    deltas = np.diff(time_array)
    finite_deltas = deltas[np.isfinite(deltas)]
    if finite_deltas.size == 0:
        return 1.0
    delta = float(np.median(finite_deltas))
    if delta <= 0:
        return 1.0
    return 1.0 / delta


def sanitize_token(value: str) -> str:
    """Return a filesystem-safe token."""
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return token or "item"


def _extract_overlapping_channels(message: str) -> list[str]:
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    if not lines:
        return []
    return [name.strip() for name in lines[-1].split(",") if name.strip()]


def pick_peak_indices(score: np.ndarray, *, n_peaks: int, min_separation: int) -> list[int]:
    """Pick separated peak indices from a 1D score array."""
    ranking = np.argsort(np.asarray(score, dtype=float))[::-1]
    peaks: list[int] = []
    for index in ranking:
        if not np.isfinite(score[index]):
            continue
        if any(abs(int(index) - existing) < min_separation for existing in peaks):
            continue
        peaks.append(int(index))
        if len(peaks) >= n_peaks:
            break
    return sorted(peaks)


def resolve_joint_times(
    beta_map: np.ndarray,
    times: np.ndarray,
    *,
    joint_times: str | Sequence[float] = "peaks",
    significance_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Resolve explicit joint times or choose data-driven peak times."""
    beta_array = np.asarray(beta_map, dtype=float)
    time_array = np.asarray(times, dtype=float)
    if isinstance(joint_times, str) and joint_times != "peaks":
        raise ValueError(f"Unsupported joint_times mode: {joint_times}")
    if not isinstance(joint_times, str):
        return np.asarray(joint_times, dtype=float)

    score = np.nanmax(np.abs(beta_array), axis=0)
    if significance_mask is not None:
        significant_times = np.any(np.asarray(significance_mask, dtype=bool), axis=0)
        if np.any(significant_times):
            score = np.where(significant_times, score, -np.inf)
    min_separation = max(1, int(round(time_array.shape[0] / 10.0)))
    peak_indices = pick_peak_indices(score, n_peaks=3, min_separation=min_separation)
    if not peak_indices:
        fallback = np.linspace(0, max(0, time_array.shape[0] - 1), num=min(3, time_array.shape[0]), dtype=int)
        peak_indices = sorted({int(index) for index in fallback.tolist()})
    return time_array[np.asarray(peak_indices, dtype=int)]


def contiguous_true_spans(mask: np.ndarray, times: np.ndarray) -> list[tuple[float, float]]:
    """Convert a boolean time mask into contiguous spans."""
    boolean_mask = np.asarray(mask, dtype=bool)
    time_array = np.asarray(times, dtype=float)
    if boolean_mask.ndim != 1 or boolean_mask.shape[0] != time_array.shape[0]:
        raise ValueError("mask must be a 1D boolean array aligned to times")
    if not np.any(boolean_mask):
        return []

    if time_array.shape[0] > 1:
        finite_deltas = np.diff(time_array)
        finite_deltas = finite_deltas[np.isfinite(finite_deltas) & (finite_deltas > 0)]
        half_step = float(np.median(finite_deltas)) / 2.0 if finite_deltas.size else 0.0
    else:
        half_step = 0.0

    spans: list[tuple[float, float]] = []
    start_index: int | None = None
    for index, flag in enumerate(boolean_mask):
        if flag and start_index is None:
            start_index = index
        if not flag and start_index is not None:
            spans.append((float(time_array[start_index]) - half_step, float(time_array[index - 1]) + half_step))
            start_index = None
    if start_index is not None:
        spans.append((float(time_array[start_index]) - half_step, float(time_array[-1]) + half_step))
    return spans


def plot_joint_map(
    beta_map: np.ndarray,
    *,
    times: np.ndarray,
    channel_names: Sequence[str],
    output_stem: Path,
    title: str | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    line_width: float = 2.5,
    joint_times: str | Sequence[float] = "peaks",
    significance_mask: np.ndarray | None = None,
) -> list[Path]:
    """Render one channel-by-time map as an MNE joint plot."""
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

    import mne

    beta_array = np.asarray(beta_map, dtype=float)
    time_array = np.asarray(times, dtype=float)
    if beta_array.ndim != 2:
        raise ValueError("beta_map must have shape (n_channels, n_times)")
    if beta_array.shape[0] != len(channel_names):
        raise ValueError("beta_map channel dimension must match channel_names")
    if beta_array.shape[1] != time_array.shape[0]:
        raise ValueError("beta_map time dimension must match times")

    significance_array = None
    if significance_mask is not None:
        significance_array = np.asarray(significance_mask, dtype=bool)
        if significance_array.shape != beta_array.shape:
            raise ValueError("significance_mask must match beta_map")

    info = mne.create_info(list(channel_names), sfreq=infer_sfreq(time_array), ch_types=["eeg"] * len(channel_names))
    try:
        info.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass

    evoked = mne.EvokedArray(beta_array, info, tmin=float(time_array[0]), nave=1, comment=title or "beta")
    selected_joint_times = resolve_joint_times(
        beta_array,
        time_array,
        joint_times=joint_times,
        significance_mask=significance_array,
    )

    topomap_args: dict[str, object] = {}
    if significance_array is not None:
        topomap_args["mask"] = significance_array
        topomap_args["mask_params"] = {
            "marker": "o",
            "markerfacecolor": "0.6",
            "markeredgecolor": "0.6",
            "markeredgewidth": 0.0,
            "linestyle": "None",
            "markersize": 8.0,
        }

    try:
        figure = evoked.plot_joint(times=selected_joint_times, title=title, show=False, topomap_args=topomap_args)
    except ValueError as exc:
        if "overlapping positions" not in str(exc):
            raise
        overlapping_channels = _extract_overlapping_channels(str(exc))
        keep_channels = [name for name in evoked.ch_names if name not in overlapping_channels]
        if not keep_channels:
            raise
        keep_indices = [evoked.ch_names.index(name) for name in keep_channels]
        reduced_topomap_args = dict(topomap_args)
        if significance_array is not None:
            reduced_topomap_args["mask"] = significance_array[np.asarray(keep_indices, dtype=int)]
        figure = evoked.copy().pick(keep_channels).plot_joint(
            times=selected_joint_times,
            title=title,
            show=False,
            topomap_args=reduced_topomap_args,
        )

    for axis in figure.axes:
        for line in axis.lines:
            line.set_linewidth(float(line_width))

    if significance_array is not None and np.any(significance_array):
        significant_time_mask = np.any(significance_array, axis=0)
        line_axes = [axis for axis in figure.axes if axis.lines]
        if line_axes:
            main_axis = max(line_axes, key=lambda axis: len(axis.lines))
            for start_time, end_time in contiguous_true_spans(significant_time_mask, time_array):
                main_axis.axvspan(start_time, end_time, color="0.85", alpha=0.7, zorder=0)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for fmt in formats:
        output_path = output_stem.with_suffix(f".{fmt}")
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(output_path)
    plt.close(figure)
    return written_paths
