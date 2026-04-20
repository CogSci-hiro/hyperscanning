"""Speech artefact summary plotting helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from hyper.config import ProjectConfig
from hyper.paths import ProjectPaths
from hyper.preprocessing.downsampling import downsample_edf_to_fif
from hyper.preprocessing.ica import apply_ica_fif_to_fif
from hyper.preprocessing.interpolation import interpolate_bads_fif_to_fif
from hyper.preprocessing.reref import rereference_fif_to_fif

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne

matplotlib.use("Agg")


@dataclass(frozen=True, slots=True)
class PsdSummary:
    """Group-average PSD summary."""

    info: mne.Info
    frequencies_hz: np.ndarray
    channel_power: np.ndarray


@dataclass(frozen=True, slots=True)
class ComponentCountSummary:
    """ICA include/exclude component counts."""

    subject_ids: tuple[str, ...]
    total_counts: np.ndarray
    included_counts: np.ndarray
    excluded_counts: np.ndarray
    bad_channel_counts: np.ndarray


RUN_STEM_PATTERN = re.compile(
    r"^(?P<subject>sub-[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)",
    flags=re.IGNORECASE,
)
EXPECTED_EEG_CHANNELS = 64.0
FONT_SCALE = 2.0
Y_AXIS_LABEL_SCALE = 0.7
THIRD_PANEL_XTICK_SCALE = 0.5
LEGEND_FONT_SCALE = 0.5
FIRST_SECOND_GAP_REDUCTION = 0.055
SECOND_THIRD_GAP_REDUCTION = 0.04


def _count_bad_channels(channels_tsv_path: Path) -> int:
    """Count channels marked as bad in a BIDS channels.tsv sidecar."""
    table = pd.read_csv(channels_tsv_path, sep="\t")
    if "name" not in table.columns or "status" not in table.columns:
        return 0
    status = table["status"].astype(str).str.lower()
    return int((status == "bad").sum())


def _speech_artefact_cfg(cfg: ProjectConfig) -> dict:
    """Return the speech artefact visualization settings."""
    raw_cfg = cfg.raw.get("viz", {}).get("speech_artefact", {})
    if not isinstance(raw_cfg, dict):
        raise ValueError("Config section `viz.speech_artefact` must be a mapping.")
    return raw_cfg


def _figure_size(cfg: ProjectConfig) -> tuple[float, float]:
    """Return the requested output figure size."""
    speech_cfg = _speech_artefact_cfg(cfg)
    figsize_cfg = speech_cfg.get("figsize", {})
    if not isinstance(figsize_cfg, dict):
        figsize_cfg = {}
    return (
        float(figsize_cfg.get("width", 15.0)),
        float(figsize_cfg.get("height", 4.5)),
    )


def _figure_dpi(cfg: ProjectConfig) -> int:
    """Return the requested output DPI."""
    return int(_speech_artefact_cfg(cfg).get("dpi", 300))


def _psd_settings(cfg: ProjectConfig) -> tuple[str, float, float, int]:
    """Return PSD estimation settings from config."""
    psd_cfg = _speech_artefact_cfg(cfg).get("psd", {})
    if not isinstance(psd_cfg, dict):
        psd_cfg = {}
    method = str(psd_cfg.get("method", "welch")).strip().lower()
    if method not in {"welch", "multitaper"}:
        raise ValueError("PSD method must be either 'welch' or 'multitaper'.")
    return (
        method,
        float(psd_cfg.get("fmin_hz", 1.0)),
        float(psd_cfg.get("fmax_hz", 40.0)),
        int(psd_cfg.get("n_fft", 256)),
    )


def _load_average_psd(
    paths: Sequence[Path],
    *,
    method: str,
    fmin_hz: float,
    fmax_hz: float,
    n_fft: int,
) -> PsdSummary:
    """Compute a grand-average per-channel PSD across all provided FIF files."""
    per_recording_psd: list[np.ndarray] = []
    frequency_axis: np.ndarray | None = None
    reference_channel_names: list[str] | None = None
    reference_info: mne.Info | None = None

    for path in paths:
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude=())
        if len(eeg_picks) == 0:
            raise ValueError(f"No EEG channels available for PSD aggregation in {path}.")
        channel_names = [str(raw.ch_names[index]) for index in eeg_picks]
        if reference_channel_names is None:
            reference_channel_names = channel_names
            reference_info = raw.copy().pick(eeg_picks).info.copy()
        elif channel_names != reference_channel_names:
            raise ValueError("All PSD inputs must share the same EEG channel order to preserve montage.")
        compute_kwargs = {
            "method": method,
            "fmin": fmin_hz,
            "fmax": fmax_hz,
            "verbose": "ERROR",
        }
        if method == "welch":
            compute_kwargs["n_fft"] = n_fft
        psd = raw.compute_psd(**compute_kwargs)
        spectrum = np.asarray(psd.get_data(), dtype=float)
        freqs = np.asarray(psd.freqs, dtype=float)
        if spectrum.ndim != 2:
            raise ValueError(f"Expected PSD with shape (n_channels, n_freqs), got {spectrum.shape} for {path}.")
        spectrum = spectrum[eeg_picks, :]
        per_recording_psd.append(spectrum)
        if frequency_axis is None:
            frequency_axis = freqs
        elif not np.allclose(frequency_axis, freqs):
            raise ValueError("All PSD inputs must share the same frequency axis.")

    if len(per_recording_psd) == 0 or frequency_axis is None or reference_info is None:
        raise ValueError("Expected at least one FIF file for PSD aggregation.")

    stacked = np.stack(per_recording_psd, axis=0)
    return PsdSummary(
        info=reference_info,
        frequencies_hz=frequency_axis,
        channel_power=np.nanmean(stacked, axis=0),
    )


def _load_component_count_summary(paths: Sequence[Path]) -> ComponentCountSummary:
    """Load included and excluded component counts from precomputed ICA files."""
    subject_ids: list[str] = []
    total_counts: list[int] = []
    included_counts: list[int] = []
    excluded_counts: list[int] = []
    bad_channel_counts: list[int] = []

    for path in paths:
        ica = mne.preprocessing.read_ica(path)
        subject_ids.append(str(path.name).split("_task-", 1)[0])
        excluded = int(len(set(getattr(ica, "exclude", []))))
        if hasattr(ica, "n_components_") and getattr(ica, "n_components_") is not None:
            total = int(ica.n_components_)
        else:
            components = np.asarray(ica.get_components())
            if components.ndim != 2:
                raise ValueError(f"Expected ICA component matrix with 2 dims, got {components.shape} for {path}.")
            total = int(components.shape[1])
        total_counts.append(total)
        included_counts.append(total - excluded)
        excluded_counts.append(excluded)
        bad_channel_counts.append(0)

    if len(included_counts) == 0:
        raise ValueError("Expected at least one ICA file for component-count aggregation.")

    return ComponentCountSummary(
        subject_ids=tuple(subject_ids),
        total_counts=np.asarray(total_counts, dtype=float),
        included_counts=np.asarray(included_counts, dtype=float),
        excluded_counts=np.asarray(excluded_counts, dtype=float),
        bad_channel_counts=np.asarray(bad_channel_counts, dtype=float),
    )


def _resolve_channels_tsv_path(paths: ProjectPaths, run_path: Path) -> Path:
    """Resolve the BIDS channels.tsv sidecar for a derived run-level FIF path."""
    match = RUN_STEM_PATTERN.match(run_path.name)
    if match is None:
        raise ValueError(f"Could not parse subject/task/run stem from {run_path}.")
    subject_id = str(match.group("subject"))
    task = str(match.group("task"))
    run_id = str(match.group("run"))
    return paths.raw_root / subject_id / "eeg" / f"{subject_id}_task-{task}_run-{run_id}_channels.tsv"


def _parse_run_stem(run_path: Path) -> tuple[str, str, str]:
    """Return subject, task, and run ids from a derived run-level filename."""
    match = RUN_STEM_PATTERN.match(run_path.name)
    if match is None:
        raise ValueError(f"Could not parse subject/task/run stem from {run_path}.")
    return (
        str(match.group("subject")),
        str(match.group("task")),
        str(match.group("run")),
    )


def _resolve_edf_path(paths: ProjectPaths, run_path: Path) -> Path:
    """Resolve the BIDS EDF path for a derived run-level file."""
    subject_id, task, run_id = _parse_run_stem(run_path)
    return paths.raw_root / subject_id / "eeg" / f"{subject_id}_task-{task}_run-{run_id}_eeg.edf"


def _resolve_downsampled_path(paths: ProjectPaths, run_path: Path) -> Path:
    """Resolve the canonical downsampled FIF path for one run."""
    subject_id, task, run_id = _parse_run_stem(run_path)
    return paths.out_dir / "eeg" / "downsampled" / f"{subject_id}_task-{task}_run-{run_id}_raw_ds.fif"


def _resolve_ica_path(cfg: ProjectConfig, run_path: Path) -> Path:
    """Resolve the precomputed ICA path for one run."""
    subject_id, task, run_id = _parse_run_stem(run_path)
    paths_cfg = cfg.raw.get("paths", {})
    if not isinstance(paths_cfg, dict) or "precomputed_ica_root" not in paths_cfg:
        raise ValueError("Config paths must define 'precomputed_ica_root' for speech artefact reconstruction.")
    preprocessing_cfg = cfg.raw.get("preprocessing", {})
    ica_cfg = preprocessing_cfg.get("ica", {}) if isinstance(preprocessing_cfg, dict) else {}
    pattern = str((ica_cfg if isinstance(ica_cfg, dict) else {}).get("path_pattern", "{subject_id}_task-{task}-ica.fif"))
    return Path(str(paths_cfg["precomputed_ica_root"])) / pattern.format(
        subject_id=subject_id,
        subject=subject_id,
        task=task,
        run=run_id,
    )


def _ensure_downsampled_input(run_path: Path, *, cfg: ProjectConfig, scratch_root: Path) -> Path:
    """Return an on-disk downsampled FIF, rebuilding only when needed."""
    project_paths = ProjectPaths.from_config(cfg)
    canonical_path = _resolve_downsampled_path(project_paths, run_path)
    if canonical_path.exists():
        return canonical_path
    rebuilt_path = scratch_root / "downsampled" / canonical_path.name
    if rebuilt_path.exists():
        return rebuilt_path
    downsample_cfg = cfg.raw.get("preprocessing", {}).get("downsample", {})
    target_sfreq_hz = float((downsample_cfg if isinstance(downsample_cfg, dict) else {}).get("sfreq_hz", 64.0))
    downsample_edf_to_fif(
        input_edf_path=_resolve_edf_path(project_paths, run_path),
        channels_tsv_path=_resolve_channels_tsv_path(project_paths, run_path),
        output_fif_path=rebuilt_path,
        config=cfg,
        target_sfreq_hz=target_sfreq_hz,
        preload=False,
    )
    return rebuilt_path


def _materialize_prebandpass_input(
    run_path: Path,
    *,
    cfg: ProjectConfig,
    scratch_root: Path,
    apply_ica: bool,
) -> Path:
    """Return a pre-bandpass FIF path, rebuilding the minimal pipeline when missing."""
    if run_path.exists():
        return run_path

    project_paths = ProjectPaths.from_config(cfg)
    channels_tsv_path = _resolve_channels_tsv_path(project_paths, run_path)
    downsampled_path = _ensure_downsampled_input(run_path, cfg=cfg, scratch_root=scratch_root)

    stem = run_path.stem
    reref_path = scratch_root / "reref" / f"{stem.replace('_raw_interp_noica', '_raw_reref').replace('_raw_interp', '_raw_reref')}.fif"
    if not reref_path.exists():
        rereference_fif_to_fif(
            input_fif_path=downsampled_path,
            channels_tsv_path=channels_tsv_path,
            output_fif_path=reref_path,
            config=cfg,
        )

    source_path = reref_path
    if apply_ica:
        ica_applied_path = scratch_root / "ica_applied" / f"{stem.replace('_raw_interp', '_raw_ica')}.fif"
        if not ica_applied_path.exists():
            apply_ica_fif_to_fif(
                input_fif_path=reref_path,
                ica_path=_resolve_ica_path(cfg, run_path),
                output_fif_path=ica_applied_path,
                config=cfg,
            )
        source_path = ica_applied_path

    rebuilt_path = scratch_root / ("interpolated" if apply_ica else "interpolated_noica") / run_path.name
    if rebuilt_path.exists():
        return rebuilt_path
    interpolation_cfg = cfg.raw.get("preprocessing", {}).get("interpolation", {})
    interpolation_method = str((interpolation_cfg if isinstance(interpolation_cfg, dict) else {}).get("method", "spline"))
    interpolate_bads_fif_to_fif(
        input_fif_path=source_path,
        channels_tsv_path=channels_tsv_path,
        output_fif_path=rebuilt_path,
        config=cfg,
        method=interpolation_method,
    )
    return rebuilt_path


def _materialize_prebandpass_inputs(
    run_paths: Sequence[Path],
    *,
    cfg: ProjectConfig,
    scratch_root: Path,
    apply_ica: bool,
) -> list[Path]:
    """Return pre-bandpass inputs, rebuilding only the runs that are missing."""
    return [
        _materialize_prebandpass_input(
            run_path,
            cfg=cfg,
            scratch_root=scratch_root,
            apply_ica=apply_ica,
        )
        for run_path in run_paths
    ]


def _add_bad_channel_counts(
    summary: ComponentCountSummary,
    *,
    cfg: ProjectConfig,
    run_paths: Sequence[Path],
) -> ComponentCountSummary:
    """Attach per-subject bad-channel counts inferred from BIDS channels.tsv files."""
    project_paths = ProjectPaths.from_config(cfg)
    bad_counts_by_subject: dict[str, float] = {}

    for run_path in run_paths:
        channels_tsv_path = _resolve_channels_tsv_path(project_paths, run_path)
        subject_id = str(run_path.name).split("_task-", 1)[0]
        run_bad_count = float(_count_bad_channels(channels_tsv_path))
        existing = bad_counts_by_subject.get(subject_id)
        if existing is None or run_bad_count > existing:
            bad_counts_by_subject[subject_id] = run_bad_count

    inferred_bad_counts = []
    for subject_id, total in zip(summary.subject_ids, summary.total_counts, strict=True):
        missing_from_ica = max(0.0, EXPECTED_EEG_CHANNELS - float(total))
        reported_bad_count = bad_counts_by_subject.get(subject_id, missing_from_ica)
        inferred_bad_counts.append(missing_from_ica if missing_from_ica > 0.0 else max(0.0, reported_bad_count))
    inferred_bad_counts = np.asarray(inferred_bad_counts, dtype=float)
    adjusted_total_counts = summary.included_counts + summary.excluded_counts + inferred_bad_counts
    return ComponentCountSummary(
        subject_ids=summary.subject_ids,
        total_counts=adjusted_total_counts,
        included_counts=summary.included_counts,
        excluded_counts=summary.excluded_counts,
        bad_channel_counts=inferred_bad_counts,
    )


def _sort_component_summary(summary: ComponentCountSummary) -> ComponentCountSummary:
    """Sort subjects by ascending number of included components."""
    order = np.argsort(summary.included_counts, kind="stable")
    subject_ids = tuple(summary.subject_ids[index] for index in order)
    return ComponentCountSummary(
        subject_ids=subject_ids,
        total_counts=summary.total_counts[order],
        included_counts=summary.included_counts[order],
        excluded_counts=summary.excluded_counts[order],
        bad_channel_counts=summary.bad_channel_counts[order],
    )


def _plot_psd(axis: plt.Axes, summary: PsdSummary, *, title: str) -> None:
    """Render one PSD panel."""
    spectrum = mne.time_frequency.SpectrumArray(summary.channel_power, summary.info, summary.frequencies_hz)
    spectrum.plot(axes=axis, show=False, average=False)
    for line in axis.lines:
        line.set_linewidth(1.8)
    axis.set_title(title)


def _plot_component_counts(axis: plt.Axes, summary: ComponentCountSummary) -> None:
    """Render one stacked included/excluded component-count bar plot."""
    positions = np.arange(len(summary.subject_ids), dtype=float)
    included_color = "#4c956c"
    excluded_color = "#c1121f"
    axis.bar(
        positions,
        summary.included_counts,
        width=0.8,
        color=included_color,
        label="Included",
    )
    axis.bar(
        positions,
        summary.excluded_counts,
        width=0.8,
        bottom=summary.included_counts,
        color=excluded_color,
        label="Excluded",
    )
    axis.bar(
        positions,
        summary.bad_channel_counts,
        width=0.8,
        bottom=summary.included_counts + summary.excluded_counts,
        color="#9e9e9e",
        label="Bad channels",
    )
    axis.set_xticks(positions, summary.subject_ids, rotation=75, ha="right")
    axis.set_xlabel("Subject")
    axis.set_ylabel("Number of components")
    axis.set_title("ICA component counts")
    axis.set_ylim(0.0, 68.0)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)


def _scale_figure_fonts(figure: plt.Figure, *, scale: float) -> None:
    """Scale all text artists in a rendered figure by a constant factor."""
    if not hasattr(figure, "findobj"):
        return
    for text in figure.findobj(matplotlib.text.Text):
        fontsize = text.get_fontsize()
        if fontsize is not None:
            text.set_fontsize(float(fontsize) * scale)


def _tune_speech_artefact_fonts(axes: Sequence[plt.Axes]) -> None:
    """Apply panel-specific font sizing adjustments after global scaling."""
    for axis in axes:
        if not hasattr(axis, "yaxis") or not hasattr(axis.yaxis, "label"):
            continue
        axis.yaxis.label.set_fontsize(axis.yaxis.label.get_fontsize() * Y_AXIS_LABEL_SCALE)

    third_axis = axes[2]
    if hasattr(third_axis, "get_xticklabels"):
        for tick in third_axis.get_xticklabels():
            tick.set_fontsize(tick.get_fontsize() * THIRD_PANEL_XTICK_SCALE)

    legend = third_axis.get_legend() if hasattr(third_axis, "get_legend") else None
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(text.get_fontsize() * LEGEND_FONT_SCALE)


def _remove_second_panel_y_axis(axis: plt.Axes) -> None:
    """Hide y-axis ticks and labels for the second PSD panel."""
    axis.set_ylabel("")
    axis.set_yticks([])


def _reduce_first_second_gap(axes: Sequence[plt.Axes], *, delta: float) -> None:
    """Move the second panel slightly left to tighten the first-second gap."""
    if len(axes) < 2:
        return
    first_axis, second_axis = axes[0], axes[1]
    if not hasattr(first_axis, "get_position") or not hasattr(second_axis, "get_position"):
        return
    first_pos = first_axis.get_position()
    second_pos = second_axis.get_position()
    current_gap = second_pos.x0 - first_pos.x1
    shift = min(delta, max(0.0, current_gap * 0.8))
    if shift <= 0.0:
        return
    second_axis.set_position([second_pos.x0 - shift, second_pos.y0, second_pos.width, second_pos.height])


def _reduce_second_third_gap(axes: Sequence[plt.Axes], *, delta: float) -> None:
    """Move the third panel slightly left to tighten the second-third gap only."""
    if len(axes) < 3:
        return
    second_axis, third_axis = axes[1], axes[2]
    if not hasattr(second_axis, "get_position") or not hasattr(third_axis, "get_position"):
        return
    second_pos = second_axis.get_position()
    third_pos = third_axis.get_position()
    current_gap = third_pos.x0 - second_pos.x1
    shift = min(delta, max(0.0, current_gap * 0.8))
    if shift <= 0.0:
        return
    third_axis.set_position([third_pos.x0 - shift, third_pos.y0, third_pos.width, third_pos.height])


def build_speech_artefact_summary_figure(
    *,
    cfg: ProjectConfig,
    filtered_noica_paths: Sequence[Path],
    filtered_paths: Sequence[Path],
    ica_paths: Sequence[Path],
    output_path: Path,
) -> Path:
    """Build the speech artefact summary figure and write it to disk."""
    method, fmin_hz, fmax_hz, n_fft = _psd_settings(cfg)
    with tempfile.TemporaryDirectory(prefix="speech-artefact-qc-") as temp_dir:
        scratch_root = Path(temp_dir)
        noica_psd_paths = _materialize_prebandpass_inputs(
            filtered_noica_paths,
            cfg=cfg,
            scratch_root=scratch_root,
            apply_ica=False,
        )
        filtered_psd_paths = _materialize_prebandpass_inputs(
            filtered_paths,
            cfg=cfg,
            scratch_root=scratch_root,
            apply_ica=True,
        )
        noica_summary = _load_average_psd(
            noica_psd_paths,
            method=method,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            n_fft=n_fft,
        )
        filtered_summary = _load_average_psd(
            filtered_psd_paths,
            method=method,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            n_fft=n_fft,
        )
    component_summary = _load_component_count_summary(ica_paths)
    component_summary = _add_bad_channel_counts(component_summary, cfg=cfg, run_paths=filtered_noica_paths)
    component_summary = _sort_component_summary(component_summary)

    figure, axes = plt.subplots(1, 3, figsize=_figure_size(cfg))
    _plot_psd(axes[0], noica_summary, title="PSD: no ICA")
    _plot_psd(axes[1], filtered_summary, title="PSD: with ICA")
    _remove_second_panel_y_axis(axes[1])
    _plot_component_counts(axes[2], component_summary)
    _scale_figure_fonts(figure, scale=FONT_SCALE)
    _tune_speech_artefact_fonts(axes)
    figure.tight_layout()
    _reduce_first_second_gap(axes, delta=FIRST_SECOND_GAP_REDUCTION)
    _reduce_second_third_gap(axes, delta=SECOND_THIRD_GAP_REDUCTION)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=_figure_dpi(cfg), bbox_inches="tight")
    plt.close(figure)
    return output_path
