# ==================================================================================================
#                               Downsampling
# ==================================================================================================
#
# Core preprocessing logic for creating a smaller raw file early in the pipeline.
#
# > Downsampling.
# > Continuous EEG data were downsampled early in the preprocessing pipeline to reduce memory
# > usage and computational cost in subsequent analyses. Raw EDF files were converted to MNE FIF
# > format and resampled to a target sampling frequency only when it differed from the original
# > acquisition rate. When the original and target sampling frequencies matched, resampling was
# > skipped but the data were still saved in FIF format to ensure a uniform representation across
# > subjects. Downsampling was performed on continuous data prior to filtering, ICA, or epoching,
# > following standard MNE practices.
#
# This module intentionally contains the "real work" (MNE I/O + resampling).
# CLI wrappers and Snakemake rules should call into this module rather than
# duplicating the logic.


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import mne
import pandas as pd

from hyper.config import ProjectConfig


# ==================================================================================================
# Constants
# ==================================================================================================

BIDS_CHANNELS_SEP: str = "\t"

# Canonical BIDS channels.tsv columns (some datasets omit some fields)
COL_NAME: str = "name"
COL_TYPE: str = "type"
COL_STATUS: str = "status"
STATUS_BAD: str = "bad"

# Map BIDS channel "type" strings to MNE channel types
# (MNE expects lower-case channel type names)
BIDS_TYPE_TO_MNE: Dict[str, str] = {
    "EEG": "eeg",
    "EOG": "eog",
    "ECG": "ecg",
    "EMG": "emg",
    "STIM": "stim",
    "MISC": "misc",
    "SEEG": "seeg",
    "DBS": "dbs",
}

CONVERSATION_DURATION_S: int = 4 * 60
# Backward-compatibility alias; prefer CONVERSATION_DURATION_S in new code.
CONVERSATION_DURATION: int = CONVERSATION_DURATION_S

# ==================================================================================================
# Types
# ==================================================================================================

@dataclass(frozen=True, slots=True)
class ChannelsInfo:
    """
    Parsed channels.tsv information.

    Parameters
    ----------
    bads
        List of channel names marked as bad in channels.tsv.
    channel_types
        Mapping from channel name to MNE channel type (e.g., "eeg", "eog").

    Usage example
    -------------
        info = _read_channels_tsv(Path("sub-01_channels.tsv"))
        print(info.bads)
    """

    bads: Tuple[str, ...]
    channel_types: Dict[str, str]


# ==================================================================================================
# Helpers
# ==================================================================================================


def _read_channels_tsv(channels_tsv_path: Path) -> ChannelsInfo:
    """
    Read BIDS channels.tsv and extract bad channels and channel types.

    Parameters
    ----------
    channels_tsv_path
        Path to a BIDS channels.tsv file.

    Returns
    -------
    ChannelsInfo
        Parsed bad channel list and per-channel types.

    Notes
    -----
    This function is robust to partial channels.tsv files. If required columns
    are missing, it returns empty information for those parts.

    Usage example
    -------------
        info = _read_channels_tsv(Path("sub-01_channels.tsv"))
        print(info.channel_types.get("HEOG"))
    """
    table = pd.read_csv(channels_tsv_path, sep=BIDS_CHANNELS_SEP)

    bads: list[str] = []
    channel_types: Dict[str, str] = {}

    if COL_NAME in table.columns:
        names = table[COL_NAME].astype(str).tolist()
    else:
        # Without "name" we cannot reliably apply anything.
        return ChannelsInfo(bads=tuple(), channel_types={})

    if COL_STATUS in table.columns:
        status = table[COL_STATUS].astype(str).str.lower().tolist()
        for name, st in zip(names, status):
            if st == STATUS_BAD:
                bads.append(name)

    if COL_TYPE in table.columns:
        types = table[COL_TYPE].astype(str).tolist()
        for name, t in zip(names, types):
            mne_type = BIDS_TYPE_TO_MNE.get(t.strip().upper(), "")
            if mne_type:
                channel_types[name] = mne_type

    return ChannelsInfo(bads=tuple(bads), channel_types=channel_types)


def _get_montage_name(cfg: ProjectConfig) -> Optional[str]:
    """
    Retrieve montage name from config if present.

    Parameters
    ----------
    cfg
        Project configuration.

    Returns
    -------
    str | None
        Montage name such as "biosemi64", or None if not set.

    Usage example
    -------------
        name = _get_montage_name(cfg)
        if name is not None:
            print(name)
    """
    eeg_cfg = cfg.raw.get("eeg", {})
    montage = eeg_cfg.get("montage", None)
    if montage is None:
        return None
    return str(montage)


def _should_resample(original_sfreq_hz: float, target_sfreq_hz: float) -> bool:
    """
    Decide whether resampling should be performed.

    Parameters
    ----------
    original_sfreq_hz
        Sampling frequency in the input raw file.
    target_sfreq_hz
        Desired sampling frequency.

    Returns
    -------
    bool
        True if resampling should be done.

    Usage example
    -------------
        if _should_resample(2048.0, 512.0):
            print("Resample needed")
    """
    return abs(float(original_sfreq_hz) - float(target_sfreq_hz)) > 1e-6


# ==================================================================================================
# Core logic
# ==================================================================================================

def _load_raw_edf(input_edf_path: Path, *, preload: bool) -> mne.io.BaseRaw:
    """Read the source EDF recording."""
    return mne.io.read_raw_edf(input_edf_path, preload=preload, verbose="ERROR")


def _crop_to_conversation(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Crop raw data to the fixed conversation window."""
    events = mne.find_events(raw)
    conversation_start = events[0, 0] / raw.info["sfreq"]
    return raw.crop(conversation_start, conversation_start + CONVERSATION_DURATION_S)


def _apply_montage_from_config(raw: mne.io.BaseRaw, config: ProjectConfig) -> None:
    """Set standard montage when configured."""
    montage_name = _get_montage_name(config)
    if montage_name is not None:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, on_missing="warn")


def _apply_channel_types(raw: mne.io.BaseRaw, channels_info: ChannelsInfo) -> None:
    """Set channel types for channels present in raw."""
    if channels_info.channel_types:
        existing_names = set(raw.ch_names)
        to_set = {k: v for k, v in channels_info.channel_types.items() if k in existing_names}
        if to_set:
            raw.set_channel_types(to_set)


def _apply_bad_channels(raw: mne.io.BaseRaw, channels_info: ChannelsInfo) -> None:
    """Mark channels as bad based on channels.tsv."""
    if channels_info.bads:
        existing_names = set(raw.ch_names)
        raw.info["bads"] = [ch for ch in channels_info.bads if ch in existing_names]


def _resample_if_needed(raw: mne.io.BaseRaw, *, target_sfreq_hz: float) -> None:
    """Resample only when requested target differs from source frequency."""
    original_sfreq_hz = float(raw.info["sfreq"])
    if _should_resample(original_sfreq_hz, float(target_sfreq_hz)):
        raw.resample(float(target_sfreq_hz), npad="auto")


def _save_downsampled_raw(raw: mne.io.BaseRaw, output_fif_path: Path) -> None:
    """Write preprocessed raw to FIF."""
    output_fif_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_fif_path, overwrite=True)


def downsample_edf_to_fif(
    *,
    input_edf_path: Path,
    channels_tsv_path: Path,
    output_fif_path: Path,
    config: ProjectConfig,
    target_sfreq_hz: float,
    preload: bool = False,
) -> None:
    """
    Read raw EDF + channels.tsv, apply metadata, optionally resample, and save FIF.

    Parameters
    ----------
    input_edf_path
        Path to input EDF file (BIDS EEG).
    channels_tsv_path
        Path to channels.tsv with channel status/type information.
    output_fif_path
        Path to output FIF file.
    config
        Parsed project configuration.
    target_sfreq_hz
        Target sampling frequency in Hz. If equal to the original sampling
        frequency, resampling is skipped (still writes FIF).
    preload
        Whether to preload data into memory before resampling. For large EDFs,
        leaving this False is often safer; MNE will load as needed.

    Returns
    -------
    None

    Notes
    -----
    - Applies montage if `config.raw["eeg"]["montage"]` is set.
    - Marks bad channels based on `channels.tsv` status == "bad" (case-insensitive).
    - Sets channel types when the `type` column is present (EEG/EOG/ECG/...).

    Usage example
    -------------
        cfg = load_project_config(Path("config/config.yaml"))
        downsample_edf_to_fif(
            input_edf_path=Path("sub-01_eeg.edf"),
            channels_tsv_path=Path("sub-01_channels.tsv"),
            output_fif_path=Path("derived/raw_ds.fif"),
            config=cfg,
            target_sfreq_hz=512.0,
        )
    """
    channels_info = _read_channels_tsv(channels_tsv_path)
    raw = _load_raw_edf(input_edf_path, preload=preload)
    raw = _crop_to_conversation(raw)
    _apply_montage_from_config(raw, config)
    _apply_channel_types(raw, channels_info)
    _apply_bad_channels(raw, channels_info)
    _resample_if_needed(raw, target_sfreq_hz=target_sfreq_hz)
    _save_downsampled_raw(raw, output_fif_path)
