# ==================================================================================================
#                         Core: rereferencing (library)
# ==================================================================================================
# > EEG rereferencing and montage application.
# > Continuous EEG data were rereferenced using an a priori reference scheme (average reference
# > by default). Prior to rereferencing, non-EEG auxiliary channels (e.g., EMG/ear/status channels)
# > were dropped when present, and channels marked as bad in the BIDS channels.tsv file were
# > propagated to the recording’s bad-channel list. A standard electrode montage (BioSemi-64 by
# > default) was then applied to ensure consistent channel locations. The rereferenced continuous
# >data were saved in FIF format for downstream preprocessing.

from pathlib import Path
from typing import Iterable, Optional

import mne
import pandas as pd

# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_REFERENCE: str = "average"
DEFAULT_MONTAGE: str = "biosemi64"
DEFAULT_ON_MISSING: str = "warn"
DEFAULT_PRELOAD: bool = False

DEFAULT_DROP_CHANNELS: tuple[str, ...] = (
    "EMG1",
    "EMG2",
    "lEAR",
    "rEAR",
    "Status",
)


# ==================================================================================================
# Helpers
# ==================================================================================================

def load_channels_tsv(channels_tsv_path: Path) -> pd.DataFrame:
    """
    Load a BIDS channels.tsv into a DataFrame.

    Parameters
    ----------
    channels_tsv_path
        Path to BIDS channels.tsv.

    Returns
    -------
    pandas.DataFrame
        Channels metadata.

    DataFrame format example
    ------------------------
    | name | type | status | status_description |
    |------|------|--------|--------------------|
    | Fp1  | EEG  | good   |                    |
    | Fp2  | EEG  | bad    | noisy              |

    Usage example
    -------------
        df = load_channels_tsv(Path("sub-01_channels.tsv"))
        bads = df.loc[df["status"] == "bad", "name"].tolist()
    """
    channels_df = pd.read_csv(channels_tsv_path, sep="\t")

    if "name" not in channels_df.columns or "status" not in channels_df.columns:
        raise ValueError(
            "channels.tsv must contain at least columns: 'name' and 'status'. "
            f"Got columns: {list(channels_df.columns)}"
        )

    return channels_df


def _load_data_inplace(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Ensure raw samples are available for in-place operations."""
    return raw.load_data()


def _drop_optional_channels(raw: mne.io.BaseRaw, drop_channels: Iterable[str]) -> None:
    """Drop configured auxiliary channels if present."""
    try:
        raw.drop_channels([ch for ch in drop_channels if ch in raw.ch_names])
    except Exception:  # noqa
        pass


def _set_bad_channels_from_table(raw: mne.io.BaseRaw, channels_df: pd.DataFrame) -> None:
    """Populate bad channels from channels.tsv."""
    bads = channels_df.loc[channels_df["status"] == "bad", "name"].astype(str).tolist()
    raw.info["bads"] = bads


def _apply_eeg_reference(raw: mne.io.BaseRaw, reference: str) -> None:
    """Apply configured EEG reference."""
    if reference == "average":
        raw.set_eeg_reference(ref_channels="average")
    else:
        raw.set_eeg_reference(ref_channels=reference)


def _apply_standard_montage(raw: mne.io.BaseRaw, *, montage_name: str, on_missing: str) -> None:
    """Set channel locations from a named standard montage."""
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing=on_missing)


def _load_raw_fif(input_fif_path: Path, *, preload: bool) -> mne.io.BaseRaw:
    """Read raw FIF for rereferencing."""
    return mne.io.read_raw_fif(input_fif_path, preload=preload, verbose="ERROR")


def _save_rereferenced_raw(raw: mne.io.BaseRaw, output_fif_path: Path) -> None:
    """Write rereferenced FIF output."""
    output_fif_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_fif_path, overwrite=True)


def rereference_raw(
    raw: mne.io.BaseRaw,
    channels_df: pd.DataFrame,
    *,
    reference: str = DEFAULT_REFERENCE,
    montage_name: str = DEFAULT_MONTAGE,
    on_missing: str = DEFAULT_ON_MISSING,
    drop_channels: Iterable[str] = DEFAULT_DROP_CHANNELS,
) -> mne.io.BaseRaw:
    """
    Apply EEG rereferencing and montage, using channels.tsv to set bad channels.

    This implements your legacy logic:
    - drop EMG / ear / stim channels if present
    - set `raw.info["bads"]` from channels.tsv status == "bad"
    - apply average EEG reference
    - set BioSemi-64 montage

    Parameters
    ----------
    raw
        Raw object to process.
    channels_df
        DataFrame created from BIDS channels.tsv.

        DataFrame format example
        ------------------------
        | name | status |
        |------|--------|
        | Fp1  | good   |
        | Fp2  | bad    |
    reference
        EEG reference to apply. Default is "average".
    montage_name
        Montage name passed to `mne.channels.make_standard_montage`.
    on_missing
        How to handle missing montage channels when calling `raw.set_montage`.
    drop_channels
        Channel names to drop if present.

    Returns
    -------
    mne.io.BaseRaw
        The same Raw instance, modified in-place and returned for convenience.

    Usage example
    -------------
        import mne
        import pandas as pd
        from pathlib import Path

        raw = mne.io.read_raw_fif("raw_ds.fif", preload=False)
        channels_df = pd.read_csv(Path("sub-01_channels.tsv"), sep="\\t")

        raw = rereference_raw(raw, channels_df)
        raw.save("raw_reref.fif", overwrite=True)
    """

    raw = _load_data_inplace(raw)
    _drop_optional_channels(raw, drop_channels)
    _set_bad_channels_from_table(raw, channels_df)
    _apply_eeg_reference(raw, reference)
    _apply_standard_montage(raw, montage_name=montage_name, on_missing=on_missing)

    return raw


# ==================================================================================================
# Core logic
# ==================================================================================================

def rereference_fif_to_fif(
    *,
    input_fif_path: Path,
    channels_tsv_path: Path,
    output_fif_path: Path,
    config: object,
    preload: bool = DEFAULT_PRELOAD,
    reference: str = DEFAULT_REFERENCE,
    montage_name: Optional[str] = None,
    on_missing: str = DEFAULT_ON_MISSING,
) -> None:
    """
    Load a FIF file, rereference EEG, apply montage, and save a new FIF file.

    Parameters
    ----------
    input_fif_path
        Input raw FIF (e.g., output of downsampling: raw_ds.fif).
    channels_tsv_path
        BIDS channels.tsv used to set bad channels.
    output_fif_path
        Output raw FIF (e.g., raw_reref.fif).
    config
        ProjectConfig-like object (passed for pipeline consistency).
        Currently not required, but kept to match the repo contract.
    preload
        Whether to preload data when reading FIF.
    reference
        EEG reference to apply (default "average").
    montage_name
        Montage name; if None, uses DEFAULT_MONTAGE.
        (You can later wire this to config, e.g., cfg.raw["eeg"]["montage"].)
    on_missing
        Passed to `raw.set_montage`.

    Returns
    -------
    None

    Usage example
    -------------
        from pathlib import Path
        from conv.config import load_project_config
        from conv.preprocessing.reref import rereference_fif_to_fif

        cfg = load_project_config(Path("config/config.yaml"))

        rereference_fif_to_fif(
            input_fif_path=Path("derived/raw_ds.fif"),
            channels_tsv_path=Path("bids/sub-01/channels.tsv"),
            output_fif_path=Path("derived/raw_reref.fif"),
            config=cfg,
            preload=False,
        )
    """
    _ = config  # reserved for future use (keeps stable signature across pipeline)

    channels_df = load_channels_tsv(channels_tsv_path)

    raw = _load_raw_fif(input_fif_path, preload=preload)
    rereference_raw(
        raw,
        channels_df,
        reference=reference,
        montage_name=montage_name or DEFAULT_MONTAGE,
        on_missing=on_missing,
    )
    _save_rereferenced_raw(raw, output_fif_path)
