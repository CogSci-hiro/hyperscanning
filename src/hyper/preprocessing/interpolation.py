# ==================================================================================================
#                      Core: bad-channel interpolation (library)
# ==================================================================================================
# > Bad-channel interpolation.
# > EEG channels marked as bad in the BIDS channels.tsv file were interpolated using
# > MNE’s built-in spatial interpolation routines. Interpolation was applied to continuous data
# > using a predefined method (spline by default), after which the bad-channel list was cleared.
# > The interpolated continuous signal was saved in FIF format for subsequent preprocessing steps.

from pathlib import Path

import mne
import pandas as pd

# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_PRELOAD: bool = True
DEFAULT_VERBOSE: str = "ERROR"
DEFAULT_METHOD: str = "spline"


# ==================================================================================================
# Core logic
# ==================================================================================================

def _read_raw(
    input_fif_path: Path,
    *,
    preload: bool,
    verbose: str,
) -> mne.io.BaseRaw:
    """Load input raw FIF for bad-channel interpolation."""
    return mne.io.read_raw_fif(input_fif_path, preload=preload, verbose=verbose)


def _read_and_validate_channels(channels_tsv_path: Path) -> pd.DataFrame:
    """Read BIDS channels metadata and validate required columns."""
    channels_df = pd.read_csv(channels_tsv_path, sep="\t")
    if "name" not in channels_df.columns or "status" not in channels_df.columns:
        raise ValueError(
            "channels.tsv must contain at least columns: 'name' and 'status'. "
            f"Got columns: {list(channels_df.columns)}"
        )
    return channels_df


def _set_bads_from_channels(raw: mne.io.BaseRaw, channels_df: pd.DataFrame) -> None:
    """Populate `raw.info['bads']` from channels.tsv."""
    bads = channels_df.loc[channels_df["status"] == "bad", "name"].astype(str).tolist()
    raw.info["bads"] = bads


def _interpolate_if_needed(raw: mne.io.BaseRaw, *, method: str) -> None:
    """Interpolate only when there are marked bad channels."""
    if len(raw.info["bads"]) > 0:
        raw.interpolate_bads(reset_bads=True, method=method)


def _save_interpolated_raw(raw: mne.io.BaseRaw, output_fif_path: Path) -> None:
    """Write interpolated raw data."""
    output_fif_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_fif_path, overwrite=True)


def interpolate_bads_fif_to_fif(
    *,
    input_fif_path: Path,
    channels_tsv_path: Path,
    output_fif_path: Path,
    config: object,
    method: str = DEFAULT_METHOD,
    preload: bool = DEFAULT_PRELOAD,
    verbose: str = DEFAULT_VERBOSE,
) -> None:
    """
    Interpolate bad channels marked in a BIDS channels.tsv file and save output.

    Parameters
    ----------
    input_fif_path
        Input raw FIF file.
    channels_tsv_path
        BIDS channels.tsv file containing at least columns `name` and `status`.
    output_fif_path
        Output raw FIF file.
    config
        ProjectConfig-like object (passed for pipeline consistency).
        Not currently used.
    method
        Interpolation method passed to `raw.interpolate_bads`.
        Common choices include "spline" and "nearest".
    preload
        Whether to preload data when reading raw FIF. Default True.
    verbose
        MNE verbosity level passed to readers.

    Returns
    -------
    None

    DataFrame format example
    ------------------------
    | name | status |
    |------|--------|
    | Fp1  | good   |
    | Fp2  | bad    |

    Usage example
    -------------
        from pathlib import Path
        from hyper.config import load_project_config
        from hyper.preprocessing.interpolation import interpolate_bads_fif_to_fif

        cfg = load_project_config(Path("config/config.yaml"))

        interpolate_bads_fif_to_fif(
            input_fif_path=Path("derived/raw_ica.fif"),
            channels_tsv_path=Path("bids/sub-01/channels.tsv"),
            output_fif_path=Path("derived/raw_interp.fif"),
            config=cfg,
            method="spline",
        )
    """
    _ = config  # reserved for future use

    raw = _read_raw(input_fif_path, preload=preload, verbose=verbose)
    channels_df = _read_and_validate_channels(channels_tsv_path)
    _set_bads_from_channels(raw, channels_df)
    _interpolate_if_needed(raw, method=method)
    _save_interpolated_raw(raw, output_fif_path)
