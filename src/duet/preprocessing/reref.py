# ==================================================================================================
#                         Core: rereferencing (library)
# ==================================================================================================
# > EEG rereferencing.
# > Continuous EEG data were rereferenced using an a priori reference scheme (average reference
# > by default) after downsampling, filtering, ICA application, and bad-channel interpolation.
# > Channel typing, bad-channel propagation, and montage application happen earlier in the
# > pipeline, so this step now performs the final rereference only and saves the fully
# > preprocessed continuous signal for downstream analyses.

from pathlib import Path
from typing import Optional

import mne
import pandas as pd

# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_REFERENCE: str = "average"
DEFAULT_PRELOAD: bool = False


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


def _apply_eeg_reference(raw: mne.io.BaseRaw, reference: str) -> None:
    """Apply configured EEG reference."""
    if reference == "average":
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    else:
        raw.set_eeg_reference(ref_channels=reference, projection=False, verbose="ERROR")


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
) -> mne.io.BaseRaw:
    """
    Apply the final EEG rereference.

    The `channels_df` argument is retained for backward compatibility with the
    existing CLI and Snakemake interfaces, but the table is no longer used to
    mutate channel state here because earlier steps already applied BIDS
    channel types, bad-channel annotations, and montage information.

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

    _ = channels_df
    raw = _load_data_inplace(raw)
    _apply_eeg_reference(raw, reference)
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
        Retained for backward compatibility with older call sites. Ignored.

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
    )
    _save_rereferenced_raw(raw, output_fif_path)
