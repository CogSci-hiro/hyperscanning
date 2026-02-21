# ==================================================================================================
#                            Core: epoching (library)
# ==================================================================================================
#
# > Epoching.
# > Continuous preprocessed EEG data were segmented into epochs time-locked to event onsets
# > using externally defined event arrays. Epochs were extracted over a fixed time window
# > relative to each event, with optional baseline correction and linear detrending applied
# > as specified. Artifact rejection was performed using a fixed peak-to-peak EEG amplitude
# > threshold. Trial-level metadata were attached to the resulting epochs,
# > and all epochs were saved in MNE FIF format for downstream analysis.

from pathlib import Path
from typing import Dict, Optional, Tuple

import mne
import numpy as np
import pandas as pd


# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_PRELOAD_RAW: bool = False
DEFAULT_REJECT_BY_ANNOTATION: bool = False
REJECT_EPOCH: Dict[str, float] = {"eeg": 100e-6}  # Above 100 µV
EVENTS_ID: Dict[str, int] = {"onset": 1}

# ==================================================================================================
# Core logic
# ==================================================================================================

def _load_epoch_inputs(
    *,
    raw_fif_path: Path,
    events_npy_path: Path,
    metadata_tsv_path: Path,
    preload_raw: bool,
) -> tuple[mne.io.BaseRaw, pd.DataFrame, np.ndarray]:
    """Load raw data, metadata table, and events array."""
    raw = mne.io.read_raw_fif(raw_fif_path, preload=preload_raw, verbose="ERROR")
    metadata = pd.read_csv(metadata_tsv_path, sep="\t")
    events = np.load(events_npy_path)
    return raw, metadata, events


def _validate_events_shape(events: np.ndarray) -> None:
    """Ensure events follow MNE's (n_events, 3) shape contract."""
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError(f"events must have shape (n_events, 3), got: {events.shape}")


def _build_epochs(
    *,
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    metadata: pd.DataFrame,
    tmin_s: float,
    tmax_s: float,
    baseline: Optional[Tuple[float, float]],
    detrend: Optional[int],
    reject_by_annotation: bool,
) -> mne.Epochs:
    """Create an MNE Epochs object from validated inputs."""
    return mne.Epochs(
        raw,
        events=events,
        event_id=EVENTS_ID,
        tmin=float(tmin_s),
        tmax=float(tmax_s),
        baseline=baseline,
        reject=REJECT_EPOCH,
        detrend=detrend,
        reject_by_annotation=bool(reject_by_annotation),
        metadata=metadata,
        preload=False,
        verbose="ERROR",
    )


def _save_epochs(epochs: mne.Epochs, output_epochs_path: Path) -> None:
    """Persist epochs to disk."""
    output_epochs_path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(output_epochs_path, overwrite=True)


def make_epochs_fif_to_fif(
    *,
    raw_fif_path: Path,
    events_npy_path: Path,
    metadata_tsv_path: Path,
    output_epochs_path: Path,
    config: object,
    tmin_s: float,
    tmax_s: float,
    baseline: Optional[Tuple[float, float]],
    detrend: Optional[int],
    preload_raw: bool = DEFAULT_PRELOAD_RAW,
    reject_by_annotation: bool = DEFAULT_REJECT_BY_ANNOTATION,
) -> None:
    """
    Create MNE Epochs from a preprocessed raw FIF, events array, and metadata table.

    Parameters
    ----------
    raw_fif_path
        Input raw FIF file.
    events_npy_path
        NumPy .npy file containing an MNE events array (n_events, 3).
    metadata_tsv_path
        TSV metadata file (will be attached to Epochs.metadata).
    output_epochs_path
        Output .fif file for epochs (typically ends with -epo.fif or epochs.fif).
    config
        ProjectConfig-like object (passed for pipeline consistency).
        Not currently used.
    tmin_s
        Epoch start relative to event (seconds).
    tmax_s
        Epoch end relative to event (seconds).
    baseline
        Baseline tuple (start, end) in seconds, or None for no baseline correction.
    detrend
        Detrend order passed to MNE (typically 0 or 1), or None to disable detrending.
    preload_raw
        Whether to preload raw when reading. Default False.
    reject_by_annotation
        Passed to MNE Epochs. Default False (matches your legacy script).

    Returns
    -------
    None

    DataFrame format example
    ------------------------
    | timestamp | self_rate | other_rate | latency |
    |----------:|----------:|-----------:|--------:|
    |     1.20  | 3.33      | 4.20       | 0.15    |

    Usage example
    -------------
        from pathlib import Path
        from hyper.config import load_project_config
        from hyper.preprocessing.epoching import make_epochs_fif_to_fif

        cfg = load_project_config(Path("config/config.yaml"))

        make_epochs_fif_to_fif(
            raw_fif_path=Path("derived/eeg/filtered/.../raw_filt.fif"),
            events_npy_path=Path("derived/beh/metadata/.../events.npy"),
            metadata_tsv_path=Path("derived/beh/metadata/.../metadata.tsv"),
            output_epochs_path=Path("derived/eeg/epochs/.../epochs-epo.fif"),
            config=cfg,
            tmin_s=-1.0,
            tmax_s=1.0,
            baseline=(-0.2, 0.0),
            detrend=None,
        )
    """
    _ = config  # reserved for future use

    raw, metadata, events = _load_epoch_inputs(
        raw_fif_path=raw_fif_path,
        events_npy_path=events_npy_path,
        metadata_tsv_path=metadata_tsv_path,
        preload_raw=preload_raw,
    )
    _validate_events_shape(events)
    epochs = _build_epochs(
        raw=raw,
        events=events,
        metadata=metadata,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        baseline=baseline,
        detrend=detrend,
        reject_by_annotation=reject_by_annotation,
    )
    _save_epochs(epochs, output_epochs_path)
