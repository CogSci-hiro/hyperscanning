# ==================================================================================================
#                        Core: band-pass filtering (library)
# ==================================================================================================
# > Band-pass filtering.
# > Continuous EEG data were band-pass filtered using MNE’s built-in filtering routines, with
# > fixed high-pass and low-pass cutoff frequencies specified a priori. Filtering was applied
# > to continuous data prior to epoching, and the resulting signals were saved in FIF format
# > for subsequent analyses.

from pathlib import Path

import mne


# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_PRELOAD: bool = True
DEFAULT_VERBOSE: str = "ERROR"

# ==================================================================================================
# Core logic
# ==================================================================================================

def _load_raw_for_filtering(
    input_fif_path: Path,
    *,
    preload: bool,
    verbose: str,
) -> mne.io.BaseRaw:
    """Read the input FIF for filtering."""
    return mne.io.read_raw_fif(input_fif_path, preload=preload, verbose=verbose)


def _apply_bandpass_filter(
    raw: mne.io.BaseRaw,
    *,
    l_freq_hz: float,
    h_freq_hz: float,
) -> None:
    """Apply the configured band-pass range in-place."""
    raw.filter(l_freq=l_freq_hz, h_freq=h_freq_hz)


def _save_filtered_raw(raw: mne.io.BaseRaw, output_fif_path: Path) -> None:
    """Persist filtered data to FIF."""
    output_fif_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_fif_path, overwrite=True)


def bandpass_filter_fif_to_fif(
    *,
    input_fif_path: Path,
    output_fif_path: Path,
    config: object,
    l_freq_hz: float,
    h_freq_hz: float,
    preload: bool = DEFAULT_PRELOAD,
    verbose: str = DEFAULT_VERBOSE,
) -> None:
    """
    Apply a band-pass filter to a raw FIF file and save output.

    Parameters
    ----------
    input_fif_path
        Input raw FIF file.
    output_fif_path
        Output raw FIF file.
    config
        ProjectConfig-like object (passed for pipeline consistency).
        Not currently used.
    l_freq_hz
        High-pass cutoff frequency in Hz.
    h_freq_hz
        Low-pass cutoff frequency in Hz.
    preload
        Whether to preload data when reading raw FIF. Default True.
    verbose
        MNE verbosity level passed to readers.

    Returns
    -------
    None

    Usage example
    -------------
        from pathlib import Path
        from hyper.config import load_project_config
        from hyper.preprocessing.filtering import bandpass_filter_fif_to_fif

        cfg = load_project_config(Path("config/config.yaml"))

        bandpass_filter_fif_to_fif(
            input_fif_path=Path("derived/raw_interp.fif"),
            output_fif_path=Path("derived/raw_filt.fif"),
            config=cfg,
            l_freq_hz=1.0,
            h_freq_hz=40.0,
        )
    """
    _ = config  # reserved for future use

    raw = _load_raw_for_filtering(input_fif_path, preload=preload, verbose=verbose)
    _apply_bandpass_filter(raw, l_freq_hz=l_freq_hz, h_freq_hz=h_freq_hz)
    _save_filtered_raw(raw, output_fif_path)
