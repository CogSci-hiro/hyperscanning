# ==================================================================================================
#                             Core: ICA application
# ==================================================================================================
# > ICA artifact correction.
# > Artifact correction was performed by applying a precomputed independent component analysis
# > (ICA) solution TODO to the continuous EEG data. ICA components selected for removal were
# > defined during a separate fitting and manual curation step, and the curated solution was
# > applied unchanged to the data. The corrected continuous signal was saved in FIF format
# > for downstream processing.

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

def _load_raw_for_ica(
    input_fif_path: Path,
    *,
    preload: bool,
    verbose: str,
) -> mne.io.BaseRaw:
    """Read the input raw file prior to ICA application."""
    return mne.io.read_raw_fif(input_fif_path, preload=preload, verbose=verbose)


def _load_ica(ica_path: Path) -> mne.preprocessing.ICA:
    """Load a precomputed ICA object."""
    return mne.preprocessing.read_ica(ica_path)


def _apply_ica(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> None:
    """Apply ICA exclusion mask in-place."""
    ica.apply(raw)


def _save_ica_output(raw: mne.io.BaseRaw, output_fif_path: Path) -> None:
    """Write ICA-corrected raw data."""
    output_fif_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_fif_path, overwrite=True)


def apply_ica_fif_to_fif(
    *,
    input_fif_path: Path,
    ica_path: Path,
    output_fif_path: Path,
    config: object,
    preload: bool = DEFAULT_PRELOAD,
    verbose: str = DEFAULT_VERBOSE,
) -> None:
    """
    Apply a precomputed ICA solution to a raw FIF file and save the result.

    Notes
    -----
    This function assumes ICA fitting + manual curation (component selection)
    has already been done elsewhere. The ICA object on disk should already
    encode `ica.exclude`.

    Parameters
    ----------
    input_fif_path
        Input raw FIF file (pre-ICA).
    ica_path
        Path to precomputed MNE ICA object (e.g., ica.fif).
    output_fif_path
        Output raw FIF file (post-ICA).
    config
        ProjectConfig-like object (passed for pipeline consistency).
        Not currently used.
    preload
        Whether to preload data when reading the raw FIF. Default True because
        `ICA.apply` modifies data.
    verbose
        MNE verbosity level passed to readers.

    Returns
    -------
    None

    Usage example
    -------------
        from pathlib import Path
        from hyper.config import load_project_config
        from hyper.preprocessing.ica import apply_ica_fif_to_fif

        cfg = load_project_config(Path("config/config.yaml"))

        apply_ica_fif_to_fif(
            input_fif_path=Path("derived/raw_reref.fif"),
            ica_path=Path("derived/ica/ica.fif"),
            output_fif_path=Path("derived/raw_ica.fif"),
            config=cfg,
            preload=True,
        )
    """
    _ = config  # reserved for future use

    raw = _load_raw_for_ica(input_fif_path, preload=preload, verbose=verbose)
    ica = _load_ica(ica_path)
    _apply_ica(raw, ica)
    _save_ica_output(raw, output_fif_path)
