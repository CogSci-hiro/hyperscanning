from pathlib import Path
from typing import Tuple

import numpy as np
import parselmouth


def extract_praat_intensity(
    audio_path: Path,
    minimum_pitch: float,
    time_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Praat intensity time series using parselmouth.

    Parameters
    ----------
    audio_path : Path
        Path to WAV file.
    minimum_pitch : float
        Minimum pitch parameter controlling intensity window length.
    time_step : float
        Time step in seconds (e.g., 0.01 for 100 Hz).

    Returns
    -------
    times : np.ndarray
        Time axis (seconds).
    intensity : np.ndarray
        Intensity in dB.
    """
    snd = parselmouth.Sound(str(audio_path))

    intensity_obj = snd.to_intensity(
        minimum_pitch=minimum_pitch,
        time_step=time_step,
    )

    times = intensity_obj.xs()
    values = intensity_obj.values[0]

    return times, values