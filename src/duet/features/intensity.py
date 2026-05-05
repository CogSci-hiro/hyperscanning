"""Praat-backed acoustic intensity extraction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_praat_intensity(
    audio_path: Path,
    minimum_pitch: float,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a Praat intensity contour from a waveform on disk."""
    import parselmouth

    snd = parselmouth.Sound(str(audio_path))

    intensity_obj = snd.to_intensity(
        minimum_pitch=minimum_pitch,
        time_step=time_step,
    )

    times = intensity_obj.xs()
    values = intensity_obj.values[0]

    return times, values
