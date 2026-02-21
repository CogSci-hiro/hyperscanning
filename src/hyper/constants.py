# ==================================================================================================
#                                   Constants
# ==================================================================================================
#
# Canonical EEG frequency band boundaries used by downstream analyses.
# Defining them once here prevents accidental drift between scripts.

from typing import Final, Tuple
# 4-7 Hz: often associated with cognitive control / memory processes.
THETA_BAND_HZ: Final[Tuple[float, float]] = (4.0, 7.0)
# 7-13 Hz: classic alpha rhythm range used in many resting-state analyses.
ALPHA_BAND_HZ: Final[Tuple[float, float]] = (7.0, 13.0)
# 13-30 Hz: beta activity, often linked to sensorimotor/cognitive state.
BETA_BAND_HZ: Final[Tuple[float, float]] = (13.0, 30.0)
