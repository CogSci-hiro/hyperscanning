"""Shared pytest fixtures and lightweight test doubles for the hyper test suite.

These fixtures intentionally replace heavy MNE objects with tiny stand-ins so
unit tests stay fast, deterministic, and easy to understand.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

# Ensure `import hyper` resolves to the in-repo source tree during tests.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Keep MNE/Numba runtime state inside the workspace to avoid sandbox writes to
# user home directories (e.g., ~/.mne) during imports and test collection.
MNE_HOME = PROJECT_ROOT / ".mne_test_home"
MNE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MNE_HOME", str(MNE_HOME))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


@dataclass
class DummyRaw:
    """Small stand-in for MNE Raw used in preprocessing tests."""

    sfreq: float = 100.0
    ch_names: list[str] = field(default_factory=lambda: ["Fp1", "Fp2", "EMG1", "Status"])
    first_samp: int = 0

    def __post_init__(self) -> None:
        self.info: dict[str, Any] = {"sfreq": self.sfreq, "bads": []}
        self.calls: list[tuple[str, Any]] = []

    def crop(self, tmin: float, tmax: float):
        self.calls.append(("crop", (tmin, tmax)))
        return self

    def set_montage(self, montage: Any, on_missing: str = "warn") -> None:
        self.calls.append(("set_montage", (montage, on_missing)))

    def set_channel_types(self, mapping: dict[str, str]) -> None:
        self.calls.append(("set_channel_types", mapping))

    def resample(self, sfreq: float, npad: str = "auto") -> None:
        self.info["sfreq"] = sfreq
        self.calls.append(("resample", (sfreq, npad)))

    def save(self, path: Path, overwrite: bool = False) -> None:
        self.calls.append(("save", (Path(path), overwrite)))

    def load_data(self):
        self.calls.append(("load_data", None))
        return self

    def drop_channels(self, channels: list[str]) -> None:
        self.calls.append(("drop_channels", channels))
        self.ch_names = [c for c in self.ch_names if c not in channels]

    def set_eeg_reference(self, ref_channels: str = "average") -> None:
        self.calls.append(("set_eeg_reference", ref_channels))

    def filter(self, l_freq: float, h_freq: float) -> None:
        self.calls.append(("filter", (l_freq, h_freq)))

    def interpolate_bads(self, reset_bads: bool = True, method: str = "spline") -> None:
        self.calls.append(("interpolate_bads", (reset_bads, method)))
        if reset_bads:
            self.info["bads"] = []


@dataclass
class DummyEpochs:
    """Small replacement for mne.Epochs that records constructor arguments."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    saved: tuple[Path, bool] | None = None

    def save(self, path: Path, overwrite: bool = False) -> None:
        self.saved = (Path(path), overwrite)


class DummyICA:
    """Tiny ICA test double used to verify that `.apply(raw)` is invoked."""

    def __init__(self) -> None:
        self.applied_to: Any = None

    def apply(self, raw: Any) -> None:
        self.applied_to = raw


@pytest.fixture
def dummy_raw() -> DummyRaw:
    """Reusable fake Raw object."""
    return DummyRaw()


@pytest.fixture
def sample_config_dict() -> dict[str, Any]:
    """Small config payload with path and EEG montage entries."""
    return {
        "paths": {
            "raw_root": "raw",
            "derived_root": "derived",
            "results_root": "results",
            "reports_root": "reports",
        },
        "eeg": {"montage": "biosemi64"},
    }


@pytest.fixture
def sample_events() -> np.ndarray:
    """Simple MNE-compatible events matrix with one event code."""
    return np.array([[100, 0, 1], [250, 0, 1]], dtype=int)
