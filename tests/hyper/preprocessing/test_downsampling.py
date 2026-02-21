"""Tests for EDF-to-FIF downsampling pipeline logic.

These tests patch MNE I/O to validate control flow decisions without touching
real EEG files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hyper.config import ProjectConfig
from hyper.preprocessing import downsampling as mod


def test_read_channels_tsv_extracts_bads_and_types(tmp_path: Path) -> None:
    """BIDS channels metadata should map into bad-channel and type structures."""
    channels_path = tmp_path / "channels.tsv"
    pd.DataFrame(
        {
            "name": ["Fp1", "HEOG"],
            "status": ["bad", "good"],
            "type": ["EEG", "EOG"],
        }
    ).to_csv(channels_path, sep="\t", index=False)

    info = mod._read_channels_tsv(channels_path)

    assert info.bads == ("Fp1",)
    assert info.channel_types == {"Fp1": "eeg", "HEOG": "eog"}


def test_downsample_edf_to_fif_applies_processing(monkeypatch, tmp_path: Path, dummy_raw) -> None:
    """Pipeline should crop, set metadata, resample when needed, and save."""
    channels_path = tmp_path / "channels.tsv"
    pd.DataFrame({"name": ["Fp1"], "status": ["bad"], "type": ["EEG"]}).to_csv(
        channels_path, sep="\t", index=False
    )

    monkeypatch.setattr(mod.mne.io, "read_raw_edf", lambda *a, **k: dummy_raw)
    monkeypatch.setattr(mod.mne, "find_events", lambda raw: np.array([[100, 0, 1]]))
    monkeypatch.setattr(mod.mne.channels, "make_standard_montage", lambda name: f"montage:{name}")

    cfg = ProjectConfig(raw={"eeg": {"montage": "biosemi64"}})
    mod.downsample_edf_to_fif(
        input_edf_path=tmp_path / "in.edf",
        channels_tsv_path=channels_path,
        output_fif_path=tmp_path / "out.fif",
        config=cfg,
        target_sfreq_hz=50.0,
    )

    call_names = [name for name, _ in dummy_raw.calls]
    assert "crop" in call_names
    assert "set_montage" in call_names
    assert "set_channel_types" in call_names
    assert "resample" in call_names
    assert "save" in call_names


def test_should_resample_uses_tolerance() -> None:
    """Tiny floating point differences below tolerance should not trigger resampling."""
    assert not mod._should_resample(100.0, 100.0 + 1e-8)
    assert mod._should_resample(100.0, 120.0)


def test_read_channels_tsv_handles_missing_name_column(tmp_path: Path) -> None:
    """Missing `name` should return empty metadata without crashing."""
    channels_path = tmp_path / "channels.tsv"
    pd.DataFrame({"status": ["bad"], "type": ["EEG"]}).to_csv(channels_path, sep="\t", index=False)

    info = mod._read_channels_tsv(channels_path)

    assert info.bads == tuple()
    assert info.channel_types == {}


def test_get_montage_name_returns_none_when_absent() -> None:
    """Montage helper should return None when config omits EEG montage."""
    cfg = ProjectConfig(raw={})
    assert mod._get_montage_name(cfg) is None


def test_downsample_skips_resample_when_sampling_matches(monkeypatch, tmp_path: Path, dummy_raw) -> None:
    """Resampling should not run when original and target sfreq are equal."""
    channels_path = tmp_path / "channels.tsv"
    pd.DataFrame({"name": ["Fp1"], "status": ["good"], "type": ["EEG"]}).to_csv(
        channels_path, sep="\t", index=False
    )

    monkeypatch.setattr(mod.mne.io, "read_raw_edf", lambda *a, **k: dummy_raw)
    monkeypatch.setattr(mod.mne, "find_events", lambda raw: np.array([[100, 0, 1]]))
    monkeypatch.setattr(mod.mne.channels, "make_standard_montage", lambda name: f"montage:{name}")

    cfg = ProjectConfig(raw={"eeg": {"montage": "biosemi64"}})
    mod.downsample_edf_to_fif(
        input_edf_path=tmp_path / "in.edf",
        channels_tsv_path=channels_path,
        output_fif_path=tmp_path / "out.fif",
        config=cfg,
        target_sfreq_hz=dummy_raw.info["sfreq"],
    )

    assert not any(name == "resample" for name, _ in dummy_raw.calls)
