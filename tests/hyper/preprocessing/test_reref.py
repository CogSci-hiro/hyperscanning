"""Tests for rereferencing and montage application behavior."""

from pathlib import Path

import pandas as pd
import pytest

from hyper.preprocessing import reref as mod


def test_load_channels_tsv_requires_required_columns(tmp_path: Path) -> None:
    """Rereference metadata loader should fail clearly on malformed TSV."""
    channels_path = tmp_path / "channels.tsv"
    pd.DataFrame({"foo": [1]}).to_csv(channels_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="must contain at least columns"):
        mod.load_channels_tsv(channels_path)


def test_rereference_raw_runs_expected_steps(monkeypatch, dummy_raw) -> None:
    """Core reref flow should drop auxiliaries, set bads, reference, and montage."""
    channels_df = pd.DataFrame({"name": ["Fp1"], "status": ["bad"]})
    monkeypatch.setattr(mod.mne.channels, "make_standard_montage", lambda name: f"montage:{name}")

    mod.rereference_raw(dummy_raw, channels_df)

    call_names = [name for name, _ in dummy_raw.calls]
    assert "load_data" in call_names
    assert "drop_channels" in call_names
    assert "set_eeg_reference" in call_names
    assert "set_montage" in call_names
    assert dummy_raw.info["bads"] == ["Fp1"]


def test_rereference_fif_to_fif_wires_io(monkeypatch, tmp_path: Path, dummy_raw) -> None:
    """Top-level function should read input and save output after processing."""
    channels_path = tmp_path / "channels.tsv"
    pd.DataFrame({"name": ["Fp1"], "status": ["good"]}).to_csv(channels_path, sep="\t", index=False)

    monkeypatch.setattr(mod.mne.io, "read_raw_fif", lambda *a, **k: dummy_raw)
    monkeypatch.setattr(mod.mne.channels, "make_standard_montage", lambda name: f"montage:{name}")

    mod.rereference_fif_to_fif(
        input_fif_path=tmp_path / "in.fif",
        channels_tsv_path=channels_path,
        output_fif_path=tmp_path / "out.fif",
        config=object(),
    )

    assert any(name == "save" for name, _ in dummy_raw.calls)


def test_rereference_raw_supports_explicit_reference_channel(monkeypatch, dummy_raw) -> None:
    """Non-average references should be forwarded to `set_eeg_reference`."""
    channels_df = pd.DataFrame({"name": ["Fp1"], "status": ["good"]})
    monkeypatch.setattr(mod.mne.channels, "make_standard_montage", lambda name: f"montage:{name}")

    mod.rereference_raw(dummy_raw, channels_df, reference="Cz")

    assert ("set_eeg_reference", "Cz") in dummy_raw.calls
