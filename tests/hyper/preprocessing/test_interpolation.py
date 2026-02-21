"""Tests for bad-channel interpolation wrapper behavior."""

from pathlib import Path

import pandas as pd
import pytest

from hyper.preprocessing import interpolation as mod


def test_interpolate_bads_requires_name_and_status_columns(monkeypatch, tmp_path: Path, dummy_raw) -> None:
    """Malformed channels TSV should raise explicit ValueError."""
    channels_path = tmp_path / "channels.tsv"
    pd.DataFrame({"foo": ["bar"]}).to_csv(channels_path, sep="\t", index=False)
    monkeypatch.setattr(mod.mne.io, "read_raw_fif", lambda *a, **k: dummy_raw)

    with pytest.raises(ValueError, match="must contain at least columns"):
        mod.interpolate_bads_fif_to_fif(
            input_fif_path=tmp_path / "in.fif",
            channels_tsv_path=channels_path,
            output_fif_path=tmp_path / "out.fif",
            config=object(),
        )


def test_interpolate_bads_calls_interpolation_only_when_needed(monkeypatch, tmp_path: Path, dummy_raw) -> None:
    """Interpolation should be skipped when no channels are marked bad."""
    channels_path = tmp_path / "channels.tsv"
    pd.DataFrame({"name": ["Fp1"], "status": ["good"]}).to_csv(channels_path, sep="\t", index=False)
    monkeypatch.setattr(mod.mne.io, "read_raw_fif", lambda *a, **k: dummy_raw)

    mod.interpolate_bads_fif_to_fif(
        input_fif_path=tmp_path / "in.fif",
        channels_tsv_path=channels_path,
        output_fif_path=tmp_path / "out.fif",
        config=object(),
    )

    assert not any(name == "interpolate_bads" for name, _ in dummy_raw.calls)
    assert any(name == "save" for name, _ in dummy_raw.calls)


def test_interpolate_bads_calls_interpolation_when_bads_present(monkeypatch, tmp_path: Path, dummy_raw) -> None:
    """Bad-channel entries should trigger interpolation and bad reset."""
    channels_path = tmp_path / "channels.tsv"
    pd.DataFrame({"name": ["Fp1"], "status": ["bad"]}).to_csv(channels_path, sep="\t", index=False)
    monkeypatch.setattr(mod.mne.io, "read_raw_fif", lambda *a, **k: dummy_raw)

    mod.interpolate_bads_fif_to_fif(
        input_fif_path=tmp_path / "in.fif",
        channels_tsv_path=channels_path,
        output_fif_path=tmp_path / "out.fif",
        config=object(),
        method="nearest",
    )

    assert any(name == "interpolate_bads" for name, _ in dummy_raw.calls)
    assert dummy_raw.info["bads"] == []
