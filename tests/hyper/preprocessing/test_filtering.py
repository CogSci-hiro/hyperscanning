"""Tests for band-pass filtering wrapper logic."""

from pathlib import Path

from hyper.preprocessing import filtering as mod


def test_bandpass_filter_fif_to_fif_calls_filter_and_save(monkeypatch, tmp_path: Path, dummy_raw) -> None:
    """Function should delegate to MNE read/filter/save in correct order."""
    monkeypatch.setattr(mod.mne.io, "read_raw_fif", lambda *a, **k: dummy_raw)

    mod.bandpass_filter_fif_to_fif(
        input_fif_path=tmp_path / "in.fif",
        output_fif_path=tmp_path / "out.fif",
        config=object(),
        l_freq_hz=1.0,
        h_freq_hz=40.0,
    )

    call_names = [name for name, _ in dummy_raw.calls]
    assert "filter" in call_names
    assert "save" in call_names
