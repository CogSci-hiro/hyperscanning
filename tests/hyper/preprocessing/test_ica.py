"""Tests for ICA application wrapper behavior."""

from pathlib import Path

from hyper.preprocessing import ica as mod


def test_apply_ica_fif_to_fif_applies_ica_and_saves(monkeypatch, tmp_path: Path, dummy_raw) -> None:
    """Top-level wrapper should load raw+ICA, apply, then save."""
    ica_obj = type("_Ica", (), {"applied": False, "apply": lambda self, raw: setattr(self, "applied", True)})()

    monkeypatch.setattr(mod.mne.io, "read_raw_fif", lambda *a, **k: dummy_raw)
    monkeypatch.setattr(mod.mne.preprocessing, "read_ica", lambda *a, **k: ica_obj)

    mod.apply_ica_fif_to_fif(
        input_fif_path=tmp_path / "in.fif",
        ica_path=tmp_path / "ica.fif",
        output_fif_path=tmp_path / "out.fif",
        config=object(),
    )

    assert ica_obj.applied is True
    assert any(name == "save" for name, _ in dummy_raw.calls)
