"""Tests for epoch creation pipeline wrapper.

These tests ensure argument validation and MNE wiring for events/metadata-driven
epoch creation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hyper.preprocessing import epoching as mod


def test_make_epochs_rejects_bad_event_shape(monkeypatch, tmp_path: Path, dummy_raw) -> None:
    """Events must be a 2D matrix of shape (n_events, 3)."""
    events_path = tmp_path / "events.npy"
    metadata_path = tmp_path / "metadata.tsv"
    np.save(events_path, np.array([1, 2, 3]))
    pd.DataFrame({"timestamp": [0.1]}).to_csv(metadata_path, sep="\t", index=False)

    monkeypatch.setattr(mod.mne.io, "read_raw_fif", lambda *a, **k: dummy_raw)

    with pytest.raises(ValueError, match="shape \(n_events, 3\)"):
        mod.make_epochs_fif_to_fif(
            raw_fif_path=tmp_path / "raw.fif",
            events_npy_path=events_path,
            metadata_tsv_path=metadata_path,
            output_epochs_path=tmp_path / "epochs-epo.fif",
            config=object(),
            tmin_s=-0.2,
            tmax_s=0.8,
            baseline=(0.0, 0.1),
            detrend=None,
        )


def test_make_epochs_creates_and_saves_epochs(monkeypatch, tmp_path: Path, dummy_raw, sample_events) -> None:
    """Valid inputs should instantiate MNE Epochs and save output."""
    events_path = tmp_path / "events.npy"
    metadata_path = tmp_path / "metadata.tsv"
    np.save(events_path, sample_events)
    pd.DataFrame({"timestamp": [0.1, 0.2]}).to_csv(metadata_path, sep="\t", index=False)

    captured = {}

    def _fake_epochs(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return type("_Epochs", (), {"save": lambda self, path, overwrite=False: captured.update({"save": (path, overwrite)})})()

    monkeypatch.setattr(mod.mne.io, "read_raw_fif", lambda *a, **k: dummy_raw)
    monkeypatch.setattr(mod.mne, "Epochs", _fake_epochs)

    out_path = tmp_path / "out" / "epochs-epo.fif"
    mod.make_epochs_fif_to_fif(
        raw_fif_path=tmp_path / "raw.fif",
        events_npy_path=events_path,
        metadata_tsv_path=metadata_path,
        output_epochs_path=out_path,
        config=object(),
        tmin_s=-0.2,
        tmax_s=0.8,
        baseline=(0.0, 0.1),
        detrend=None,
    )

    assert captured["kwargs"]["tmin"] == -0.2
    assert captured["kwargs"]["tmax"] == 0.8
    assert captured["save"] == (out_path, True)
