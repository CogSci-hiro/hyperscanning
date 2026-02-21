"""Tests for conversational metadata construction and event conversion."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hyper.preprocessing import metadata as mod


def _ipu_df(rows: list[tuple[float, float, str, float, int, float]]) -> pd.DataFrame:
    """Build a compact IPU-like table for tests."""
    return pd.DataFrame(
        rows,
        columns=["start", "end", "annotation", "duration", "n_syllables", "rate"],
    )


def test_infer_partner_id_and_run_from_ipu_path() -> None:
    """Odd IDs should map to next even ID and preserve run number."""
    other_id, run = mod.infer_partner_id_and_run_from_ipu_path(Path("sub-001_run-3_ipu.csv"))
    assert other_id == "002"
    assert run == "3"


def test_infer_partner_id_and_run_rejects_bad_filename() -> None:
    """Unexpected filename patterns should fail with context-rich errors."""
    with pytest.raises(RuntimeError, match="must match pattern"):
        mod.infer_partner_id_and_run_from_ipu_path(Path("bad-name.csv"))


def test_metadata_df_to_mne_events_filters_non_numeric_timestamps() -> None:
    """Only finite numeric timestamps should become output events."""
    df = pd.DataFrame({"timestamp": [0.1, "bad", np.nan, 0.5]})

    events = mod.metadata_df_to_mne_events(df, sfreq_hz=100.0, first_samp=10, event_id=7)

    assert events.shape == (2, 3)
    assert events[:, 0].tolist() == [20, 60]
    assert events[:, 2].tolist() == [7, 7]


def test_metadata_df_to_mne_events_requires_timestamp_column() -> None:
    """Missing timestamp column should fail explicitly."""
    with pytest.raises(ValueError, match="must contain column"):
        mod.metadata_df_to_mne_events(pd.DataFrame({"x": [1]}), sfreq_hz=100.0, first_samp=0)


def test_make_metadata_onset_self_anchor_produces_expected_columns() -> None:
    """Metadata merge should produce timestamp, partner features, and abs_diff."""
    self_ipu = _ipu_df([
        (0.0, 0.5, "a", 0.5, 2, 4.0),
        (1.0, 1.5, "b", 0.5, 3, 6.0),
    ])
    other_ipu = _ipu_df([
        (0.2, 0.8, "x", 0.6, 2, 3.0),
        (1.2, 1.9, "y", 0.7, 4, 5.0),
    ])

    out = mod.make_metadata(self_ipu, other_ipu, time_lock="onset", anchor="self", margin_s=1.0)

    assert "timestamp" in out.columns
    assert "other_rate" in out.columns
    assert "latency" in out.columns
    assert "abs_diff" in out.columns
    # abs_diff should be exact absolute difference between speaker rates.
    np.testing.assert_allclose(
        out["abs_diff"].to_numpy(dtype=float),
        np.abs(out["self_rate"].to_numpy(dtype=float) - out["other_rate"].to_numpy(dtype=float)),
        rtol=0.0,
        atol=0.0,
    )


def test_make_metadata_validates_inputs() -> None:
    """Invalid control arguments should be rejected before processing."""
    self_ipu = _ipu_df([(0.0, 0.5, "a", 0.5, 1, 2.0)])
    other_ipu = _ipu_df([(0.1, 0.6, "b", 0.5, 1, 3.0)])

    with pytest.raises(ValueError, match="time_lock"):
        mod.make_metadata(self_ipu, other_ipu, time_lock="bad", anchor="self", margin_s=1.0)

    with pytest.raises(ValueError, match="anchor"):
        mod.make_metadata(self_ipu, other_ipu, time_lock="onset", anchor="bad", margin_s=1.0)

    with pytest.raises(ValueError, match="margin_s"):
        mod.make_metadata(self_ipu, other_ipu, time_lock="onset", anchor="self", margin_s=0.0)


def test_make_metadata_supports_offset_other_anchor() -> None:
    """Offset locking with other anchor should produce valid timestamp output."""
    self_ipu = _ipu_df([(0.0, 0.5, "a", 0.5, 1, 2.0)])
    other_ipu = _ipu_df([(0.1, 0.6, "b", 0.5, 2, 3.0)])

    out = mod.make_metadata(self_ipu, other_ipu, time_lock="offset", anchor="other", margin_s=1.0)

    assert "timestamp" in out.columns
    assert "self_rate" in out.columns
    assert "other_rate" in out.columns


def test_make_metadata_drops_placeholder_rows() -> None:
    """Rows with placeholder annotation should be removed before matching."""
    self_ipu = _ipu_df([
        (0.0, 0.2, "#", 0.2, 1, 1.0),
        (0.2, 0.7, "speech", 0.5, 2, 4.0),
    ])
    other_ipu = _ipu_df([(0.1, 0.6, "b", 0.5, 2, 3.0)])

    out = mod.make_metadata(self_ipu, other_ipu, time_lock="onset", anchor="self", margin_s=1.0)

    assert len(out) == 1
    assert float(out.iloc[0]["self_rate"]) == 4.0


def test_make_metadata_handles_no_partner_match_with_nans() -> None:
    """When no adjacent partner exists in margin, partner features should be NaN."""
    self_ipu = _ipu_df([(10.0, 10.5, "a", 0.5, 2, 4.0)])
    other_ipu = _ipu_df([(0.1, 0.6, "b", 0.5, 2, 3.0)])

    out = mod.make_metadata(self_ipu, other_ipu, time_lock="onset", anchor="self", margin_s=0.1)

    assert np.isnan(float(out.iloc[0]["other_rate"]))
    assert np.isnan(float(out.iloc[0]["latency"]))


def test_make_metadata_and_events_raises_when_partner_missing(tmp_path: Path) -> None:
    """End-to-end helper should fail clearly if inferred partner file does not exist."""
    self_path = tmp_path / "sub-001_run-1_ipu.csv"
    _ipu_df([(0.0, 0.5, "a", 0.5, 2, 4.0)]).to_csv(self_path, index=False)

    with pytest.raises(FileNotFoundError, match="Partner IPU CSV not found"):
        mod.make_metadata_and_events_from_self_ipu(
            self_ipu_csv_path=self_path,
            raw_path=tmp_path / "raw.fif",
            output_tsv_path=tmp_path / "out.tsv",
            output_events_npy_path=tmp_path / "events.npy",
            config=object(),
        )


def test_make_metadata_and_events_from_self_ipu_end_to_end(monkeypatch, tmp_path: Path) -> None:
    """End-to-end helper should write both metadata TSV and events NPY outputs."""
    self_path = tmp_path / "sub-001_run-1_ipu.csv"
    other_path = tmp_path / "sub-002_run-1_ipu.csv"
    raw_path = tmp_path / "raw.fif"
    out_tsv = tmp_path / "out" / "metadata.tsv"
    out_events = tmp_path / "out" / "events.npy"

    self_df = _ipu_df([(0.0, 0.5, "a", 0.5, 2, 4.0)])
    other_df = _ipu_df([(0.1, 0.6, "b", 0.5, 2, 3.0)])
    self_df.to_csv(self_path, index=False)
    other_df.to_csv(other_path, index=False)

    dummy_raw = type("_Raw", (), {"info": {"sfreq": 100.0}, "first_samp": 0})()
    monkeypatch.setattr(mod.mne.io, "read_raw", lambda *a, **k: dummy_raw)

    mod.make_metadata_and_events_from_self_ipu(
        self_ipu_csv_path=self_path,
        raw_path=raw_path,
        output_tsv_path=out_tsv,
        output_events_npy_path=out_events,
        config=object(),
    )

    assert out_tsv.exists()
    assert out_events.exists()
    saved_events = np.load(out_events)
    assert saved_events.shape[1] == 3
