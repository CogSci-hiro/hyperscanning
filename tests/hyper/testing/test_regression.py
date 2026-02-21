"""Unit tests for regression comparison helpers.

These tests focus on clear failure messaging and deterministic type-aware
comparisons across supported artifact formats.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from hyper.testing.regression import RegressionTolerance, assert_paths_equal, compare_paths


def test_compare_npy_matches_within_tolerance(tmp_path: Path) -> None:
    """Numeric arrays should match when differences are below tolerances."""
    actual = tmp_path / "actual.npy"
    expected = tmp_path / "expected.npy"

    np.save(actual, np.array([1.0, 2.0, 3.0]))
    np.save(expected, np.array([1.0, 2.0, 3.0 + 1e-9]))

    result = compare_paths(actual, expected)
    assert result.ok


def test_compare_npy_reports_max_diff_and_index(tmp_path: Path) -> None:
    """Numeric mismatch message should include max abs/rel and worst index."""
    actual = tmp_path / "actual.npy"
    expected = tmp_path / "expected.npy"

    np.save(actual, np.array([1.0, 99.0]))
    np.save(expected, np.array([1.0, 2.0]))

    result = compare_paths(actual, expected, tolerance=RegressionTolerance(rtol=1e-12, atol=1e-12))
    assert not result.ok
    assert "max_abs=" in result.message
    assert "worst_index=" in result.message


def test_compare_npz_requires_same_keys(tmp_path: Path) -> None:
    """NPZ comparator should fail fast when key sets differ."""
    actual = tmp_path / "actual.npz"
    expected = tmp_path / "expected.npz"

    np.savez(actual, a=np.array([1]))
    np.savez(expected, b=np.array([1]))

    result = compare_paths(actual, expected)
    assert not result.ok
    assert "NPZ key mismatch" in result.message


def test_compare_csv_ignores_row_order_with_stable_sort(tmp_path: Path) -> None:
    """CSV comparison should be order-insensitive but value-sensitive."""
    actual = tmp_path / "actual.csv"
    expected = tmp_path / "expected.csv"

    pd.DataFrame({"id": [2, 1], "value": [20.0, 10.0]}).to_csv(actual, index=False)
    pd.DataFrame({"id": [1, 2], "value": [10.0, 20.0]}).to_csv(expected, index=False)

    result = compare_paths(actual, expected)
    assert result.ok


def test_compare_tsv_reports_text_mismatch_location(tmp_path: Path) -> None:
    """Text mismatch should include row/column context."""
    actual = tmp_path / "actual.tsv"
    expected = tmp_path / "expected.tsv"

    pd.DataFrame({"label": ["A"], "x": [1.0]}).to_csv(actual, sep="\t", index=False)
    pd.DataFrame({"label": ["B"], "x": [1.0]}).to_csv(expected, sep="\t", index=False)

    result = compare_paths(actual, expected)
    assert not result.ok
    assert "column=label" in result.message


def test_compare_json_canonicalizes_key_order(tmp_path: Path) -> None:
    """JSON key order differences should not fail comparison."""
    actual = tmp_path / "actual.json"
    expected = tmp_path / "expected.json"

    actual.write_text(json.dumps({"b": 2, "a": 1}), encoding="utf-8")
    expected.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")

    result = compare_paths(actual, expected)
    assert result.ok


def test_compare_unsupported_extension_returns_clear_error(tmp_path: Path) -> None:
    """Unsupported file suffixes should return an explanatory result."""
    actual = tmp_path / "a.bin"
    expected = tmp_path / "b.bin"
    actual.write_text("x", encoding="utf-8")
    expected.write_text("x", encoding="utf-8")

    result = compare_paths(actual, expected)
    assert not result.ok
    assert "Unsupported regression file type" in result.message


def test_assert_paths_equal_raises_on_mismatch(tmp_path: Path) -> None:
    """Assertion wrapper should raise with comparator message when not equal."""
    actual = tmp_path / "a.npy"
    expected = tmp_path / "b.npy"
    np.save(actual, np.array([1.0]))
    np.save(expected, np.array([2.0]))

    with pytest.raises(AssertionError, match="Numeric mismatch"):
        assert_paths_equal(actual, expected, tolerance=RegressionTolerance(rtol=0.0, atol=0.0))


def test_compare_hdf5_keys_and_arrays(tmp_path: Path) -> None:
    """HDF5 comparator should validate dataset keys and numeric payloads."""
    h5py = pytest.importorskip("h5py")

    actual = tmp_path / "actual.h5"
    expected = tmp_path / "expected.h5"

    with h5py.File(actual, "w") as f:
        grp = f.create_group("g")
        grp.create_dataset("x", data=np.array([1.0, 2.0]))

    with h5py.File(expected, "w") as f:
        grp = f.create_group("g")
        grp.create_dataset("x", data=np.array([1.0, 2.0]))

    result = compare_paths(actual, expected)
    assert result.ok


def test_compare_paths_reports_missing_expected(tmp_path: Path) -> None:
    """Missing baseline files should return clear, actionable messages."""
    actual = tmp_path / "actual.npy"
    np.save(actual, np.array([1.0]))

    result = compare_paths(actual, tmp_path / "missing.npy")
    assert not result.ok
    assert "Missing expected baseline file" in result.message


def test_compare_paths_reports_suffix_mismatch(tmp_path: Path) -> None:
    """Comparisons should fail fast when extensions do not match."""
    actual = tmp_path / "a.npy"
    expected = tmp_path / "a.csv"
    np.save(actual, np.array([1.0]))
    expected.write_text("x\n1\n", encoding="utf-8")

    result = compare_paths(actual, expected)
    assert not result.ok
    assert "Suffix mismatch" in result.message


def test_compare_hdf5_reports_key_mismatch(tmp_path: Path) -> None:
    """HDF5 comparator should identify differing dataset paths."""
    h5py = pytest.importorskip("h5py")

    actual = tmp_path / "actual.h5"
    expected = tmp_path / "expected.h5"

    with h5py.File(actual, "w") as f:
        f.create_dataset("a", data=np.array([1.0]))
    with h5py.File(expected, "w") as f:
        f.create_dataset("b", data=np.array([1.0]))

    result = compare_paths(actual, expected)
    assert not result.ok
    assert "HDF5 dataset key mismatch" in result.message


def test_compare_done_file_exact_text(tmp_path: Path) -> None:
    """Sentinel .done files should compare by exact text content."""
    actual = tmp_path / "a.done"
    expected = tmp_path / "b.done"
    actual.write_text("canary_ok\n", encoding="utf-8")
    expected.write_text("canary_ok\n", encoding="utf-8")

    result = compare_paths(actual, expected)
    assert result.ok
