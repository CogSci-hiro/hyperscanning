"""Type-aware regression comparison helpers for canary/integration testing.

This module centralizes file-format-specific comparisons so integration tests can
report actionable diffs instead of generic byte mismatches.
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd


# ==================================================================================================
#                                   TYPES
# ==================================================================================================

@dataclass(frozen=True, slots=True)
class RegressionTolerance:
    """Numeric tolerance settings shared by array/table comparators."""

    rtol: float = 1e-6
    atol: float = 1e-8


@dataclass(frozen=True, slots=True)
class RegressionResult:
    """Detailed outcome returned by `compare_paths(...)`."""

    ok: bool
    message: str


# ==================================================================================================
#                                   HELPERS
# ==================================================================================================

def _compare_npy(actual: Path, expected: Path, *, tolerance: RegressionTolerance) -> RegressionResult:
    actual_arr = np.load(actual, allow_pickle=False)
    expected_arr = np.load(expected, allow_pickle=False)
    return _compare_arrays(actual_arr, expected_arr, tolerance=tolerance, context=str(actual))


def _compare_npz(actual: Path, expected: Path, *, tolerance: RegressionTolerance) -> RegressionResult:
    actual_npz = np.load(actual, allow_pickle=False)
    expected_npz = np.load(expected, allow_pickle=False)

    actual_keys = sorted(actual_npz.files)
    expected_keys = sorted(expected_npz.files)
    if actual_keys != expected_keys:
        return RegressionResult(
            False,
            f"NPZ key mismatch for {actual}: actual={actual_keys}, expected={expected_keys}",
        )

    for key in actual_keys:
        result = _compare_arrays(
            actual_npz[key],
            expected_npz[key],
            tolerance=tolerance,
            context=f"{actual}::{key}",
        )
        if not result.ok:
            return result

    # Each key matched, so the NPZ container as a whole is considered equal.
    return RegressionResult(True, f"NPZ match: {actual}")


def _compare_delimited(actual: Path, expected: Path, *, tolerance: RegressionTolerance) -> RegressionResult:
    sep = "\t" if actual.suffix.lower() == ".tsv" else ","
    actual_df = pd.read_csv(actual, sep=sep)
    expected_df = pd.read_csv(expected, sep=sep)

    actual_cols = list(actual_df.columns)
    expected_cols = list(expected_df.columns)
    if actual_cols != expected_cols:
        return RegressionResult(
            False,
            f"Column mismatch for {actual}: actual={actual_cols}, expected={expected_cols}",
        )

    actual_sorted = _stable_sort_df(actual_df)
    expected_sorted = _stable_sort_df(expected_df)

    if len(actual_sorted) != len(expected_sorted):
        return RegressionResult(
            False,
            f"Row count mismatch for {actual}: actual={len(actual_sorted)}, expected={len(expected_sorted)}",
        )

    numeric_cols = [
        col
        for col in actual_cols
        if pd.api.types.is_numeric_dtype(actual_sorted[col]) and pd.api.types.is_numeric_dtype(expected_sorted[col])
    ]
    text_cols = [col for col in actual_cols if col not in numeric_cols]

    for col in text_cols:
        lhs = actual_sorted[col].astype(str).to_numpy()
        rhs = expected_sorted[col].astype(str).to_numpy()
        diff_idx = np.where(lhs != rhs)[0]
        if diff_idx.size > 0:
            i = int(diff_idx[0])
            return RegressionResult(
                False,
                f"Text mismatch for {actual}, column={col}, row={i}: actual={lhs[i]!r}, expected={rhs[i]!r}",
            )

    # Numeric columns are compared with tolerances because floating-point rounding varies.
    for col in numeric_cols:
        lhs = actual_sorted[col].to_numpy(dtype=float)
        rhs = expected_sorted[col].to_numpy(dtype=float)
        arr_result = _compare_arrays(lhs, rhs, tolerance=tolerance, context=f"{actual}::{col}")
        if not arr_result.ok:
            return arr_result

    return RegressionResult(True, f"Delimited file match: {actual}")


def _compare_json(actual: Path, expected: Path) -> RegressionResult:
    with actual.open("r", encoding="utf-8") as f_actual:
        actual_obj = json.load(f_actual)
    with expected.open("r", encoding="utf-8") as f_expected:
        expected_obj = json.load(f_expected)

    if _canonical_json(actual_obj) != _canonical_json(expected_obj):
        return RegressionResult(False, f"JSON mismatch for {actual}")

    return RegressionResult(True, f"JSON match: {actual}")


def _compare_text(actual: Path, expected: Path) -> RegressionResult:
    lhs = actual.read_text(encoding="utf-8")
    rhs = expected.read_text(encoding="utf-8")
    if lhs != rhs:
        return RegressionResult(False, f"Text mismatch for {actual}")
    return RegressionResult(True, f"Text match: {actual}")


def _compare_hdf5(actual: Path, expected: Path, *, tolerance: RegressionTolerance) -> RegressionResult:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Comparing .h5/.hdf5 files requires h5py to be installed") from exc

    with h5py.File(actual, "r") as f_actual, h5py.File(expected, "r") as f_expected:
        actual_keys = sorted(_h5_dataset_keys(f_actual))
        expected_keys = sorted(_h5_dataset_keys(f_expected))

        if actual_keys != expected_keys:
            return RegressionResult(
                False,
                f"HDF5 dataset key mismatch for {actual}: actual={actual_keys}, expected={expected_keys}",
            )

        for key in actual_keys:
            lhs = np.asarray(f_actual[key][...])
            rhs = np.asarray(f_expected[key][...])
            arr_result = _compare_arrays(lhs, rhs, tolerance=tolerance, context=f"{actual}::{key}")
            if not arr_result.ok:
                return arr_result

    return RegressionResult(True, f"HDF5 match: {actual}")


def _compare_arrays(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    tolerance: RegressionTolerance,
    context: str,
) -> RegressionResult:
    lhs_arr = np.asarray(lhs)
    rhs_arr = np.asarray(rhs)

    if lhs_arr.shape != rhs_arr.shape:
        return RegressionResult(False, f"Shape mismatch for {context}: actual={lhs_arr.shape}, expected={rhs_arr.shape}")

    if lhs_arr.dtype.kind in {"f", "i", "u", "c"} and rhs_arr.dtype.kind in {"f", "i", "u", "c"}:
        if np.allclose(lhs_arr, rhs_arr, rtol=tolerance.rtol, atol=tolerance.atol, equal_nan=True):
            return RegressionResult(True, f"Array match for {context}")

        diff = np.abs(lhs_arr.astype(float) - rhs_arr.astype(float))
        max_abs = float(np.nanmax(diff))

        denom = np.maximum(np.abs(rhs_arr.astype(float)), tolerance.atol)
        rel = diff / denom
        max_rel = float(np.nanmax(rel))

        idx = tuple(int(v) for v in np.unravel_index(int(np.nanargmax(diff)), diff.shape)) if diff.size else ()
        return RegressionResult(
            False,
            (
                f"Numeric mismatch for {context}: max_abs={max_abs:.6g}, "
                f"max_rel={max_rel:.6g}, worst_index={idx}, rtol={tolerance.rtol}, atol={tolerance.atol}"
            ),
        )

    if np.array_equal(lhs_arr, rhs_arr):
        return RegressionResult(True, f"Array match for {context}")

    return RegressionResult(False, f"Exact array mismatch for {context}")


def _stable_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    sort_proxy = df.copy()
    sort_cols: list[str] = []

    for col in df.columns:
        proxy_col = f"__sort__{col}"
        sort_proxy[proxy_col] = df[col].astype(str)
        sort_cols.append(proxy_col)

    # Mergesort keeps stability so equal rows remain in deterministic order.
    sorted_index = sort_proxy.sort_values(by=sort_cols, kind="mergesort").index
    return df.loc[sorted_index].reset_index(drop=True)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _h5_dataset_keys(h5_group: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key, value in h5_group.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if hasattr(value, "keys"):
            keys.extend(_h5_dataset_keys(value, prefix=full_key))
        else:
            keys.append(full_key)
    return keys


# ==================================================================================================
#                                   CORE LOGIC
# ==================================================================================================

def assert_paths_equal(
    actual: Path,
    expected: Path,
    *,
    tolerance: RegressionTolerance = RegressionTolerance(),
) -> None:
    """Assert that two files are equivalent under a type-aware comparator."""
    result = compare_paths(actual, expected, tolerance=tolerance)
    if not result.ok:
        raise AssertionError(result.message)


def compare_paths(
    actual: Path,
    expected: Path,
    *,
    tolerance: RegressionTolerance = RegressionTolerance(),
) -> RegressionResult:
    """Compare two files using format-specific logic based on suffix."""
    if not expected.exists():
        return RegressionResult(False, f"Missing expected baseline file: {expected}")
    if not actual.exists():
        return RegressionResult(False, f"Missing actual output file: {actual}")

    suffix = actual.suffix.lower()
    if suffix != expected.suffix.lower():
        return RegressionResult(False, f"Suffix mismatch: {actual.suffix} vs {expected.suffix}")

    if suffix == ".npy":
        return _compare_npy(actual, expected, tolerance=tolerance)
    if suffix == ".npz":
        return _compare_npz(actual, expected, tolerance=tolerance)
    if suffix in {".csv", ".tsv"}:
        return _compare_delimited(actual, expected, tolerance=tolerance)
    if suffix == ".json":
        return _compare_json(actual, expected)
    if suffix in {".txt", ".done"}:
        return _compare_text(actual, expected)
    if suffix in {".h5", ".hdf5"}:
        return _compare_hdf5(actual, expected, tolerance=tolerance)

    # Fall back to an explicit rejection so maintainers know when a new type appears.
    return RegressionResult(False, f"Unsupported regression file type: {suffix} ({actual})")
