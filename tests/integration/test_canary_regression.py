"""Snakemake canary integration tests.

The canary slice should be cheap and deterministic enough for routine
regression checks while still traversing one datapoint through the rule chain.
"""


import pytest

from hyper.testing.regression import RegressionTolerance, assert_paths_equal
from tests.integration.canary_utils import (
    CanarySpec,
    build_checksum_map,
    canary_expected_rule_outputs,
    prepare_canary_run,
    read_json,
    read_manifest_relpaths,
    run_canary,
)


@pytest.fixture(scope="module")
def canary_paths(tmp_path_factory):
    """Run canary once per module and provide resolved run/baseline paths."""
    tmp_path = tmp_path_factory.mktemp("integration_canary")
    paths = prepare_canary_run(tmp_path=tmp_path, spec=CanarySpec())
    run_canary(paths)
    return paths


@pytest.mark.integration
def test_snakemake_canary_regression_against_baseline(canary_paths) -> None:
    """Compare canary outputs listed in manifest against frozen baseline files."""
    relpaths = read_manifest_relpaths(canary_paths.manifest_path)
    if not relpaths:
        pytest.skip(f"No baseline files listed in manifest: {canary_paths.manifest_path}")

    missing_baseline = [rel for rel in relpaths if not (canary_paths.baseline_dir / rel).exists()]
    if missing_baseline:
        pytest.skip(
            "Baseline files are not populated yet. Missing:\n"
            + "\n".join(f"- {rel}" for rel in missing_baseline)
        )

    tol = RegressionTolerance(rtol=1e-6, atol=1e-8)
    for rel in relpaths:
        assert_paths_equal(
            actual=canary_paths.derived_root / rel,
            expected=canary_paths.baseline_dir / rel,
            tolerance=tol,
        )


@pytest.mark.integration
def test_snakemake_canary_produces_one_output_per_preprocessing_rule(canary_paths) -> None:
    """Ensure the canary run leaves one representative artifact per rule stage."""
    expected = canary_expected_rule_outputs(derived_root=canary_paths.derived_root, spec=CanarySpec())

    missing = [str(path) for path in expected if not path.exists()]
    assert not missing, "Missing expected canary outputs:\n" + "\n".join(missing)

    empty = [str(path) for path in expected if path.exists() and path.stat().st_size == 0]
    assert not empty, "Empty canary outputs detected:\n" + "\n".join(empty)


@pytest.mark.integration
def test_snakemake_canary_rule_outputs_match_checksum_baseline(canary_paths) -> None:
    """Compare checksums for all representative per-rule canary outputs."""
    expected_paths = canary_expected_rule_outputs(derived_root=canary_paths.derived_root, spec=CanarySpec())

    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        pytest.skip("Cannot compute checksums because some canary outputs are missing:\n" + "\n".join(missing))

    if not canary_paths.checksums_path.exists():
        pytest.skip(f"Checksum baseline missing: {canary_paths.checksums_path}")

    actual_checksums = build_checksum_map(expected_paths, root=canary_paths.derived_root)
    expected_checksums = read_json(canary_paths.checksums_path)

    if actual_checksums != expected_checksums:
        missing_keys = sorted(set(expected_checksums) - set(actual_checksums))
        extra_keys = sorted(set(actual_checksums) - set(expected_checksums))
        mismatch_keys = sorted(
            k
            for k in set(actual_checksums).intersection(expected_checksums)
            if actual_checksums[k] != expected_checksums[k]
        )

        lines = ["Canary checksum mismatch detected."]
        if missing_keys:
            lines.append("Missing keys:\n" + "\n".join(f"- {k}" for k in missing_keys))
        if extra_keys:
            lines.append("Extra keys:\n" + "\n".join(f"- {k}" for k in extra_keys))
        if mismatch_keys:
            lines.append(
                "Differing checksums:\n"
                + "\n".join(
                    f"- {k}: actual={actual_checksums[k]} expected={expected_checksums[k]}"
                    for k in mismatch_keys
                )
            )
        raise AssertionError("\n".join(lines))
