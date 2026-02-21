"""One-command helper to refresh canary baseline fixtures.

Usage:
    PYTHONPATH=src python tests/integration/update_canary_baseline.py
"""

from pathlib import Path
import shutil

from tests.integration.canary_utils import (
    CanarySpec,
    build_checksum_map,
    canary_expected_rule_outputs,
    prepare_canary_run,
    read_manifest_relpaths,
    run_canary,
    write_json,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runtime_root = repo_root / "tests" / "fixtures" / ".runtime_canary"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    paths = prepare_canary_run(tmp_path=runtime_root, spec=CanarySpec())
    run_canary(paths)

    relpaths = read_manifest_relpaths(paths.manifest_path)
    if not relpaths:
        raise RuntimeError(f"No entries in manifest: {paths.manifest_path}")

    for rel in relpaths:
        src = paths.derived_root / rel
        dst = paths.baseline_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"updated: {dst}")

    expected_outputs = canary_expected_rule_outputs(derived_root=paths.derived_root, spec=CanarySpec())
    missing = [str(path) for path in expected_outputs if not path.exists()]
    if missing:
        raise RuntimeError("Missing expected canary outputs:\n" + "\n".join(missing))

    checksum_map = build_checksum_map(expected_outputs, root=paths.derived_root)
    write_json(paths.checksums_path, checksum_map)
    print(f"updated: {paths.checksums_path}")


if __name__ == "__main__":
    main()
