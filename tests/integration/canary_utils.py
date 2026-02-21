"""Shared helpers for Snakemake canary integration tests and fixture updates."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest
import yaml


@dataclass(frozen=True, slots=True)
class CanarySpec:
    """Identity tuple for the canary data slice."""

    subject: str = "sub-006"
    task: str = "conversation"
    run: str = "1"


@dataclass(frozen=True, slots=True)
class CanaryPaths:
    """Resolved paths produced by canary setup and execution."""

    repo_root: Path
    run_root: Path
    derived_root: Path
    baseline_dir: Path
    manifest_path: Path
    checksums_path: Path
    config_path: Path
    target_path: Path


def prepare_canary_run(*, tmp_path: Path, spec: CanarySpec = CanarySpec()) -> CanaryPaths:
    """Create runtime config and derived tree for a canary Snakemake run."""
    repo_root = Path(__file__).resolve().parents[2]
    base_cfg_path = repo_root / "config" / "config.yaml"

    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bids_root = Path(cfg["paths"]["bids_root"])
    annotation_root = Path(cfg["paths"]["annotation_root"])
    if not bids_root.exists():
        pytest.skip(f"BIDS root not available on this machine: {bids_root}")
    if not annotation_root.exists():
        pytest.skip(f"Annotation root not available on this machine: {annotation_root}")

    original_derived = Path(cfg["paths"]["derived_root"])
    source_ica = original_derived / "precomputed_ica" / f"{spec.subject}_task-{spec.task}-ica.fif"
    if not source_ica.exists():
        pytest.skip(f"Required precomputed ICA not found: {source_ica}")

    run_root = tmp_path / "canary_run"
    derived_root = run_root / "derived"
    derived_root.mkdir(parents=True, exist_ok=True)

    target_ica = derived_root / "precomputed_ica" / source_ica.name
    target_ica.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_ica, target_ica)

    cfg["paths"]["derived_root"] = str(derived_root)
    cfg["debug"]["enabled"] = True
    cfg["debug"]["subjects"] = [spec.subject]
    cfg["tasks"]["include"] = [spec.task]
    cfg["runs"]["include"][spec.task] = [int(spec.run)]
    cfg["canary"] = {"subject": spec.subject, "task": spec.task, "run": spec.run}

    config_path = run_root / "canary_config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    baseline_dir = repo_root / "tests" / "fixtures" / "baseline" / "canary"
    manifest_path = baseline_dir / "expected_files.txt"
    checksums_path = baseline_dir / "expected_checksums.json"
    target_path = derived_root / "canary" / "all.done"

    return CanaryPaths(
        repo_root=repo_root,
        run_root=run_root,
        derived_root=derived_root,
        baseline_dir=baseline_dir,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
        config_path=config_path,
        target_path=target_path,
    )


def run_canary(paths: CanaryPaths) -> None:
    """Execute Snakemake canary target using the generated runtime config."""
    if shutil.which("snakemake") is None:
        pytest.skip("snakemake executable not found")

    snakefile = paths.repo_root / "workflow" / "Snakefile"
    cmd = [
        "snakemake",
        "-s",
        str(snakefile),
        "--configfile",
        str(paths.config_path),
        "--cores",
        "1",
        str(paths.target_path),
    ]

    env: dict[str, Any] = dict(os.environ)
    venv_bin = paths.repo_root / ".venv" / "bin"
    if venv_bin.exists():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    subprocess.run(cmd, check=True, cwd=paths.repo_root, env=env)


def read_manifest_relpaths(manifest_path: Path) -> list[str]:
    """Load non-comment, non-empty relative paths from baseline manifest."""
    if not manifest_path.exists():
        return []
    return [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def canary_expected_rule_outputs(*, derived_root: Path, spec: CanarySpec = CanarySpec()) -> list[Path]:
    """Return one representative expected output path per preprocessing rule."""
    subj = spec.subject
    task = spec.task
    run = spec.run
    return [
        derived_root / "eeg" / "downsampled" / f"{subj}_task-{task}_run-{run}_raw_ds.fif",
        derived_root / "eeg" / "reref" / f"{subj}_task-{task}_run-{run}_raw_reref.fif",
        derived_root / "eeg" / "ica_applied" / f"{subj}_task-{task}_run-{run}_raw_ica.fif",
        derived_root / "eeg" / "interpolated" / f"{subj}_task-{task}_run-{run}_raw_interp.fif",
        derived_root / "eeg" / "filtered" / f"{subj}_task-{task}_run-{run}_raw_filt.fif",
        derived_root / "beh" / "metadata" / f"{subj}_task-{task}_run-{run}_metadata.tsv",
        derived_root / "beh" / "metadata" / f"{subj}_task-{task}_run-{run}_events.npy",
        derived_root / "eeg" / "epochs" / f"{subj}_task-{task}_run-{run}_epochs-epo.fif",
        derived_root / "canary" / "all.done",
    ]


def file_sha256(path: Path) -> str:
    """Compute SHA256 checksum for a file with streaming reads."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_checksum_map(paths: list[Path], *, root: Path) -> dict[str, str]:
    """Build a normalized relative-path -> sha256 mapping."""
    checksums: dict[str, str] = {}
    for path in sorted(paths):
        rel = str(path.relative_to(root))
        checksums[rel] = file_sha256(path)
    return checksums


def write_json(path: Path, payload: dict[str, str]) -> None:
    """Write deterministic JSON with sorted keys and trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, str]:
    """Read JSON dictionary from disk."""
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj)}")
    return {str(k): str(v) for k, v in obj.items()}
