# Test Layout

This repository uses two complementary test layers.

## `tests/hyper`

Purpose:
- fast, pure unit tests for Python modules under `src/hyper`
- no Snakemake DAG execution required

Typical usage:
- `pytest tests/hyper -q`

## `tests/integration`

Purpose:
- workflow-level tests that execute a small Snakemake canary slice
- regression checks against frozen baseline artifacts in `tests/fixtures/baseline/`

Status:
- scaffolded and wired to a Snakemake canary target
- skips automatically until baseline files are populated

Typical usage:
- `pytest -m integration -q`

## `tests/fixtures/baseline`

Purpose:
- stores expected outputs for canary regression comparisons

Current strategy:
- keep lightweight baseline files in git when feasible
- if baseline artifacts become too large, move to Git LFS / DVC / CI artifact download

## Baseline generation/update (canary)

Use a trusted environment (with access to BIDS + precomputed ICA paths from
`config/config.yaml`) and run:

```bash
snakemake -s workflow/Snakefile --cores 1 "$(python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path('config/config.yaml').read_text())
print(Path(cfg['paths']['derived_root']) / 'canary' / 'all.done')
PY
)"
```

Then copy files listed in
`tests/fixtures/baseline/canary/expected_files.txt` from the canary derived
root into `tests/fixtures/baseline/canary/`, preserving relative paths.

Or use the helper script:

```bash
PYTHONPATH=src python tests/integration/update_canary_baseline.py
```

This updates:
- baseline files listed in `expected_files.txt`
- `expected_checksums.json` for full per-rule canary checksum regression
