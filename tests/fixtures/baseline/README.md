# Baseline Fixtures

`tests/fixtures/baseline/` stores expected outputs for integration regression tests.

## Layout

- `canary/`: expected outputs for the minimal Snakemake canary slice

## Notes

- Keep baseline deterministic and as small as possible.
- If baseline size grows too large for normal git usage, adopt one of:
  - Git LFS
  - DVC-managed artifacts
  - CI artifact download in test setup
