# Integration Tests (Scaffold)

This folder will contain Snakemake canary regression tests.

Design goals:
- run a minimal, deterministic canary target
- compare produced files against `tests/fixtures/baseline/canary/`
- emit clear mismatch diagnostics (max abs/rel diffs, failing keys/columns)
- validate checksum fingerprints for one representative output per rule stage

Integration tests are marked with `@pytest.mark.integration` and should be run explicitly:

```bash
pytest -m integration -q
```
