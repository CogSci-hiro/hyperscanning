# Canary Baseline (Scaffold)

This directory stores baseline artifacts for the Snakemake canary slice.

Expected canary identity (default):
- subject: `sub-006`
- task: `conversation`
- run: `1`

Populate files listed in `expected_files.txt` under this directory, preserving
relative paths.

`expected_checksums.json` stores SHA256 hashes for one representative output per
preprocessing rule (including FIF artifacts) and is used by integration tests
for broad canary regression coverage without committing large binary fixtures.
