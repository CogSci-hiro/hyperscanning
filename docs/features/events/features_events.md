# Event-Like Feature Templates

This document defines the publication template for sparse event-like derivatives.

## What counts as an event-like feature

An event-like feature is a row-based annotation table in which each row corresponds to a time-anchored event or interval rather than a value at every EEG sample.

Examples include:

- words
- phonemes
- vowels
- IPUs
- turn transitions
- language-annotation rows used to map separate embedding arrays

Event-like derivatives are shared as:

- `*.tsv` for the event table
- `*.json` for the metadata sidecar

## Timing conventions

- `onset` is measured in seconds from run onset unless explicitly stated otherwise.
- `duration` should be included for interval-like rows.
- Point events may use `duration = 0.0`.
- If additional timing columns are needed, define them explicitly in the sidecar.

## Row semantics

Each row must represent one scientifically interpretable unit.

Good row definitions include:

- one spoken word token
- one aligned phoneme
- one vowel interval
- one IPU
- one turn transition anchored to an offset or onset

Avoid packing multiple distinct units into one row unless the aggregation rule is spelled out in `Method.AggregationRules`.

## Required sidecar fields

All event-feature JSON sidecars should contain these top-level fields:

- `FeatureName`
- `FeatureLabel`
- `FeatureType`
- `Description`
- `TimeBase`
- `Delimiter`
- `RowDefinition`
- `Columns`
- `Source`
- `Generation`
- `Method`
- `QualityControl`
- `FeatureFile`
- `Notes`

## Column documentation rule

Every TSV column must be documented in the JSON sidecar under `Columns`.
For each column, document:

- `Description`
- `Units`
- `DType`
- `Levels` when categorical
- `MissingValues` when applicable

## Missing-value conventions

- Use `n/a` for unavailable values in TSV files.
- Do not silently zero-fill failed extractions.
- If a value is unavailable because alignment failed, include both `n/a` and a status column when possible.
- For vowel features, include `extraction_status` and `notes`.

## QC conventions

Each event sidecar should include at least:

- `NumRows`
- `NumMissingValues`
- `ExcludedCount`
- relevant category or status counts
- any timing validity flags the writer can compute

Recommended release checks:

- both `.tsv` and `.json` files exist
- required timing columns are present
- every TSV column appears in `Columns`
- units are explicit where applicable
- missing values are encoded as `n/a` rather than silently imputed

## Feature-family notes

### Word features

Document:

- tokenization or forced-alignment source
- normalization rules
- POS tag set if included
- surprisal or entropy definitions if included
- mapping fields used to join separate language-model arrays, if present

POS tags should live in the word table rather than in a redundant standalone POS derivative unless the project later has a strong reason to split them.

### Phoneme features

Document:

- label convention, for example IPA, SAMPA, or a forced-aligner-specific set
- whether rows reflect phone intervals, phonological categories, or merged labels
- any confidence or alignment status fields

### Vowel features

Document:

- vowel label convention
- source of the intervals
- extraction parameters for formant estimation
- explicit semantics for `extraction_status`

Median `F1` and `F2` values are event-level by default and should not be recast as continuous arrays unless a separate derivative is intentionally defined.

### Turn transitions

Document:

- what anchors the row onset
- whether positive values indicate gap and negative values indicate overlap
- whether `latency` is identical to or distinct from `gap_or_overlap`

### Embedding-linked annotations

If high-dimensional embeddings are distributed later:

- keep timing, token, and alignment metadata in the TSV table
- keep vectors in separate array files
- document the external vector file and row-mapping fields in the JSON sidecar

## Filename examples

- `sub-01_ses-01_task-conversation_run-01_desc-words_features.tsv`
- `sub-01_ses-01_task-conversation_run-01_desc-words_features.json`
- `sub-01_ses-01_task-conversation_run-01_desc-phonemes_features.tsv`
- `sub-01_ses-01_task-conversation_run-01_desc-phonemes_features.json`
- `sub-01_ses-01_task-conversation_run-01_desc-vowels_features.tsv`
- `sub-01_ses-01_task-conversation_run-01_desc-vowels_features.json`

## Template status

The TSV and JSON files in this directory are examples and templates for future writers.
They provide concrete field names and row patterns, but final values such as file paths, counts, and software versions should be filled at generation time.
