# Event features

This document defines the storage, metadata, and documentation conventions for event-like features shared with the dataset.

## Scope

Event features are sparse, row-based features associated with discrete time intervals or onsets rather than a value at every EEG sample. They are stored as:

* `*.tsv` for the event table
* `*.json` for the sidecar metadata

These features are intended for reuse in TRF design matrix construction, regression-based analyses, annotation inspection, and general dataset interpretation.

## File-level convention

Each event feature is stored as a tabular file in which each row corresponds to a time-anchored observation.

### Standard representation

* **Data file**: `*_features.tsv`
* **Metadata file**: `*_features.json`
* **Time axis**: event-based, usually referenced in seconds from run onset
* **Row meaning**: one event, interval, token, segment, or annotation unit per row
* **Missing values**: encoded as `n/a` in TSV unless otherwise documented

## Core principles

### Explicit temporal anchoring

Each row must be tied to time explicitly. At minimum, an event table should include:

* `onset`
* optionally `duration`

with time expressed in seconds relative to run onset unless otherwise documented.

For interval-like events:

[
\mathrm{offset}_i = \mathrm{onset}_i + \mathrm{duration}_i
]

where row (i) defines one event or interval.

### One row, one interpretable unit

Each row should represent a scientifically meaningful unit, for example:

* one word
* one phoneme
* one vowel interval
* one IPU
* one turn transition

Do not collapse distinct units into a single row unless the aggregation rule is explicitly documented.

### Explicit provenance

Each sidecar must document:

* source annotation files
* source modality
* extraction software and version
* transformation steps
* definition of each column
* any filtering or exclusion rules

### Reproducibility

Any decision that affects which rows appear or how column values are computed must be documented in the sidecar. This includes tokenization rules, alignment rules, label normalization, exclusion criteria, and aggregation strategies.

## Standard metadata fields

All event-feature JSON sidecars should contain the following top-level sections:

* `FeatureName`
* `FeatureType`
* `Description`
* `TimeBase`
* `Delimiter`
* `Columns`
* `RowDefinition`
* `Source`
* `Generation`
* `Method`
* `QualityControl`
* `Notes`

## Required table columns

The exact columns depend on the feature family, but event tables should generally include:

* `onset`
* `duration` for interval-like features
* one or more label or value columns
* identifiers where useful, such as `speaker`, `word_index`, or `source_interval_id`

## Column documentation

Every column in the TSV must be described in the sidecar JSON under `Columns`.

Each column description should include:

* `Description`
* `Units`
* `DType`
* `Levels` when categorical
* `MissingValues` if relevant

## Missing-data conventions

Missing or undefined values in TSV files should be explicit.

### Recommended conventions

* use `n/a` for unavailable scalar values
* use empty fields only if documented and unavoidable
* never silently coerce extraction failures to zero

For example:

* unstable formant estimate → `n/a`
* unknown POS tag → documented missing category or `n/a`
* unmatched token mapping → `n/a` plus explanatory status field

## Common event-feature families

### Word-level features

Typical columns may include:

* `onset`
* `duration`
* `speaker`
* `word`
* `pos`
* `surprisal`
* `entropy`
* `token_count`
* `token_ids`

### Vowel-level acoustic features

Typical columns may include:

* `onset`
* `duration`
* `speaker`
* `vowel_label`
* `f1_median_hz`
* `f2_median_hz`
* `source_interval_id`
* `extraction_status`

### Structural annotation features

Typical columns may include:

* `onset`
* `duration`
* `speaker`
* `event_label`
* `source`
* `notes`

## Recommended documentation pattern

For each feature family:

1. Store the table in `*.tsv`
2. Store the metadata in `*.json`
3. Document shared conventions in this file
4. Add dedicated feature-specific markdown only when the feature definition is complex or likely to be scrutinized by reviewers

## Validation checklist

Before release, each event feature should pass the following checks:

* TSV file exists
* JSON sidecar exists
* required timing columns are present
* all columns are documented in JSON
* units are documented where applicable
* categorical levels are documented where applicable
* extraction and filtering rules are documented
* QC counts are present

## Suggested filename pattern

Examples:

* `sub-01_ses-01_task-conversation_run-01_desc-words_features.tsv`
* `sub-01_ses-01_task-conversation_run-01_desc-words_features.json`
* `sub-01_ses-01_task-conversation_run-01_desc-vowels_features.tsv`
* `sub-01_ses-01_task-conversation_run-01_desc-vowels_features.json`

## Summary

Event features are shared as sparse TSV tables with JSON sidecars that fully document timing, row semantics, columns, provenance, processing, and quality control. The central requirement is that each row correspond to a clearly defined, time-anchored scientific unit.
