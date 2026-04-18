# Continuous features
- **Alignment**: how exact length matching was achieved

## Missing-data conventions

The sidecar must document how missing or undefined values are represented.

### Examples

- unvoiced F0 segments encoded as `NaN`
- mask arrays encoded as `0` and `1`
- extraction failures retained as `NaN` rather than silently imputed

If interpolation is used for model-ready regressors, this must be clearly documented in `Method.Postprocessing` and `Method.Parameters`.

## Feature-specific notes

### Envelope

Envelope features should record:

- the extraction method or variant used
- filtering/compression details if applicable
- whether the implementation directly follows a published method or an approximation

### F0

F0 features should record:

- tracker used
- floor and ceiling settings
- handling of unvoiced regions
- whether the shared array is raw, interpolated, or both

### Binary masks

Mask features should record:

- semantic interpretation of `0` and `1`
- annotation source
- projection rule from events/intervals to samplewise representation

## Recommended documentation pattern

For each feature family:

1. Store the numeric data in `*.npy`
2. Store the metadata in `*.json`
3. Document shared conventions in this file
4. Add a dedicated feature-specific markdown file when the method needs extra detail or may be scrutinized by reviewers

## Validation checklist

Before release, each continuous feature should pass the following checks:

- data file exists
- sidecar file exists
- sample count matches EEG exactly
- sampling frequency is documented
- units are documented
- extraction parameters are documented
- missing-value handling is documented
- QC values are present

## Suggested filename pattern

Examples:

- `sub-01_ses-01_task-conversation_run-01_desc-f0_feature.npy`
- `sub-01_ses-01_task-conversation_run-01_desc-f0_feature.json`
- `sub-01_ses-01_task-conversation_run-01_desc-oganianEnv_feature.npy`
- `sub-01_ses-01_task-conversation_run-01_desc-oganianEnv_feature.json`

## Summary

Continuous features are shared as sample-aligned NumPy arrays with JSON sidecars that fully document provenance, processing, alignment, and quality control. The central requirement is exact compatibility with the EEG time base together with explicit, reproducible metadata.