# Continuous Feature Templates

This document defines the publication template for continuous feature derivatives.

## What counts as a continuous feature

A continuous feature is a numeric signal with one value per EEG sample.
Published continuous derivatives are intended for features such as speech envelope, F0, and samplewise conversational masks.

Continuous derivatives are shared as:

- `*.npy` for the numeric array
- `*.json` for the metadata sidecar

## Core alignment rule

Continuous features are aligned to the EEG time base exactly.

The final feature array must satisfy:

- `NumSamples == EEG sample count for the run`
- `Shape == [NumSamples]` unless a future feature explicitly documents extra axes
- `AlignmentTarget == "EEG"`
- `AlignmentAxis == "time"`

If extraction happens at another rate, the sidecar must document:

- source sampling frequency
- resampling method
- target EEG sampling frequency
- how exact final length matching was enforced

## Required sidecar fields

All continuous-feature JSON sidecars should contain these top-level fields:

- `FeatureName`
- `FeatureLabel`
- `FeatureType`
- `Description`
- `Units`
- `Shape`
- `DType`
- `SamplingFrequency`
- `SamplingFrequencyUnit`
- `AlignmentTarget`
- `AlignmentAxis`
- `NumSamples`
- `StartTime`
- `StartTimeUnit`
- `MissingValueEncoding`
- `ValueRange`
- `Source`
- `Generation`
- `Method`
- `QualityControl`
- `FeatureFile`
- `Notes`

## Units conventions

- Use explicit physical units when available, for example `Hz` for F0.
- Use `0/1` for binary masks and document the meaning of each value.
- Use `arbitrary` only when there is no stable physical unit, for example for some envelope representations.
- If values are transformed, normalized, or z-scored, say so explicitly in `Method.Postprocessing` and `Units`.

## Missing-value conventions

- Undefined continuous values should be encoded explicitly, typically as `NaN`.
- Do not silently zero-fill extraction failures.
- If a model-ready filled version is also shared, it should be a distinct derivative with its own sidecar.
- For F0, the sidecar must document unvoiced handling and whether interpolation or another fill strategy was applied before EEG alignment.

## QC conventions

Each continuous sidecar should report enough QC to validate both shape and signal validity:

- `ExactEEGSampleMatch`
- `FiniteValueRatio`
- `NumNaN`
- `NumInf`
- optional summary statistics
- feature-specific checks where relevant

Recommended release checks:

- both `.npy` and `.json` files exist
- `NumSamples` matches the EEG sample count exactly
- `SamplingFrequency` equals the EEG sampling frequency used for the published run
- units are explicit
- extraction and resampling steps are documented
- missing-value handling is explicit

## Feature-family notes

### Envelope

Document:

- extraction method or library
- envelope variant
- smoothing or compression steps
- whether the published derivative is a direct library output or a project-specific approximation

### F0

Document:

- tracker or extractor
- floor and ceiling
- frame settings
- unvoiced handling
- interpolation or fill strategy, if any

### Masks

Document:

- semantic meaning of `0` and `1`
- source annotations used to derive the mask
- how interval annotations were projected into sample space
- precedence rules if masks can conflict

## Filename examples

- `sub-01_ses-01_task-conversation_run-01_desc-envelope_feature.npy`
- `sub-01_ses-01_task-conversation_run-01_desc-envelope_feature.json`
- `sub-01_ses-01_task-conversation_run-01_desc-f0_feature.npy`
- `sub-01_ses-01_task-conversation_run-01_desc-f0_feature.json`
- `sub-01_ses-01_task-conversation_run-01_desc-speechMask_feature.npy`
- `sub-01_ses-01_task-conversation_run-01_desc-speechMask_feature.json`

## Template status

The JSON examples in this directory are examples and templates for future writers.
They are concrete enough for implementation, but values such as software version, exact file paths, and final run-specific counts should be filled at generation time.
