# Continuous features

This document defines the storage, metadata, and documentation conventions for continuous features shared with the dataset.

## Scope

Continuous features are time-resolved numeric signals that are aligned samplewise to the EEG time base and stored as:

* `*.npy` for the numeric array
* `*.json` for the sidecar metadata

These features are intended to be directly reusable in TRF, encoding, decoding, and related time-series analyses.

## File-level convention

Each continuous feature is stored as a one-dimensional NumPy array unless explicitly noted otherwise.

### Standard representation

* **Data file**: `*_feature.npy`
* **Metadata file**: `*_feature.json`
* **Time axis**: samplewise, aligned to EEG sample space
* **Shape**: typically `[n_samples]`
* **SamplingFrequency**: equal to the EEG sampling rate for the target run
* **AlignmentTarget**: always `EEG`
* **StartTime**: `0.0` s relative to run start, unless otherwise documented
* **Missing values**: encoded as `NaN`

## Core principles

### Exact EEG alignment

All continuous features must match the EEG sample count exactly:

[
N_{\mathrm{feature}} = N_{\mathrm{EEG}}
]

where:

* (N_{\mathrm{feature}}) is the final number of samples in the feature array
* (N_{\mathrm{EEG}}) is the number of EEG samples in the corresponding run

If a feature is initially extracted at another sampling frequency, it must be resampled and then checked against the EEG length.

### Explicit provenance

Each sidecar must document:

* source input files
* source sampling frequency
* extraction software and version
* extraction algorithm
* preprocessing steps
* postprocessing steps
* resampling and alignment strategy
* feature-specific parameters

### Reproducibility

Any parameter that could change the numerical output must be recorded in the sidecar JSON. This includes extraction settings, filtering, interpolation, and missing-data handling.

## Standard metadata fields

All continuous-feature JSON sidecars should contain the following top-level sections:

* `FeatureName`
* `FeatureType`
* `Description`
* `Units`
* `Shape`
* `DType`
* `SamplingFrequency`
* `NumSamples`
* `AlignmentTarget`
* `StartTime`
* `MissingValueEncoding`
* `Source`
* `Generation`
* `Method`
* `QualityControl`
* `Notes`

## Required numerical metadata

The following fields should always be populated:

* `Shape`
* `DType`
* `SamplingFrequency`
* `NumSamples`
* `MissingValueEncoding`

For one-dimensional time series:

[
\mathrm{Shape} = [N]
]

with (N = N_{\mathrm{EEG}}).

## Recommended quality control fields

Each continuous-feature sidecar should report:

* finite value ratio
* number of `NaN` values
* number of infinite values
* optional summary statistics

Example:

[
\mathrm{FiniteValueRatio} = \frac{N_{\mathrm{finite}}}{N_{\mathrm{samples}}}
]

where (N_{\mathrm{finite}}) is the number of finite values in the final array.

## Units

Units must be explicit whenever possible.

### Recommended unit conventions

* `Hz` for F0
* `0/1` or `boolean` for masks
* `arbitrary` only when no physically interpretable unit exists
* `zscore` only if normalization has already been applied and documented

## Common processing fields

The `Method` block should distinguish between:

* **Algorithm**: what was extracted
* **Parameters**: feature-specific settings
* **Preprocessing**: operations before extraction
* **Postprocessing**: operations after extraction
* **Resampling**: how the feature was brought to EEG sample space
* **Alignment**: how exact length matching was achieved

## Missing-data conventions

The sidecar must document how missing or undefined values are represented.

### Examples

* unvoiced F0 segments encoded as `NaN`
* mask arrays encoded as `0` and `1`
* extraction failures retained as `NaN` rather than silently imputed

If interpolation is used for model-ready regressors, this must be clearly documented in `Method.Postprocessing` and `Method.Parameters`.

## Feature-specific notes

### Envelope

Envelope features should record:

* the extraction method or variant used
* filtering/compression details if applicable
* whether the implementation directly follows a published method or an approximation

### F0

F0 features should record:

* tracker used
* floor and ceiling settings
* handling of unvoiced regions
* whether the shared array is raw, interpolated, or both

### Binary masks

Mask features should record:

* semantic interpretation of `0` and `1`
* annotation source
* projection rule from events/intervals to samplewise representation

## Recommended documentation pattern

For each feature family:

1. Store the numeric data in `*.npy`
2. Store the metadata in `*.json`
3. Document shared conventions in this file
4. Add a dedicated feature-specific markdown file when the method needs extra detail or may be scrutinized by reviewers

## Validation checklist

Before release, each continuous feature should pass the following checks:

* data file exists
* sidecar file exists
* sample count matches EEG exactly
* sampling frequency is documented
* units are documented
* extraction parameters are documented
* missing-value handling is documented
* QC values are present

## Suggested filename pattern

Examples:

* `sub-01_ses-01_task-conversation_run-01_desc-f0_feature.npy`
* `sub-01_ses-01_task-conversation_run-01_desc-f0_feature.json`
* `sub-01_ses-01_task-conversation_run-01_desc-oganianEnv_feature.npy`
* `sub-01_ses-01_task-conversation_run-01_desc-oganianEnv_feature.json`

## Summary

Continuous features are shared as sample-aligned NumPy arrays with JSON sidecars that fully document provenance, processing, alignment, and quality control. The central requirement is exact compatibility with the EEG time base together with explicit, reproducible metadata.
