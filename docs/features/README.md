# Feature Derivative Templates

This directory defines the publication-facing template system for planned feature derivatives.
The files here are documentation and example metadata assets, not finalized derivative outputs.

## Scope

- Continuous features are samplewise arrays aligned exactly to EEG sample count and shared as `.npy` plus `.json`.
- Event-like features are sparse tabular annotations anchored in time and shared as `.tsv` plus `.json`.
- Vowel median formant features (`F1` median and `F2` median) are documented as event-level features by default.
- POS tags are represented in the word-level event tables rather than as a redundant standalone derivative.
- If language-model or other high-dimensional embeddings are shared later, timing and token mapping should live in the TSV table while vectors should live in separate arrays rather than inside TSV columns.

## Layout

- [continuous/features_continuous.md](/Users/hiro/Projects/active/hyperscanning/docs/features/continuous/features_continuous.md)
  Shared rules for continuous derivatives.
- [continuous/continuous_feature_template.json](/Users/hiro/Projects/active/hyperscanning/docs/features/continuous/continuous_feature_template.json)
  Copyable JSON sidecar template for any continuous derivative.
- [events/features_events.md](/Users/hiro/Projects/active/hyperscanning/docs/features/events/features_events.md)
  Shared rules for event-like derivatives.
- [events/event_feature_template.json](/Users/hiro/Projects/active/hyperscanning/docs/features/events/event_feature_template.json)
  Copyable JSON sidecar template for any event-like derivative.

## Included feature-specific examples

Continuous examples:

- envelope
- F0
- listening-only mask
- speech mask
- silence mask
- overlap mask

Event-like examples:

- word features, including POS and language-model token mapping fields
- phoneme features
- vowel features with median `F1` and `F2`
- IPU features
- turn-transition features

## Naming conventions

- Continuous values: `sub-<id>_ses-<id>_task-<task>_run-<id>_desc-<feature>_feature.npy`
- Continuous sidecar: `sub-<id>_ses-<id>_task-<task>_run-<id>_desc-<feature>_feature.json`
- Event table: `sub-<id>_ses-<id>_task-<task>_run-<id>_desc-<feature>_features.tsv`
- Event sidecar: `sub-<id>_ses-<id>_task-<task>_run-<id>_desc-<feature>_features.json`

## Implementation intent

These templates are designed so that future extraction code can emit valid derivatives without inventing output structure at write time. If a later implementation needs additional fields, those additions should preserve the shared field names and timing conventions documented here.
