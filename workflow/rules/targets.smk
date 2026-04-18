conda:
    CONDA_PY_ENV

from pathlib import Path


def _existing_subject_task_runs():
    combos = []
    for subject in SUBJECTS:
        for task in TASKS:
            for run in RUNS_BY_TASK.get(task, RUNS):
                if _is_explicitly_missing(cfg=config, subject=subject, task=task, run=str(run)):
                    continue

                eeg_dir = BIDS_ROOT / subject / "eeg"
                edf = eeg_dir / f"{subject}_task-{task}_run-{run}_eeg.edf"
                channels = eeg_dir / f"{subject}_task-{task}_run-{run}_channels.tsv"
                if edf.exists() and channels.exists():
                    combos.append((subject, task, str(run)))
    return combos


VALID_SUBJECT_TASK_RUNS = _existing_subject_task_runs()


def _continuous_feature_targets(desc: str):
    targets = []
    for subject, task, run in VALID_SUBJECT_TASK_RUNS:
        stem = f"{subject}_task-{task}_run-{run}_desc-{desc}_feature"
        targets.append(derived_path("features", "continuous", f"{stem}.npy"))
        targets.append(derived_path("features", "continuous", f"{stem}.json"))
    return targets


def _event_feature_targets(desc: str):
    targets = []
    for subject, task, run in VALID_SUBJECT_TASK_RUNS:
        stem = f"{subject}_task-{task}_run-{run}_desc-{desc}_features"
        targets.append(derived_path("features", "events", f"{stem}.tsv"))
        targets.append(derived_path("features", "events", f"{stem}.json"))
    return targets


def _annotation_targets(annotation_set: str, pattern: str):
    root = Path(config["paths"]["annotation_root"]) / annotation_set
    return sorted(str(path) for path in root.glob(pattern))

rule preprocessed_all:
    input:
        filter_non_existent(
            expand(
                derived_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )

rule downsample_all:
    input:
        filter_non_existent(
            expand(
                derived_path("eeg", "downsampled", "{subject}_task-{task}_run-{run}_raw_ds.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )

rule reref_all:
    input:
        filter_non_existent(
            expand(
                derived_path(
                    "eeg", "reref", "{subject}_task-{task}_run-{run}_raw_reref.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )

rule ica_all:
    input:
        filter_non_existent(
            expand(
                derived_path("eeg", "ica_applied", "{subject}_task-{task}_run-{run}_raw_ica.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )


rule interpolate_all:
    input:
        filter_non_existent(
            expand(
                derived_path("eeg", "interpolated", "{subject}_task-{task}_run-{run}_raw_interp.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )

rule filter_all:
    input:
        filter_non_existent(
            expand(
                derived_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )

rule metadata_all:
    input:
        filter_non_existent(
            expand(
                derived_path("beh", "metadata", "{subject}_task-{task}_run-{run}_metadata.tsv"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )

rule speech_envelope_all:
    input:
        _continuous_feature_targets("envelope")


rule f0_all:
    input:
        _continuous_feature_targets("f0")


rule f1_f2_all:
    input:
        _event_feature_targets("vowels")


rule phoneme_onsets_all:
    input:
        _annotation_targets("palign_v1", "sub-*_run-*_palign.csv")


rule syllable_onsets_all:
    input:
        _annotation_targets("syllable_v1", "sub-*_run-*_syllable.csv")


rule token_onsets_all:
    input:
        _annotation_targets("tokens_v1", "dyad-*_tokens.csv")


rule epoch_all:
    input:
        filter_non_existent(
            expand(
                derived_path("eeg", "epochs", "{subject}_task-{task}_run-{run}_epochs-epo.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )


rule canary_preprocessing:
    input:
        ds=derived_path("eeg", "downsampled", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_ds.fif"),
        filt=derived_path("eeg", "filtered", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_filt.fif"),
        ica=derived_path("eeg", "ica_applied", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_ica.fif"),
        interp=derived_path("eeg", "interpolated", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_interp.fif"),
        reref=derived_path("eeg", "reref", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_reref.fif"),
        metadata=derived_path("beh", "metadata", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_metadata.tsv"),
        events=derived_path("beh", "metadata", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_events.npy"),
        epochs=derived_path("eeg", "epochs", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_epochs-epo.fif"),
    output:
        done=derived_path("canary", "all.done")
    shell:
        r"""
        mkdir -p "$(dirname {output.done})"
        printf "canary_ok\n" > {output.done}
        """
