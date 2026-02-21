conda:
    CONDA_PY_ENV

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
        reref=derived_path("eeg", "reref", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_reref.fif"),
        ica=derived_path("eeg", "ica_applied", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_ica.fif"),
        interp=derived_path("eeg", "interpolated", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_interp.fif"),
        filt=derived_path("eeg", "filtered", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_filt.fif"),
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
