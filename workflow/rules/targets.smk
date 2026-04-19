conda:
    CONDA_PY_ENV


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
FEATURE_ROLES = ("self", "other")


def _role_descriptor(desc: str, role: str):
    return f"{role}_{desc}"


def _continuous_feature_targets(desc: str):
    targets = []
    for subject, task, run in VALID_SUBJECT_TASK_RUNS:
        for role in FEATURE_ROLES:
            role_desc = _role_descriptor(desc, role)
            stem = f"{subject}_task-{task}_run-{run}_desc-{role_desc}_feature"
            targets.append(out_path("features", "continuous", desc, f"{stem}.npy"))
            targets.append(out_path("features", "continuous", desc, f"{stem}.json"))
    return targets


def _event_feature_targets(desc: str):
    targets = []
    for subject, task, run in VALID_SUBJECT_TASK_RUNS:
        for role in FEATURE_ROLES:
            role_desc = _role_descriptor(desc, role)
            stem = f"{subject}_task-{task}_run-{run}_desc-{role_desc}_features"
            targets.append(out_path("features", "events", desc, f"{stem}.tsv"))
            targets.append(out_path("features", "events", desc, f"{stem}.json"))
    return targets


def _trf_targets():
    if not bool(PATHS.get("out_dir", PATHS.get("derived_root"))):
        return []
    targets = []
    for subject in SUBJECTS:
        task = "conversation"
        if task not in TASKS:
            continue
        if not _trf_subject_has_eligible_run(subject, task):
            continue
        targets.append(out_path("trf", subject, f"task-{task}", "fold_scores.json"))
        targets.append(out_path("trf", subject, f"task-{task}", "selected_alpha_per_fold.json"))
        targets.append(out_path("trf", subject, f"task-{task}", "coefficients.npz"))
        targets.append(out_path("trf", subject, f"task-{task}", "design_info.json"))
    return targets


def _trf_qc_score_table_targets():
    task = "conversation"
    if task not in TASKS:
        return []
    return [
        out_path("trf_qc", f"task-{task}", "eeg_scores.tsv"),
        out_path("trf_qc", f"task-{task}", "feature_scores.tsv"),
    ]

rule preprocessed_all:
    input:
        filter_non_existent(
            expand(
                out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif"),
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
                out_path("eeg", "downsampled", "{subject}_task-{task}_run-{run}_raw_ds.fif"),
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
                out_path(
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
                out_path("eeg", "ica_applied", "{subject}_task-{task}_run-{run}_raw_ica.fif"),
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
                out_path("eeg", "interpolated", "{subject}_task-{task}_run-{run}_raw_interp.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )

rule interpolate_noica_all:
    input:
        filter_non_existent(
            expand(
                out_path("eeg", "interpolated_noica", "{subject}_task-{task}_run-{run}_raw_interp_noica.fif"),
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
                out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )

rule filter_noica_all:
    input:
        filter_non_existent(
            expand(
                out_path("eeg", "filtered_noica", "{subject}_task-{task}_run-{run}_raw_filt_noica.fif"),
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
                out_path("beh", "metadata", "{subject}_task-{task}_run-{run}_metadata.tsv"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )

rule metadata_noica_all:
    input:
        filter_non_existent(
            expand(
                out_path("beh", "metadata_noica", "{subject}_task-{task}_run-{run}_metadata_noica.tsv"),
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
        _event_feature_targets("phonemes")


rule syllable_onsets_all:
    input:
        _event_feature_targets("syllables")


rule token_onsets_all:
    input:
        _event_feature_targets("tokens")


if bool(FEATURES.get("stanza_pos", {}).get("enabled", True)):
    rule token_pos_all:
        input:
            _event_feature_targets("pos")


if bool(FEATURES.get("stanza_pos", {}).get("enabled", True)) and len(_event_feature_targets("pos")) > 0:
    rule qc_pos_all:
        input:
            out_path("qc", "pos", "pos_distribution.png"),
            out_path("qc", "pos", "pos_heatmap_by_run.png"),
            out_path("qc", "pos", "pos_problematic_tokens.png"),
            out_path("qc", "pos", "pos_counts.tsv"),
            out_path("qc", "pos", "pos_proportions.tsv"),
            out_path("qc", "pos", "pos_proportions_by_run.tsv"),
            out_path("qc", "pos", "pos_problematic_metrics_by_run.tsv")


rule epoch_all:
    input:
        filter_non_existent(
            expand(
                out_path("eeg", "epochs", "{subject}_task-{task}_run-{run}_epochs-epo.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )

rule epoch_noica_all:
    input:
        filter_non_existent(
            expand(
                out_path("eeg", "epochs_noica", "{subject}_task-{task}_run-{run}_epochs_noica-epo.fif"),
                subject=SUBJECTS,
                task=TASKS,
                run=RUNS,
            ),
          bids_root=BIDS_ROOT,
          cfg=config
        )


rule main_figures_all:
    input:
        reports_path("figures", str(VIZ.get("speech_artefact", {}).get("filename", "speech_artefact_summary.png"))),
        reports_path("figures", str(VIZ.get("trf_score", {}).get("filename", "trf_score_summary.png")))


if bool(TRF.get("enabled", False)) and bool(PATHS.get("out_dir", PATHS.get("derived_root"))):
    rule trf_all:
        input:
            _trf_targets()


if bool(TRF.get("enabled", False)) and bool(PATHS.get("out_dir", PATHS.get("derived_root"))):
    rule qc_trf_kernels_all:
        input:
            out_path("figures", "trf_kernels", "manifest.json")


if bool(TRF.get("enabled", False)) and bool(PATHS.get("out_dir", PATHS.get("derived_root"))):
    rule qc_trf_alpha_scores_all:
        input:
            out_path("figures", "trf_alpha_scores", "manifest.json")


if bool(TRF.get("enabled", False)) and bool(PATHS.get("out_dir", PATHS.get("derived_root"))):
    rule qc_trf_score_tables_all:
        input:
            _trf_qc_score_table_targets()


rule canary_preprocessing:
    input:
        ds_timing=out_path("eeg", "downsampled", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_ds_timing.json"),
        filt=out_path("eeg", "filtered", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_filt.fif"),
        metadata=out_path("beh", "metadata", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_metadata.tsv"),
        events=out_path("beh", "metadata", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_events.npy"),
        epochs=out_path("eeg", "epochs", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_epochs-epo.fif"),
    output:
        done=out_path("canary", "all.done")
    shell:
        r"""
        mkdir -p "$(dirname {output.done})"
        printf "canary_ok\n" > {output.done}
        """


rule canary_preprocessing_noica:
    input:
        ds_timing=out_path("eeg", "downsampled", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_ds_timing.json"),
        filt=out_path("eeg", "filtered_noica", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_raw_filt_noica.fif"),
        metadata=out_path("beh", "metadata_noica", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_metadata_noica.tsv"),
        events=out_path("beh", "metadata_noica", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_events_noica.npy"),
        epochs=out_path("eeg", "epochs_noica", f"{CANARY_SUBJECT}_task-{CANARY_TASK}_run-{CANARY_RUN}_epochs_noica-epo.fif"),
    output:
        done=out_path("canary", "all_noica.done")
    shell:
        r"""
        mkdir -p "$(dirname {output.done})"
        printf "canary_noica_ok\n" > {output.done}
        """
