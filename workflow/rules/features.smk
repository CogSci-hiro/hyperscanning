# =============================================================================
#                               Feature extraction rules
# =============================================================================

rule speech_envelope:
    input:
        audio=bids_path("{subject}", "audio", "{subject}_task-{task}_run-{run}.wav"),
        raw=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif"),
        config="config/config.yaml"
    output:
        values=out_path("features", "continuous", "envelope", "{subject}_task-{task}_run-{run}_desc-envelope_feature.npy"),
        sidecar=out_path("features", "continuous", "envelope", "{subject}_task-{task}_run-{run}_desc-envelope_feature.json")
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} acoustic-envelope \
            --config {input.config} \
            --audio {input.audio} \
            --raw {input.raw} \
            --out {output.values} \
            --out-sidecar {output.sidecar}
        """


rule f0:
    input:
        audio=bids_path("{subject}", "audio", "{subject}_task-{task}_run-{run}.wav"),
        raw=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif"),
        config="config/config.yaml"
    output:
        values=out_path("features", "continuous", "f0", "{subject}_task-{task}_run-{run}_desc-f0_feature.npy"),
        sidecar=out_path("features", "continuous", "f0", "{subject}_task-{task}_run-{run}_desc-f0_feature.json")
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} acoustic-pitch \
            --config {input.config} \
            --audio {input.audio} \
            --raw {input.raw} \
            --out {output.values} \
            --out-sidecar {output.sidecar}
        """


rule f1_f2:
    input:
        audio=bids_path("{subject}", "audio", "{subject}_task-{task}_run-{run}.wav"),
        alignment=annotation_path("palign_v1", "{subject}_run-{run}_palign.csv"),
        config="config/config.yaml"
    output:
        table=out_path("features", "events", "vowels", "{subject}_task-{task}_run-{run}_desc-vowels_features.tsv"),
        sidecar=out_path("features", "events", "vowels", "{subject}_task-{task}_run-{run}_desc-vowels_features.json")
    params:
        tier="PhonAlign"
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} acoustic-formants \
            --config {input.config} \
            --audio {input.audio} \
            --alignment {input.alignment} \
            --tier {params.tier} \
            --out-tsv {output.table} \
            --out-sidecar {output.sidecar}
        """


rule phoneme_onsets:
    input:
        alignment=annotation_path("palign_v1", "{subject}_run-{run}_palign.csv"),
        config="config/config.yaml"
    output:
        table=out_path("features", "events", "phonemes", "{subject}_task-{task}_run-{run}_desc-phonemes_features.tsv"),
        sidecar=out_path("features", "events", "phonemes", "{subject}_task-{task}_run-{run}_desc-phonemes_features.json")
    params:
        tier="PhonAlign",
        feature_name="phonemes"
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} alignment-events \
            --config {input.config} \
            --alignment {input.alignment} \
            --tier {params.tier} \
            --feature-name {params.feature_name} \
            --exclude-label '#' \
            --exclude-label 'noise' \
            --exclude-label 'fp' \
            --out-tsv {output.table} \
            --out-sidecar {output.sidecar}
        """


rule syllable_onsets:
    input:
        alignment=annotation_path("syllable_v1", "{subject}_run-{run}_syllable.csv"),
        config="config/config.yaml"
    output:
        table=out_path("features", "events", "syllables", "{subject}_task-{task}_run-{run}_desc-syllables_features.tsv"),
        sidecar=out_path("features", "events", "syllables", "{subject}_task-{task}_run-{run}_desc-syllables_features.json")
    params:
        tier="SyllAlign",
        feature_name="syllables"
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} alignment-events \
            --config {input.config} \
            --alignment {input.alignment} \
            --tier {params.tier} \
            --feature-name {params.feature_name} \
            --exclude-label '#' \
            --exclude-label 'noise' \
            --exclude-label 'fp' \
            --out-tsv {output.table} \
            --out-sidecar {output.sidecar}
        """


rule token_onsets:
    input:
        tokens=lambda wildcards: annotation_path(
            "tokens_v1",
            f"dyad-{str((int(str(wildcards.subject).replace('sub-', '')) + 1) // 2).zfill(3)}_tokens.csv",
        ),
        config="config/config.yaml"
    output:
        table=out_path("features", "events", "tokens", "{subject}_task-{task}_run-{run}_desc-tokens_features.tsv"),
        sidecar=out_path("features", "events", "tokens", "{subject}_task-{task}_run-{run}_desc-tokens_features.json")
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} token-events \
            --config {input.config} \
            --tokens {input.tokens} \
            --subject {wildcards.subject} \
            --run {wildcards.run} \
            --exclude-label '#' \
            --exclude-label '*' \
            --exclude-label '@' \
            --out-tsv {output.table} \
            --out-sidecar {output.sidecar}
        """


def _trf_subject_inputs(wildcards):
    predictor_map = {"speech_envelope": "envelope", "envelope": "envelope", "f0": "f0"}
    predictor_names = list(config.get("trf", {}).get("predictors", []))
    descriptors = [predictor_map[name] for name in predictor_names]
    run_inputs = ["config/config.yaml"]
    for run in RUNS_BY_TASK.get(str(wildcards.task), RUNS):
        if _is_explicitly_missing(cfg=config, subject=wildcards.subject, task=wildcards.task, run=str(run)):
            continue
        eeg_dir = BIDS_ROOT / wildcards.subject / "eeg"
        edf = eeg_dir / f"{wildcards.subject}_task-{wildcards.task}_run-{run}_eeg.edf"
        channels = eeg_dir / f"{wildcards.subject}_task-{wildcards.task}_run-{run}_channels.tsv"
        if not (edf.exists() and channels.exists()):
            continue
        run_inputs.append(out_path("eeg", "filtered", f"{wildcards.subject}_task-{wildcards.task}_run-{run}_raw_filt.fif"))
        for descriptor in descriptors:
            run_inputs.append(
                out_path(
                    "features",
                    "continuous",
                    descriptor,
                    f"{wildcards.subject}_task-{wildcards.task}_run-{run}_desc-{descriptor}_feature.npy",
                )
            )
    return run_inputs


if bool(config.get("trf", {}).get("enabled", False)) and bool(config.get("paths", {}).get("out_dir", config.get("paths", {}).get("derived_root"))):
    rule trf:
        input:
            _trf_subject_inputs
        output:
            fold_scores=out_path("trf", "{subject}", "task-{task}", "fold_scores.json"),
            selected_alpha=out_path("trf", "{subject}", "task-{task}", "selected_alpha_per_fold.json"),
            coefficients=out_path("trf", "{subject}", "task-{task}", "coefficients.npz"),
            design_info=out_path("trf", "{subject}", "task-{task}", "design_info.json")
        conda:
            CONDA_PY_ENV
        shell:
            """
            {HYPER_MODULE_CMD} trf \
                --config config/config.yaml \
                --subject {wildcards.subject} \
                --task {wildcards.task} \
                --out-dir $(dirname {output.fold_scores})
            """
