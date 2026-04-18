# =============================================================================
#                               Feature extraction rules
# =============================================================================

FEATURE_ROLES = ("self", "other")


def _partner_subject(subject):
    subject_num = int(str(subject).replace("sub-", ""))
    partner_num = subject_num - 1 if subject_num % 2 == 0 else subject_num + 1
    return f"sub-{partner_num:03d}"


def _descriptor_dirname(descriptor):
    return descriptor.removeprefix("self_").removeprefix("other_")

rule speech_envelope:
    input:
        self_audio=bids_path("{subject}", "audio", "{subject}_task-{task}_run-{run}.wav"),
        other_audio=lambda wildcards: bids_path(
            _partner_subject(wildcards.subject),
            "audio",
            f"{_partner_subject(wildcards.subject)}_task-{wildcards.task}_run-{wildcards.run}.wav",
        ),
        raw=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif"),
        config="config/config.yaml"
    output:
        self_values=out_path("features", "continuous", "envelope", "{subject}_task-{task}_run-{run}_desc-self_envelope_feature.npy"),
        self_sidecar=out_path("features", "continuous", "envelope", "{subject}_task-{task}_run-{run}_desc-self_envelope_feature.json"),
        other_values=out_path("features", "continuous", "envelope", "{subject}_task-{task}_run-{run}_desc-other_envelope_feature.npy"),
        other_sidecar=out_path("features", "continuous", "envelope", "{subject}_task-{task}_run-{run}_desc-other_envelope_feature.json")
    params:
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} acoustic-envelope \
            --config {input.config} \
            --audio {input.self_audio} \
            --raw {input.raw} \
            --feature-name self_speech_envelope \
            --source-subject {wildcards.subject} \
            --source-role self \
            --out {output.self_values} \
            --out-sidecar {output.self_sidecar}
        {HYPER_MODULE_CMD} acoustic-envelope \
            --config {input.config} \
            --audio {input.other_audio} \
            --raw {input.raw} \
            --feature-name other_speech_envelope \
            --source-subject {params.other_subject} \
            --source-role other \
            --out {output.other_values} \
            --out-sidecar {output.other_sidecar}
        """


rule f0:
    input:
        self_audio=bids_path("{subject}", "audio", "{subject}_task-{task}_run-{run}.wav"),
        other_audio=lambda wildcards: bids_path(
            _partner_subject(wildcards.subject),
            "audio",
            f"{_partner_subject(wildcards.subject)}_task-{wildcards.task}_run-{wildcards.run}.wav",
        ),
        raw=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif"),
        config="config/config.yaml"
    output:
        self_values=out_path("features", "continuous", "f0", "{subject}_task-{task}_run-{run}_desc-self_f0_feature.npy"),
        self_sidecar=out_path("features", "continuous", "f0", "{subject}_task-{task}_run-{run}_desc-self_f0_feature.json"),
        other_values=out_path("features", "continuous", "f0", "{subject}_task-{task}_run-{run}_desc-other_f0_feature.npy"),
        other_sidecar=out_path("features", "continuous", "f0", "{subject}_task-{task}_run-{run}_desc-other_f0_feature.json")
    params:
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} acoustic-pitch \
            --config {input.config} \
            --audio {input.self_audio} \
            --raw {input.raw} \
            --feature-name self_f0 \
            --source-subject {wildcards.subject} \
            --source-role self \
            --out {output.self_values} \
            --out-sidecar {output.self_sidecar}
        {HYPER_MODULE_CMD} acoustic-pitch \
            --config {input.config} \
            --audio {input.other_audio} \
            --raw {input.raw} \
            --feature-name other_f0 \
            --source-subject {params.other_subject} \
            --source-role other \
            --out {output.other_values} \
            --out-sidecar {output.other_sidecar}
        """


rule f1_f2:
    input:
        self_audio=bids_path("{subject}", "audio", "{subject}_task-{task}_run-{run}.wav"),
        other_audio=lambda wildcards: bids_path(
            _partner_subject(wildcards.subject),
            "audio",
            f"{_partner_subject(wildcards.subject)}_task-{wildcards.task}_run-{wildcards.run}.wav",
        ),
        self_alignment=annotation_path("palign_v1", "{subject}_run-{run}_palign.csv"),
        other_alignment=lambda wildcards: annotation_path(
            "palign_v1", f"{_partner_subject(wildcards.subject)}_run-{wildcards.run}_palign.csv"
        ),
        config="config/config.yaml"
    output:
        self_table=out_path("features", "events", "vowels", "{subject}_task-{task}_run-{run}_desc-self_vowels_features.tsv"),
        self_sidecar=out_path("features", "events", "vowels", "{subject}_task-{task}_run-{run}_desc-self_vowels_features.json"),
        other_table=out_path("features", "events", "vowels", "{subject}_task-{task}_run-{run}_desc-other_vowels_features.tsv"),
        other_sidecar=out_path("features", "events", "vowels", "{subject}_task-{task}_run-{run}_desc-other_vowels_features.json")
    params:
        tier="PhonAlign",
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} acoustic-formants \
            --config {input.config} \
            --audio {input.self_audio} \
            --alignment {input.self_alignment} \
            --tier {params.tier} \
            --feature-name self_vowels \
            --source-subject {wildcards.subject} \
            --source-role self \
            --out-tsv {output.self_table} \
            --out-sidecar {output.self_sidecar}
        {HYPER_MODULE_CMD} acoustic-formants \
            --config {input.config} \
            --audio {input.other_audio} \
            --alignment {input.other_alignment} \
            --tier {params.tier} \
            --feature-name other_vowels \
            --source-subject {params.other_subject} \
            --source-role other \
            --out-tsv {output.other_table} \
            --out-sidecar {output.other_sidecar}
        """


rule phoneme_onsets:
    input:
        self_alignment=annotation_path("palign_v1", "{subject}_run-{run}_palign.csv"),
        other_alignment=lambda wildcards: annotation_path(
            "palign_v1", f"{_partner_subject(wildcards.subject)}_run-{wildcards.run}_palign.csv"
        ),
        config="config/config.yaml"
    output:
        self_table=out_path("features", "events", "phonemes", "{subject}_task-{task}_run-{run}_desc-self_phonemes_features.tsv"),
        self_sidecar=out_path("features", "events", "phonemes", "{subject}_task-{task}_run-{run}_desc-self_phonemes_features.json"),
        other_table=out_path("features", "events", "phonemes", "{subject}_task-{task}_run-{run}_desc-other_phonemes_features.tsv"),
        other_sidecar=out_path("features", "events", "phonemes", "{subject}_task-{task}_run-{run}_desc-other_phonemes_features.json")
    params:
        tier="PhonAlign",
        self_feature_name="self_phonemes",
        other_feature_name="other_phonemes",
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} alignment-events \
            --config {input.config} \
            --alignment {input.self_alignment} \
            --tier {params.tier} \
            --feature-name {params.self_feature_name} \
            --exclude-label '#' \
            --exclude-label 'noise' \
            --exclude-label 'fp' \
            --source-subject {wildcards.subject} \
            --source-role self \
            --out-tsv {output.self_table} \
            --out-sidecar {output.self_sidecar}
        {HYPER_MODULE_CMD} alignment-events \
            --config {input.config} \
            --alignment {input.other_alignment} \
            --tier {params.tier} \
            --feature-name {params.other_feature_name} \
            --exclude-label '#' \
            --exclude-label 'noise' \
            --exclude-label 'fp' \
            --source-subject {params.other_subject} \
            --source-role other \
            --out-tsv {output.other_table} \
            --out-sidecar {output.other_sidecar}
        """


rule syllable_onsets:
    input:
        self_alignment=annotation_path("syllable_v1", "{subject}_run-{run}_syllable.csv"),
        other_alignment=lambda wildcards: annotation_path(
            "syllable_v1", f"{_partner_subject(wildcards.subject)}_run-{wildcards.run}_syllable.csv"
        ),
        config="config/config.yaml"
    output:
        self_table=out_path("features", "events", "syllables", "{subject}_task-{task}_run-{run}_desc-self_syllables_features.tsv"),
        self_sidecar=out_path("features", "events", "syllables", "{subject}_task-{task}_run-{run}_desc-self_syllables_features.json"),
        other_table=out_path("features", "events", "syllables", "{subject}_task-{task}_run-{run}_desc-other_syllables_features.tsv"),
        other_sidecar=out_path("features", "events", "syllables", "{subject}_task-{task}_run-{run}_desc-other_syllables_features.json")
    params:
        tier="SyllAlign",
        self_feature_name="self_syllables",
        other_feature_name="other_syllables",
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} alignment-events \
            --config {input.config} \
            --alignment {input.self_alignment} \
            --tier {params.tier} \
            --feature-name {params.self_feature_name} \
            --exclude-label '#' \
            --exclude-label 'noise' \
            --exclude-label 'fp' \
            --source-subject {wildcards.subject} \
            --source-role self \
            --out-tsv {output.self_table} \
            --out-sidecar {output.self_sidecar}
        {HYPER_MODULE_CMD} alignment-events \
            --config {input.config} \
            --alignment {input.other_alignment} \
            --tier {params.tier} \
            --feature-name {params.other_feature_name} \
            --exclude-label '#' \
            --exclude-label 'noise' \
            --exclude-label 'fp' \
            --source-subject {params.other_subject} \
            --source-role other \
            --out-tsv {output.other_table} \
            --out-sidecar {output.other_sidecar}
        """


rule token_onsets:
    input:
        tokens=lambda wildcards: annotation_path(
            "tokens_v1",
            f"dyad-{str((int(str(wildcards.subject).replace('sub-', '')) + 1) // 2).zfill(3)}_tokens.csv",
        ),
        config="config/config.yaml"
    output:
        self_table=out_path("features", "events", "tokens", "{subject}_task-{task}_run-{run}_desc-self_tokens_features.tsv"),
        self_sidecar=out_path("features", "events", "tokens", "{subject}_task-{task}_run-{run}_desc-self_tokens_features.json"),
        other_table=out_path("features", "events", "tokens", "{subject}_task-{task}_run-{run}_desc-other_tokens_features.tsv"),
        other_sidecar=out_path("features", "events", "tokens", "{subject}_task-{task}_run-{run}_desc-other_tokens_features.json")
    params:
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} token-events \
            --config {input.config} \
            --tokens {input.tokens} \
            --subject {wildcards.subject} \
            --run {wildcards.run} \
            --feature-name self_tokens \
            --exclude-label '#' \
            --exclude-label '*' \
            --exclude-label '@' \
            --source-subject {wildcards.subject} \
            --source-role self \
            --out-tsv {output.self_table} \
            --out-sidecar {output.self_sidecar}
        {HYPER_MODULE_CMD} token-events \
            --config {input.config} \
            --tokens {input.tokens} \
            --subject {params.other_subject} \
            --run {wildcards.run} \
            --feature-name other_tokens \
            --exclude-label '#' \
            --exclude-label '*' \
            --exclude-label '@' \
            --source-subject {params.other_subject} \
            --source-role other \
            --out-tsv {output.other_table} \
            --out-sidecar {output.other_sidecar}
        """


def _trf_subject_inputs(wildcards):
    predictor_map = {
        "speech_envelope": "self_envelope",
        "envelope": "self_envelope",
        "self_speech_envelope": "self_envelope",
        "other_speech_envelope": "other_envelope",
        "self_envelope": "self_envelope",
        "other_envelope": "other_envelope",
        "f0": "self_f0",
        "self_f0": "self_f0",
        "other_f0": "other_f0",
    }
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
                    _descriptor_dirname(descriptor),
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


def _trf_qc_inputs(_wildcards):
    inputs = []
    task = "conversation"
    if task not in TASKS:
        return inputs
    for subject in SUBJECTS:
        inputs.append(out_path("trf", subject, f"task-{task}", "coefficients.npz"))
        inputs.append(out_path("trf", subject, f"task-{task}", "design_info.json"))
    return inputs


if bool(config.get("trf", {}).get("enabled", False)) and bool(config.get("paths", {}).get("out_dir", config.get("paths", {}).get("derived_root"))):
    rule trf_qc_kernels:
        input:
            _trf_qc_inputs
        output:
            manifest=out_path("figures", "trf_kernels", "manifest.json")
        conda:
            CONDA_PY_ENV
        threads: 1
        resources:
            mem_mb=4_000
        shell:
            """
            {HYPER_MODULE_CMD} trf-kernel-qc \
                --config config/config.yaml \
                --task conversation \
                --manifest {output.manifest}
            """
