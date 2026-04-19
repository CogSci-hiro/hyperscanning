# =============================================================================
#                               Feature extraction rules
# =============================================================================

FEATURE_ROLES = ("self", "other")


def _partner_subject(subject):
    subject_num = int(str(subject).replace("sub-", ""))
    partner_num = subject_num - 1 if subject_num % 2 == 0 else subject_num + 1
    return f"sub-{partner_num:03d}"


def _speaker_for_subject(subject):
    subject_num = int(str(subject).replace("sub-", ""))
    return "A" if (subject_num % 2 == 1) else "B"


def _trf_predictor_spec(predictor_name, subject, task, run):
    acoustic_specs = {
        "speech_envelope": ("continuous", out_path, subject, "envelope", "self_envelope"),
        "envelope": ("continuous", out_path, subject, "envelope", "self_envelope"),
        "self_speech_envelope": ("continuous", out_path, subject, "envelope", "self_envelope"),
        "other_speech_envelope": ("continuous", out_path, subject, "envelope", "other_envelope"),
        "self_envelope": ("continuous", out_path, subject, "envelope", "self_envelope"),
        "other_envelope": ("continuous", out_path, subject, "envelope", "other_envelope"),
        "f0": ("continuous", out_path, subject, "f0", "self_f0"),
        "self_f0": ("continuous", out_path, subject, "f0", "self_f0"),
        "other_f0": ("continuous", out_path, subject, "f0", "other_f0"),
    }
    event_specs = {
        "self_f1_f2": ("event", out_path, subject, "vowels", "self_vowels"),
        "other_f1_f2": ("event", out_path, subject, "vowels", "other_vowels"),
        "self_f1": ("event", out_path, subject, "vowels", "self_vowels"),
        "other_f1": ("event", out_path, subject, "vowels", "other_vowels"),
        "self_f2": ("event", out_path, subject, "vowels", "self_vowels"),
        "other_f2": ("event", out_path, subject, "vowels", "other_vowels"),
        "self_phoneme_onsets": ("event", out_path, subject, "phonemes", "self_phonemes"),
        "other_phoneme_onsets": ("event", out_path, subject, "phonemes", "other_phonemes"),
        "self_syllable_onsets": ("event", out_path, subject, "syllables", "self_syllables"),
        "other_syllable_onsets": ("event", out_path, subject, "syllables", "other_syllables"),
        "self_token_onsets": ("event", out_path, subject, "tokens", "self_tokens"),
        "other_token_onsets": ("event", out_path, subject, "tokens", "other_tokens"),
        "self_surprisal": ("event", lm_feature_path, subject, "lm_surprisal", "lmSurprisal"),
        "other_surprisal": ("event", lm_feature_path, _partner_subject(subject), "lm_surprisal", "lmSurprisal"),
        "self_entropy": ("event", lm_feature_path, subject, "lm_shannon_entropy", "lmShannonEntropy"),
        "other_entropy": ("event", lm_feature_path, _partner_subject(subject), "lm_shannon_entropy", "lmShannonEntropy"),
    }
    if predictor_name in acoustic_specs:
        return acoustic_specs[predictor_name]
    if predictor_name in event_specs:
        return event_specs[predictor_name]
    raise WorkflowError(f"Unsupported TRF predictor {predictor_name!r}")


def _trf_predictor_input_path(predictor_name, subject, task, run):
    storage_kind, root_fn, storage_subject, dirname, descriptor = _trf_predictor_spec(
        predictor_name,
        subject,
        task,
        run,
    )
    if storage_kind == "continuous":
        path = root_fn(
            "features",
            "continuous",
            dirname,
            f"{storage_subject}_task-{task}_run-{run}_desc-{descriptor}_feature.npy",
        )
    elif storage_kind == "event":
        canonical_path = root_fn(
            "features",
            "events",
            dirname,
            f"{storage_subject}_task-{task}_run-{run}_desc-{descriptor}_features.tsv",
        )
        path = canonical_path
        if root_fn is lm_feature_path and not Path(canonical_path).exists():
            pattern = (
                Path(root_fn("features", "events", dirname))
                / f"{storage_subject}_ses-*_task-{task}_run-{run}_desc-{descriptor}_features.tsv"
            )
            matches = sorted(pattern.parent.glob(pattern.name))
            if matches:
                path = str(matches[0])
    else:
        raise WorkflowError(f"Unsupported TRF predictor storage kind {storage_kind!r}")
    return storage_kind, root_fn, path


def _trf_run_is_externally_available(subject, task, run, predictor_names):
    for predictor_name in predictor_names:
        _storage_kind, root_fn, predictor_path = _trf_predictor_input_path(
            predictor_name,
            subject,
            task,
            run,
        )
        if root_fn is lm_feature_path and not Path(predictor_path).exists():
            return False
    return True


def _trf_subject_has_eligible_run(subject, task, predictor_names=None):
    predictor_names = list(predictor_names or TRF.get("predictors", []))
    for run in RUNS_BY_TASK.get(str(task), RUNS):
        run_str = str(run)
        if _is_explicitly_missing(cfg=config, subject=subject, task=task, run=run_str):
            continue
        eeg_dir = BIDS_ROOT / subject / "eeg"
        edf = eeg_dir / f"{subject}_task-{task}_run-{run}_eeg.edf"
        channels = eeg_dir / f"{subject}_task-{task}_run-{run}_channels.tsv"
        if not (edf.exists() and channels.exists()):
            continue
        if _trf_run_is_externally_available(subject, task, run_str, predictor_names):
            return True
    return False


def _trf_qc_required_predictors():
    predictors = list(TRF.get("predictors", []))
    qc_predictors = list(TRF.get("qc_predictors", predictors))
    combined = []
    for predictor_name in [*predictors, *qc_predictors]:
        if predictor_name not in combined:
            combined.append(predictor_name)
    return combined


FEATURES_SIGNATURE = _config_signature({"features": FEATURES})
TRF_SIGNATURE = _config_signature({"trf": TRF}, {"paths": PATHS})
TRF_WORKFLOW_SIGNATURE = _config_signature(
    {"trf": TRF},
    {"paths": PATHS},
    {"runs": config.get("runs", {})},
    {"subjects": config.get("subjects", {})},
    {"debug": config.get("debug", {})},
)

rule speech_envelope:
    input:
        self_audio=bids_path("{subject}", "audio", "{subject}_task-{task}_run-{run}.wav"),
        other_audio=lambda wildcards: bids_path(
            _partner_subject(wildcards.subject),
            "audio",
            f"{_partner_subject(wildcards.subject)}_task-{wildcards.task}_run-{wildcards.run}.wav",
        ),
        raw=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif")
    output:
        self_values=out_path("features", "continuous", "envelope", "{subject}_task-{task}_run-{run}_desc-self_envelope_feature.npy"),
        self_sidecar=out_path("features", "continuous", "envelope", "{subject}_task-{task}_run-{run}_desc-self_envelope_feature.json"),
        other_values=out_path("features", "continuous", "envelope", "{subject}_task-{task}_run-{run}_desc-other_envelope_feature.npy"),
        other_sidecar=out_path("features", "continuous", "envelope", "{subject}_task-{task}_run-{run}_desc-other_envelope_feature.json")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: FEATURES_SIGNATURE,
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} acoustic-envelope \
            --config {params.config_path} \
            --audio {input.self_audio} \
            --raw {input.raw} \
            --feature-name self_speech_envelope \
            --source-subject {wildcards.subject} \
            --source-role self \
            --out {output.self_values} \
            --out-sidecar {output.self_sidecar}
        {HYPER_MODULE_CMD} acoustic-envelope \
            --config {params.config_path} \
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
        raw=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif")
    output:
        self_values=out_path("features", "continuous", "f0", "{subject}_task-{task}_run-{run}_desc-self_f0_feature.npy"),
        self_sidecar=out_path("features", "continuous", "f0", "{subject}_task-{task}_run-{run}_desc-self_f0_feature.json"),
        other_values=out_path("features", "continuous", "f0", "{subject}_task-{task}_run-{run}_desc-other_f0_feature.npy"),
        other_sidecar=out_path("features", "continuous", "f0", "{subject}_task-{task}_run-{run}_desc-other_f0_feature.json")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: FEATURES_SIGNATURE,
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} acoustic-pitch \
            --config {params.config_path} \
            --audio {input.self_audio} \
            --raw {input.raw} \
            --feature-name self_f0 \
            --source-subject {wildcards.subject} \
            --source-role self \
            --out {output.self_values} \
            --out-sidecar {output.self_sidecar}
        {HYPER_MODULE_CMD} acoustic-pitch \
            --config {params.config_path} \
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
        )
    output:
        self_table=out_path("features", "events", "vowels", "{subject}_task-{task}_run-{run}_desc-self_vowels_features.tsv"),
        self_sidecar=out_path("features", "events", "vowels", "{subject}_task-{task}_run-{run}_desc-self_vowels_features.json"),
        other_table=out_path("features", "events", "vowels", "{subject}_task-{task}_run-{run}_desc-other_vowels_features.tsv"),
        other_sidecar=out_path("features", "events", "vowels", "{subject}_task-{task}_run-{run}_desc-other_vowels_features.json")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: FEATURES_SIGNATURE,
        tier="PhonAlign",
        other_subject=lambda wildcards: _partner_subject(wildcards.subject),
        self_speaker=lambda wildcards: _speaker_for_subject(wildcards.subject),
        other_speaker=lambda wildcards: _speaker_for_subject(_partner_subject(wildcards.subject))
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} acoustic-formants \
            --config {params.config_path} \
            --audio {input.self_audio} \
            --alignment {input.self_alignment} \
            --tier {params.tier} \
            --feature-name self_vowels \
            --speaker {params.self_speaker} \
            --source-subject {wildcards.subject} \
            --source-role self \
            --out-tsv {output.self_table} \
            --out-sidecar {output.self_sidecar}
        {HYPER_MODULE_CMD} acoustic-formants \
            --config {params.config_path} \
            --audio {input.other_audio} \
            --alignment {input.other_alignment} \
            --tier {params.tier} \
            --feature-name other_vowels \
            --speaker {params.other_speaker} \
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
        )
    output:
        self_table=out_path("features", "events", "phonemes", "{subject}_task-{task}_run-{run}_desc-self_phonemes_features.tsv"),
        self_sidecar=out_path("features", "events", "phonemes", "{subject}_task-{task}_run-{run}_desc-self_phonemes_features.json"),
        other_table=out_path("features", "events", "phonemes", "{subject}_task-{task}_run-{run}_desc-other_phonemes_features.tsv"),
        other_sidecar=out_path("features", "events", "phonemes", "{subject}_task-{task}_run-{run}_desc-other_phonemes_features.json")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: FEATURES_SIGNATURE,
        tier="PhonAlign",
        self_feature_name="self_phonemes",
        other_feature_name="other_phonemes",
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} alignment-events \
            --config {params.config_path} \
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
            --config {params.config_path} \
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
        )
    output:
        self_table=out_path("features", "events", "syllables", "{subject}_task-{task}_run-{run}_desc-self_syllables_features.tsv"),
        self_sidecar=out_path("features", "events", "syllables", "{subject}_task-{task}_run-{run}_desc-self_syllables_features.json"),
        other_table=out_path("features", "events", "syllables", "{subject}_task-{task}_run-{run}_desc-other_syllables_features.tsv"),
        other_sidecar=out_path("features", "events", "syllables", "{subject}_task-{task}_run-{run}_desc-other_syllables_features.json")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: FEATURES_SIGNATURE,
        tier="SyllAlign",
        self_feature_name="self_syllables",
        other_feature_name="other_syllables",
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} alignment-events \
            --config {params.config_path} \
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
            --config {params.config_path} \
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
        )
    output:
        self_table=out_path("features", "events", "tokens", "{subject}_task-{task}_run-{run}_desc-self_tokens_features.tsv"),
        self_sidecar=out_path("features", "events", "tokens", "{subject}_task-{task}_run-{run}_desc-self_tokens_features.json"),
        other_table=out_path("features", "events", "tokens", "{subject}_task-{task}_run-{run}_desc-other_tokens_features.tsv"),
        other_sidecar=out_path("features", "events", "tokens", "{subject}_task-{task}_run-{run}_desc-other_tokens_features.json")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: FEATURES_SIGNATURE,
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} token-events \
            --config {params.config_path} \
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
            --config {params.config_path} \
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


rule token_pos:
    input:
        tokens=lambda wildcards: annotation_path(
            "tokens_v1",
            f"dyad-{str((int(str(wildcards.subject).replace('sub-', '')) + 1) // 2).zfill(3)}_tokens.csv",
        )
    output:
        self_table=out_path("features", "events", "pos", "{subject}_task-{task}_run-{run}_desc-self_pos_features.tsv"),
        self_sidecar=out_path("features", "events", "pos", "{subject}_task-{task}_run-{run}_desc-self_pos_features.json"),
        other_table=out_path("features", "events", "pos", "{subject}_task-{task}_run-{run}_desc-other_pos_features.tsv"),
        other_sidecar=out_path("features", "events", "pos", "{subject}_task-{task}_run-{run}_desc-other_pos_features.json")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: FEATURES_SIGNATURE,
        other_subject=lambda wildcards: _partner_subject(wildcards.subject)
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} pos-tags \
            --config {params.config_path} \
            --tokens {input.tokens} \
            --subject {wildcards.subject} \
            --run {wildcards.run} \
            --feature-name self_pos \
            --exclude-label '#' \
            --exclude-label '*' \
            --exclude-label '@' \
            --source-subject {wildcards.subject} \
            --source-role self \
            --out-tsv {output.self_table} \
            --out-sidecar {output.self_sidecar}
        {HYPER_MODULE_CMD} pos-tags \
            --config {params.config_path} \
            --tokens {input.tokens} \
            --subject {params.other_subject} \
            --run {wildcards.run} \
            --feature-name other_pos \
            --exclude-label '#' \
            --exclude-label '*' \
            --exclude-label '@' \
            --source-subject {params.other_subject} \
            --source-role other \
            --out-tsv {output.other_table} \
            --out-sidecar {output.other_sidecar}
        """


def _pos_qc_feature_tables():
    inputs = []
    for subject in SUBJECTS:
        for task in TASKS:
            for run in RUNS_BY_TASK.get(task, RUNS):
                run_str = str(run)
                if _is_explicitly_missing(cfg=config, subject=subject, task=task, run=run_str):
                    continue
                for role in FEATURE_ROLES:
                    inputs.append(
                        out_path(
                            "features",
                            "events",
                            "pos",
                            f"{subject}_task-{task}_run-{run_str}_desc-{role}_pos_features.tsv",
                        )
                    )
    return inputs


def _pos_qc_inputs(_wildcards):
    return _pos_qc_feature_tables()


if bool(FEATURES.get("stanza_pos", {}).get("enabled", True)) and len(_pos_qc_feature_tables()) > 0:
    rule pos_qc:
        input:
            _pos_qc_inputs
        output:
            distribution=out_path("qc", "pos", "pos_distribution.png"),
            heatmap=out_path("qc", "pos", "pos_heatmap_by_run.png"),
            problematic=out_path("qc", "pos", "pos_problematic_tokens.png"),
            counts=out_path("qc", "pos", "pos_counts.tsv"),
            proportions=out_path("qc", "pos", "pos_proportions.tsv"),
            proportions_by_run=out_path("qc", "pos", "pos_proportions_by_run.tsv"),
            problematic_by_run=out_path("qc", "pos", "pos_problematic_metrics_by_run.tsv")
        params:
            config_path=str(ACTIVE_CONFIG_PATH),
            config_signature=lambda wildcards: FEATURES_SIGNATURE,
            input_glob=lambda wildcards, input: str(Path(str(input[0])).parent / "*.tsv"),
            out_dir=lambda wildcards, output: str(Path(str(output.distribution)).parent)
        conda:
            CONDA_PY_ENV
        threads: 1
        resources:
            mem_mb=2_000
        shell:
            """
            {HYPER_MODULE_CMD} pos-qc \
                --config {params.config_path} \
                --glob '{params.input_glob}' \
                --out-dir {params.out_dir}
            """


def _trf_subject_inputs(wildcards):
    predictor_names = list(TRF.get("predictors", []))
    run_inputs = []
    for run in RUNS_BY_TASK.get(str(wildcards.task), RUNS):
        run_str = str(run)
        if _is_explicitly_missing(cfg=config, subject=wildcards.subject, task=wildcards.task, run=str(run)):
            continue
        eeg_dir = BIDS_ROOT / wildcards.subject / "eeg"
        edf = eeg_dir / f"{wildcards.subject}_task-{wildcards.task}_run-{run}_eeg.edf"
        channels = eeg_dir / f"{wildcards.subject}_task-{wildcards.task}_run-{run}_channels.tsv"
        if not (edf.exists() and channels.exists()):
            continue
        if not _trf_run_is_externally_available(wildcards.subject, wildcards.task, run_str, predictor_names):
            continue
        run_inputs.append(out_path("eeg", "filtered", f"{wildcards.subject}_task-{wildcards.task}_run-{run}_raw_filt.fif"))
        run_inputs.append(
            out_path("eeg", "downsampled", f"{wildcards.subject}_task-{wildcards.task}_run-{run}_raw_ds_timing.json")
        )
        for predictor_name in predictor_names:
            _storage_kind, _root_fn, predictor_path = _trf_predictor_input_path(
                predictor_name,
                wildcards.subject,
                wildcards.task,
                run_str,
            )
            run_inputs.append(predictor_path)
    return run_inputs


if bool(TRF.get("enabled", False)) and bool(PATHS.get("out_dir", PATHS.get("derived_root"))):
    rule trf:
        input:
            _trf_subject_inputs
        output:
            fold_scores=out_path("trf", "{subject}", "task-{task}", "fold_scores.json"),
            selected_alpha=out_path("trf", "{subject}", "task-{task}", "selected_alpha_per_fold.json"),
            coefficients=out_path("trf", "{subject}", "task-{task}", "coefficients.npz"),
            design_info=out_path("trf", "{subject}", "task-{task}", "design_info.json")
        params:
            config_path=str(ACTIVE_CONFIG_PATH),
            config_signature=lambda wildcards: TRF_WORKFLOW_SIGNATURE
        conda:
            CONDA_PY_ENV
        shell:
            """
            {HYPER_MODULE_CMD} trf \
                --config {params.config_path} \
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
        if not _trf_subject_has_eligible_run(subject, task):
            continue
        inputs.append(out_path("trf", subject, f"task-{task}", "coefficients.npz"))
        inputs.append(out_path("trf", subject, f"task-{task}", "design_info.json"))
    return inputs


def _trf_qc_score_table_inputs(_wildcards):
    inputs = []
    task = "conversation"
    if task not in TASKS:
        return inputs
    predictor_names = _trf_qc_required_predictors()
    for subject in SUBJECTS:
        for run in RUNS_BY_TASK.get(task, RUNS):
            run_str = str(run)
            if _is_explicitly_missing(cfg=config, subject=subject, task=task, run=run_str):
                continue
            eeg_dir = BIDS_ROOT / subject / "eeg"
            edf = eeg_dir / f"{subject}_task-{task}_run-{run}_eeg.edf"
            channels = eeg_dir / f"{subject}_task-{task}_run-{run}_channels.tsv"
            if not (edf.exists() and channels.exists()):
                continue
            inputs.append(out_path("eeg", "filtered", f"{subject}_task-{task}_run-{run}_raw_filt.fif"))
            inputs.append(out_path("eeg", "downsampled", f"{subject}_task-{task}_run-{run}_raw_ds_timing.json"))
            for predictor_name in predictor_names:
                _storage_kind, root_fn, predictor_path = _trf_predictor_input_path(
                    predictor_name,
                    subject,
                    task,
                    run_str,
                )
                if root_fn is lm_feature_path and not Path(predictor_path).exists():
                    continue
                inputs.append(predictor_path)
    return inputs


if bool(TRF.get("enabled", False)) and bool(PATHS.get("out_dir", PATHS.get("derived_root"))):
    rule trf_qc_kernels:
        input:
            _trf_qc_inputs
        output:
            manifest=out_path("figures", "trf_kernels", "manifest.json")
        params:
            config_path=str(ACTIVE_CONFIG_PATH),
            config_signature=lambda wildcards: TRF_SIGNATURE
        conda:
            CONDA_PY_ENV
        threads: 1
        resources:
            mem_mb=4_000
        shell:
            """
            {HYPER_MODULE_CMD} trf-kernel-qc \
                --config {params.config_path} \
                --task conversation \
                --manifest {output.manifest}
            """


if bool(TRF.get("enabled", False)) and bool(PATHS.get("out_dir", PATHS.get("derived_root"))):
    rule trf_qc_alpha_scores:
        input:
            _trf_qc_inputs
        output:
            manifest=out_path("figures", "trf_alpha_scores", "manifest.json")
        params:
            config_path=str(ACTIVE_CONFIG_PATH),
            config_signature=lambda wildcards: TRF_SIGNATURE
        conda:
            CONDA_PY_ENV
        threads: 1
        resources:
            mem_mb=4_000
        shell:
            """
            {HYPER_MODULE_CMD} trf-alpha-qc \
                --config {params.config_path} \
                --task conversation \
                --manifest {output.manifest}
            """


if bool(TRF.get("enabled", False)) and bool(PATHS.get("out_dir", PATHS.get("derived_root"))):
    rule trf_qc_score_tables:
        input:
            _trf_qc_score_table_inputs
        output:
            eeg_scores=out_path("trf_qc", "task-{task}", "eeg_scores.tsv"),
            feature_scores=out_path("trf_qc", "task-{task}", "feature_scores.tsv")
        params:
            config_path=str(ACTIVE_CONFIG_PATH),
            config_signature=lambda wildcards: TRF_WORKFLOW_SIGNATURE
        conda:
            CONDA_PY_ENV
        threads: 1
        resources:
            mem_mb=4_000
        shell:
            """
            {HYPER_MODULE_CMD} trf-score-qc \
                --config {params.config_path} \
                --task {wildcards.task} \
                --eeg-out {output.eeg_scores} \
                --feature-out {output.feature_scores}
            """
