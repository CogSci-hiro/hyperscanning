# =============================================================================
#                                   Reports
# =============================================================================

conda:
    CONDA_PY_ENV


def _report_task_runs(task: str):
    combos = []
    for subject in SUBJECTS:
        for run in RUNS_BY_TASK.get(task, RUNS):
            if _is_explicitly_missing(cfg=config, subject=subject, task=task, run=str(run)):
                continue
            eeg_dir = BIDS_ROOT / subject / "eeg"
            edf = eeg_dir / f"{subject}_task-{task}_run-{run}_eeg.edf"
            channels = eeg_dir / f"{subject}_task-{task}_run-{run}_channels.tsv"
            if edf.exists() and channels.exists():
                combos.append((subject, task, str(run)))
    return combos


def _speech_artefact_inputs(kind: str):
    task = str(VIZ.get("speech_artefact", {}).get("task", "conversation"))
    triples = _report_task_runs(task)
    inputs = []
    for subject, task_name, run in triples:
        stem = f"{subject}_task-{task_name}_run-{run}"
        if kind == "filtered_noica":
            inputs.append(out_path("eeg", "interpolated_noica", f"{stem}_raw_interp_noica.fif"))
        elif kind == "filtered":
            inputs.append(out_path("eeg", "interpolated", f"{stem}_raw_interp.fif"))
        elif kind == "ica":
            pattern = str(PREPROCESSING.get("ica", {}).get("path_pattern", "{subject_id}_task-{task}-ica.fif"))
            resolved = precomputed_ica_path(pattern).format(subject=subject, task=task_name, run=run)
            if resolved not in inputs:
                inputs.append(resolved)
        else:
            raise ValueError(f"Unsupported speech artefact input kind: {kind!r}")
    return inputs


def _speech_artefact_cli_args(flag: str, values):
    return " ".join(f"{flag} {shlex.quote(str(value))}" for value in values)


def _trf_score_task() -> str:
    return str(VIZ.get("trf_score", {}).get("task", "conversation"))


def _trf_main_figure_filename() -> str:
    return str(VIZ.get("trf_main_figure", {}).get("filename", "trf_main_figure_summary.png"))


def _ipu_turn_taking_inputs():
    annotation_dir = Path(PATHS["annotation_root"]) / str(config["annotations"]["ipu"])
    return sorted(str(path) for path in annotation_dir.glob("sub-*_run-*_ipu.csv"))

rule fooof_qc_figure:
    input:
      filter_non_existent(
        expand(
          out_path("eeg","downsampled","{subject}","{task}","run-{run}","raw_ds.fif"),
          subject=SUBJECTS,
          task=TASKS,
          run=RUNS,
        ),
        bids_root=BIDS_ROOT,
        cfg=config,
      )
    output:
        fig=reports_path("figures", "fooof_qc.png")
    conda:
        "workflow/envs/python.yaml"
    threads: 1
    resources:
        mem_mb=2_000
    shell:
        r"""
        conv fooof-qc \
          --config {configfile} \
          --in-table {input.table} \
          --out-fig {output.fig}
        """


rule speech_artefact_figure:
    input:
        filtered_noica=_speech_artefact_inputs("filtered_noica"),
        filtered=_speech_artefact_inputs("filtered"),
        ica=_speech_artefact_inputs("ica")
    output:
        fig=reports_path("figures", str(VIZ.get("speech_artefact", {}).get("filename", "speech_artefact_summary.png")))
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        filtered_noica_args=lambda wildcards, input: _speech_artefact_cli_args("--filtered-noica", input.filtered_noica),
        filtered_args=lambda wildcards, input: _speech_artefact_cli_args("--filtered", input.filtered),
        ica_args=lambda wildcards, input: _speech_artefact_cli_args("--ica", input.ica)
    conda:
        CONDA_PY_ENV
    shell:
        r"""
        {HYPER_MODULE_CMD} speech-artefact-qc \
          --config {params.config_path} \
          {params.filtered_noica_args} \
          {params.filtered_args} \
          {params.ica_args} \
          --out-fig {output.fig}
        """


rule trf_score_qc_figure:
    input:
        eeg_table=out_path("trf_qc", f"task-{_trf_score_task()}", "eeg_scores.tsv"),
        feature_table=out_path("trf_qc", f"task-{_trf_score_task()}", "feature_scores.tsv")
    output:
        fig=reports_path("figures", str(VIZ.get("trf_score", {}).get("filename", "trf_score_summary.png")))
    params:
        config_path=str(ACTIVE_CONFIG_PATH)
    conda:
        CONDA_PY_ENV
    shell:
        r"""
        {HYPER_MODULE_CMD} trf-score-qc-figure \
          --config {params.config_path} \
          --eeg-table {input.eeg_table} \
          --feature-table {input.feature_table} \
          --out-fig {output.fig}
        """


rule trf_main_figure:
    input:
        _trf_qc_inputs
    output:
        fig=reports_path("figures", _trf_main_figure_filename())
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: TRF_SIGNATURE
    conda:
        CONDA_PY_ENV
    threads: 1
    resources:
        mem_mb=4_000
    shell:
        r"""
        {HYPER_MODULE_CMD} trf-main-figure \
          --config {params.config_path} \
          --out-fig {output.fig}
        """


rule ipu_turn_taking_figure:
    input:
        ipu_csvs=_ipu_turn_taking_inputs()
    output:
        fig=reports_path("figures", str(VIZ.get("ipu_turn_taking", {}).get("filename", "ipu_turn_taking_summary.png")))
    params:
        config_path=str(ACTIVE_CONFIG_PATH)
    conda:
        CONDA_PY_ENV
    shell:
        r"""
        {HYPER_MODULE_CMD} ipu-turn-taking-figure \
          --config {params.config_path} \
          --out-fig {output.fig}
        """
