# =============================================================================
#                               Preprocessing rules
# =============================================================================
#
# Preprocessing pipeline summary (per subject × task × run)
# ---------------------------------------------------------
# 1) downsample     : raw EDF → raw_ds.fif (channel typing + EEG-only selection + montage + bads)
# 2) reref          : raw_ds.fif → raw_reref.fif
# 3) apply_ica      : raw_reref.fif + precomputed ICA → raw_ica.fif
# 4) interpolate    : raw_ica.fif + channels.tsv → raw_interp.fif (persistent pre-bandpass endpoint)
# 5) no-ICA branch  : raw_reref.fif + channels.tsv → raw_interp_noica.fif (persistent pre-bandpass endpoint)
# 6) filter_raw     : raw_interp.fif → raw_filt.fif (additional band-pass step for TRF/downstream modeling)
# 7) filter_noica   : raw_interp_noica.fif → raw_filt_noica.fif
#
# Notes
# -----
# - "bids_path(...)" always refers to raw, immutable inputs.
# - Pre-bandpass interpolated outputs are kept on disk because reports/QC depend on them directly.
# - Downsampling is placed first to reduce compute/memory for all downstream steps.
#

rule downsample:
    input:
        eeg_edf=bids_path("{subject}", "eeg", "{subject}_task-{task}_run-{run}_eeg.edf"),
        channels_tsv=bids_path("{subject}", "eeg", "{subject}_task-{task}_run-{run}_channels.tsv")
    output:
        out_fif=maybe_temp(out_path("eeg", "downsampled", "{subject}_task-{task}_run-{run}_raw_ds.fif")),
        timing_json=out_path("eeg", "downsampled", "{subject}_task-{task}_run-{run}_raw_ds_timing.json")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature(
            {"eeg": config.get("eeg", {})},
            {"preprocessing": {
                "downsample": PREPROCESSING.get("downsample", {}),
                "montage": PREPROCESSING.get("montage", {}),
            }},
        ),
        sfreq=float(config["eeg"]["sfreq_hz"]),
        target_sfreq=float(PREPROCESSING["downsample"]["sfreq_hz"])
    conda:
        CONDA_PY_ENV
    threads: 1
    resources:
        mem_mb=2_000
    shell:
        r"""
        {HYPER_MODULE_CMD} downsample \
          --config {params.config_path} \
          --in-edf {input.eeg_edf} \
          --channels {input.channels_tsv} \
          --sfreq {params.sfreq} \
          --target-sfreq {params.target_sfreq} \
          --out {output.out_fif}
        """


rule reref:
    input:
        raw_ds=out_path("eeg", "downsampled", "{subject}_task-{task}_run-{run}_raw_ds.fif"),
        channels=bids_path("{subject}", "eeg", "{subject}_task-{task}_run-{run}_channels.tsv")
    output:
        raw_reref=maybe_temp(out_path("eeg", "reref", "{subject}_task-{task}_run-{run}_raw_reref.fif"))
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature(
            {"preprocessing": {"rereference": PREPROCESSING.get("rereference", {})}},
        )
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} reref \
            --config {params.config_path} \
            --in-fif {input.raw_ds} \
            --channels {input.channels} \
            --out {output.raw_reref}
        """


# =============================================================================
# ICA application (apply precomputed ICA)
# =============================================================================

rule ica_apply:
    input:
        raw_reref=out_path("eeg", "reref", "{subject}_task-{task}_run-{run}_raw_reref.fif"),
        ica=precomputed_ica_path(PREPROCESSING["ica"]["path_pattern"])
    output:
        raw_ica=maybe_temp(out_path("eeg", "ica_applied", "{subject}_task-{task}_run-{run}_raw_ica.fif"))
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature(
            {"preprocessing": {"ica": PREPROCESSING.get("ica", {})}},
        )
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} ica-apply \
            --config {params.config_path} \
            --in-fif {input.raw_reref} \
            --ica {input.ica} \
            --out {output.raw_ica}
        """


# =============================================================================
# Interpolation
# =============================================================================

rule interpolate:
    input:
        raw_ica=out_path("eeg", "ica_applied", "{subject}_task-{task}_run-{run}_raw_ica.fif"),
        channels=bids_path("{subject}", "eeg", "{subject}_task-{task}_run-{run}_channels.tsv")
    output:
        raw_interp=out_path("eeg", "interpolated", "{subject}_task-{task}_run-{run}_raw_interp.fif")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature(
            {"preprocessing": {"interpolation": PREPROCESSING.get("interpolation", {})}},
        ),
        method=str(PREPROCESSING["interpolation"]["method"])
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} interpolate \
            --config {params.config_path} \
            --in-fif {input.raw_ica} \
            --channels {input.channels} \
            --method {params.method} \
            --out {output.raw_interp}
        """


rule interpolate_noica:
    input:
        raw_reref=out_path("eeg", "reref", "{subject}_task-{task}_run-{run}_raw_reref.fif"),
        channels=bids_path("{subject}", "eeg", "{subject}_task-{task}_run-{run}_channels.tsv")
    output:
        raw_interp=out_path("eeg", "interpolated_noica", "{subject}_task-{task}_run-{run}_raw_interp_noica.fif")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature(
            {"preprocessing": {"interpolation": PREPROCESSING.get("interpolation", {})}},
        ),
        method=str(PREPROCESSING["interpolation"]["method"])
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} interpolate \
            --config {params.config_path} \
            --in-fif {input.raw_reref} \
            --channels {input.channels} \
            --method {params.method} \
            --out {output.raw_interp}
        """


# =============================================================================
# Additional band-pass filtering for TRF/downstream modeling
# =============================================================================

rule filter_raw:
    input:
        raw_interp=out_path("eeg", "interpolated", "{subject}_task-{task}_run-{run}_raw_interp.fif")
    output:
        raw_filt=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature(
            {"preprocessing": {"filter": PREPROCESSING.get("filter", {})}},
        ),
        l_freq=float(PREPROCESSING["filter"]["l_freq_hz"]),
        h_freq=float(PREPROCESSING["filter"]["h_freq_hz"])
    conda:
        CONDA_PY_ENV
    threads: 1
    resources:
        mem_mb=1_000
    shell:
        """
        {HYPER_MODULE_CMD} filter \
            --config {params.config_path} \
            --in-fif {input.raw_interp} \
            --l-freq {params.l_freq} \
            --h-freq {params.h_freq} \
            --out {output.raw_filt}
        """


rule filter_raw_noica:
    input:
        raw_interp=out_path("eeg", "interpolated_noica", "{subject}_task-{task}_run-{run}_raw_interp_noica.fif")
    output:
        raw_filt=out_path("eeg", "filtered_noica", "{subject}_task-{task}_run-{run}_raw_filt_noica.fif")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature(
            {"preprocessing": {"filter": PREPROCESSING.get("filter", {})}},
        ),
        l_freq=float(PREPROCESSING["filter"]["l_freq_hz"]),
        h_freq=float(PREPROCESSING["filter"]["h_freq_hz"])
    conda:
        CONDA_PY_ENV
    threads: 1
    resources:
        mem_mb=1_000
    shell:
        """
        {HYPER_MODULE_CMD} filter \
            --config {params.config_path} \
            --in-fif {input.raw_interp} \
            --l-freq {params.l_freq} \
            --h-freq {params.h_freq} \
            --out {output.raw_filt}
        """

# =============================================================================
# Metadata
# =============================================================================

rule metadata:
    input:
        ipu=annotation_path(config["annotations"]["ipu"], "{subject}_run-{run}_ipu.csv"),
        raw=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif")
    output:
        tsv=out_path("beh", "metadata", "{subject}_task-{task}_run-{run}_metadata.tsv"),
        events=out_path("beh", "metadata", "{subject}_task-{task}_run-{run}_events.npy")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature(
            {"annotations": config.get("annotations", {})},
        ),
        time_lock="onset",
        anchor="self",
        margin=1.0
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} metadata \
            --config {params.config_path} \
            --ipu {input.ipu} \
            --raw {input.raw} \
            --time-lock {params.time_lock} \
            --anchor {params.anchor} \
            --margin {params.margin} \
            --out-tsv {output.tsv} \
            --out-events {output.events}
        """


rule metadata_noica:
    input:
        ipu=annotation_path(config["annotations"]["ipu"], "{subject}_run-{run}_ipu.csv"),
        raw=out_path("eeg", "filtered_noica", "{subject}_task-{task}_run-{run}_raw_filt_noica.fif")
    output:
        tsv=out_path("beh", "metadata_noica", "{subject}_task-{task}_run-{run}_metadata_noica.tsv"),
        events=out_path("beh", "metadata_noica", "{subject}_task-{task}_run-{run}_events_noica.npy")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature(
            {"annotations": config.get("annotations", {})},
        ),
        time_lock="onset",
        anchor="self",
        margin=1.0
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} metadata \
            --config {params.config_path} \
            --ipu {input.ipu} \
            --raw {input.raw} \
            --time-lock {params.time_lock} \
            --anchor {params.anchor} \
            --margin {params.margin} \
            --out-tsv {output.tsv} \
            --out-events {output.events}
        """


# =============================================================================
# Epoching
# =============================================================================

rule epoch:
    input:
        raw=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif"),
        metadata=out_path("beh", "metadata", "{subject}_task-{task}_run-{run}_metadata.tsv"),
        events=out_path("beh", "metadata", "{subject}_task-{task}_run-{run}_events.npy")
    output:
        epochs=out_path("eeg", "epochs", "{subject}_task-{task}_run-{run}_epochs-epo.fif")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature({"epochs": EPOCH_SETTINGS}),
        tmin=float(EPOCH_SETTINGS["tmin_s"]),
        tmax=float(EPOCH_SETTINGS["tmax_s"]),
        detrend=int(EPOCH_SETTINGS["detrend"]),
        baseline_args=(
            f"--baseline --baseline-start {EPOCH_SETTINGS['baseline_start_s']} "
            f"--baseline-end {EPOCH_SETTINGS['baseline_end_s']}")
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} epoch \
            --config {params.config_path} \
            --raw {input.raw} \
            --events {input.events} \
            --metadata {input.metadata} \
            --tmin {params.tmin} \
            --tmax {params.tmax} \
            --detrend {params.detrend} \
            {params.baseline_args} \
            --out {output.epochs}
        """


rule epoch_noica:
    input:
        raw=out_path("eeg", "filtered_noica", "{subject}_task-{task}_run-{run}_raw_filt_noica.fif"),
        metadata=out_path("beh", "metadata_noica", "{subject}_task-{task}_run-{run}_metadata_noica.tsv"),
        events=out_path("beh", "metadata_noica", "{subject}_task-{task}_run-{run}_events_noica.npy")
    output:
        epochs=out_path("eeg", "epochs_noica", "{subject}_task-{task}_run-{run}_epochs_noica-epo.fif")
    params:
        config_path=str(ACTIVE_CONFIG_PATH),
        config_signature=lambda wildcards: _config_signature({"epochs": EPOCH_SETTINGS}),
        tmin=float(EPOCH_SETTINGS["tmin_s"]),
        tmax=float(EPOCH_SETTINGS["tmax_s"]),
        detrend=int(EPOCH_SETTINGS["detrend"]),
        baseline_args=(
            f"--baseline --baseline-start {EPOCH_SETTINGS['baseline_start_s']} "
            f"--baseline-end {EPOCH_SETTINGS['baseline_end_s']}")
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} epoch \
            --config {params.config_path} \
            --raw {input.raw} \
            --events {input.events} \
            --metadata {input.metadata} \
            --tmin {params.tmin} \
            --tmax {params.tmax} \
            --detrend {params.detrend} \
            {params.baseline_args} \
            --out {output.epochs}
        """
