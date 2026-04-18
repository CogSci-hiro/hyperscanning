# =============================================================================
#                               Preprocessing rules
# =============================================================================
#
# Preprocessing pipeline summary (per subject × task × run)
# ---------------------------------------------------------
# 1) downsample     : raw EDF → raw_ds.fif (channel typing + EEG-only selection + montage + bads)
# 2) reref          : raw_ds.fif → raw_reref.fif
# 3) apply_ica      : raw_reref.fif + precomputed ICA → raw_ica.fif
# 4) interpolate    : raw_ica.fif + channels.tsv → raw_interp.fif
# 5) filter_raw     : raw_interp.fif → raw_filt.fif (final continuous preprocessed EEG)
#
# Notes
# -----
# - "bids_path(...)" always refers to raw, immutable inputs.
# - "out_path(...)" are regenerable intermediate outputs.
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
        config_path = CONFIG_PATH,
        sfreq=float(config["eeg"]["sfreq_hz"]),
        target_sfreq=float(config["preprocessing"]["downsample"]["sfreq_hz"])
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
        channels=bids_path("{subject}", "eeg", "{subject}_task-{task}_run-{run}_channels.tsv"),
        config="config/config.yaml"
    output:
        raw_reref=maybe_temp(out_path("eeg", "reref", "{subject}_task-{task}_run-{run}_raw_reref.fif"))
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} reref \
            --config {input.config} \
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
        ica=precomputed_ica_path(config["preprocessing"]["ica"]["path_pattern"]),
        config="config/config.yaml"
    output:
        raw_ica=maybe_temp(out_path("eeg", "ica_applied", "{subject}_task-{task}_run-{run}_raw_ica.fif"))
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} ica-apply \
            --config {input.config} \
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
        channels=bids_path("{subject}", "eeg", "{subject}_task-{task}_run-{run}_channels.tsv"),
        config="config/config.yaml"
    output:
        raw_interp=maybe_temp(out_path("eeg", "interpolated", "{subject}_task-{task}_run-{run}_raw_interp.fif"))
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} interpolate \
            --config {input.config} \
            --in-fif {input.raw_ica} \
            --channels {input.channels} \
            --method {config[preprocessing][interpolation][method]} \
            --out {output.raw_interp}
        """


# =============================================================================
# Final band-pass filtering
# =============================================================================

rule filter_raw:
    input:
        raw_interp=out_path("eeg", "interpolated", "{subject}_task-{task}_run-{run}_raw_interp.fif"),
        config="config/config.yaml"
    output:
        raw_filt=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif")
    params:
        l_freq=float(config["preprocessing"]["filter"]["l_freq_hz"]),
        h_freq=float(config["preprocessing"]["filter"]["h_freq_hz"])
    conda:
        CONDA_PY_ENV
    threads: 1
    resources:
        mem_mb=1_000
    shell:
        """
        {HYPER_MODULE_CMD} filter \
            --config {input.config} \
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
        raw=out_path("eeg", "filtered", "{subject}_task-{task}_run-{run}_raw_filt.fif"),
        config="config/config.yaml"
    output:
        tsv=out_path("beh", "metadata", "{subject}_task-{task}_run-{run}_metadata.tsv"),
        events=out_path("beh", "metadata", "{subject}_task-{task}_run-{run}_events.npy")
    params:
        time_lock="onset",
        anchor="self",
        margin=1.0
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} metadata \
            --config {input.config} \
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
        events=out_path("beh", "metadata", "{subject}_task-{task}_run-{run}_events.npy"),
        config="config/config.yaml"
    output:
        epochs=out_path("eeg", "epochs", "{subject}_task-{task}_run-{run}_epochs-epo.fif")
    params:
        tmin=float(config["epochs"]["tmin_s"]),
        tmax=float(config["epochs"]["tmax_s"]),
        detrend=int(config["epochs"]["detrend"]),
        baseline_args=(
            f"--baseline --baseline-start {config['epochs']['baseline_start_s']} "
            f"--baseline-end {config['epochs']['baseline_end_s']}")
    conda:
        CONDA_PY_ENV
    shell:
        """
        {HYPER_MODULE_CMD} epoch \
            --config {input.config} \
            --raw {input.raw} \
            --events {input.events} \
            --metadata {input.metadata} \
            --tmin {params.tmin} \
            --tmax {params.tmax} \
            --detrend {params.detrend} \
            {params.baseline_args} \
            --out {output.epochs}
        """
