# =============================================================================
#                                   Reports
# =============================================================================

conda:
    CONDA_PY_ENV

rule fooof_qc_figure:
    input:
      filter_non_existent(
        expand(
          derived_path("eeg","downsampled","{subject}","{task}","run-{run}","raw_ds.fif"),
          subject=SUBJECTS,
          task=TASKS,
          run=RUNS,
        )
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
