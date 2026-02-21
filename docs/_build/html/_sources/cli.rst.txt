CLI
===

Entry point
-----------

The command-line interface is implemented in ``src/hyper/cli/main.py`` and
exposes subcommands that correspond to preprocessing rules.

Subcommands
-----------

- ``downsample``: raw EDF to ``raw_ds.fif``
- ``reref``: ``raw_ds.fif`` to ``raw_reref.fif``
- ``ica-apply``: ``raw_reref.fif`` + ICA to ``raw_ica.fif``
- ``interpolate``: ``raw_ica.fif`` + channels.tsv to ``raw_interp.fif``
- ``filter``: ``raw_interp.fif`` to ``raw_filt.fif``
- ``metadata``: IPU annotations + ``raw_filt.fif`` to metadata TSV + events NPY
- ``epoch``: ``raw_filt.fif`` + events + metadata to epochs FIF

These commands are invoked by Snakemake rules in
``workflow/rules/preprocessing.smk``.
