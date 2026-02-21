[TITLE]
=======

Scientific Snakemake pipeline for [FIELD]. This documentation describes the
data flow, rule dependencies, and APIs used by the analysis workflow.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   workflow
   cli
   api

Reproducibility
---------------

- The Snakemake DAG encodes all file dependencies declared in the rules.
- Outputs are deterministic given the same inputs, configuration, and software
  environment.
- Use the canary target for an inexpensive end-to-end integrity check.
