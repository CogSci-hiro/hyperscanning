Usage
=====

This pipeline is driven by Snakemake and configured via YAML.

Basic usage
-----------

.. code-block:: bash

   snakemake -s workflow/Snakefile --cores 1

Canary target
-------------

.. code-block:: bash

   snakemake -s workflow/Snakefile derived/canary/all.done --cores 1

TODO
----

- Add data layout examples for BIDS inputs.
- Add a minimal example config file.
