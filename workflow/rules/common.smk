# =============================================================================
#                               Shared helpers
# =============================================================================
#
# This file contains small, pure helper functions used across multiple
# Snakemake rule files.
#
# Design principles
# -----------------
# - No side effects: functions should not modify global state.
# - Deterministic: given the same config and filesystem, they return the same
#   results every time.
# - Centralized paths: all path construction lives here so the on-disk layout
#   can be changed in exactly one place later.
#

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from snakemake.exceptions import WorkflowError

conda:
    CONDA_PY_ENV


# =============================================================================
#                           Subject discovery
# =============================================================================

def discover_subjects(cfg: Dict) -> List[str]:
    """
    Discover available BIDS subjects on disk and apply exclusion rules.

    This function defines the *universe of subjects* for the Snakemake DAG.
    Instead of hardcoding subject IDs in the Snakefile, we:
      1) Scan the BIDS root directory for subject folders
      2) Filter out any subjects explicitly listed in the config

    This keeps the pipeline declarative:
    - The filesystem defines what exists
    - The config defines what is excluded

    Parameters
    ----------
    cfg
        Parsed project configuration dictionary. Must contain:
        - cfg["subjects"]["bids_root"] : str or Path
            Root directory of the BIDS dataset (contains sub-* folders)
        - cfg["subjects"]["pattern"] : str, optional
            Glob pattern for subject folders (default: "sub-*")
        - cfg["subjects"]["exclude"] : list[str], optional
            Subject IDs to exclude from the run

    Returns
    -------
    list[str]
        Sorted list of subject IDs that will be included in the pipeline.

    Usage example
    -------------
        subjects = discover_subjects(config)
        print(subjects)
        # ['sub-01', 'sub-02', 'sub-04']
    """
    # Root directory where BIDS subject folders live
    root = Path(cfg["paths"]["bids_root"])

    # Glob pattern for subject discovery (e.g., "sub-*")
    # Defaults to BIDS convention if not explicitly set in config
    pattern = cfg["subjects"].get("pattern", "sub-*")

    # Set of subject IDs to exclude (faster lookup than list)
    exclude = set(cfg["subjects"].get("exclude", []))

    # Discover subject directories matching the pattern
    # We explicitly check `is_dir()` to avoid picking up stray files
    discovered = sorted(
        p.name
        for p in root.glob(pattern)
        if p.is_dir()
    )

    # Filter out excluded subjects
    return [s for s in discovered if s not in exclude]


TargetList = Union[str, Sequence[str]]


# Tries to extract (subject, task, run) from a target path that contains
# "...sub-XXX_task-YYY_run-ZZZ..."
_TARGET_KEYS_RE = re.compile(r"(sub-\d+).*?_task-([A-Za-z0-9]+).*?_run-(\d+)")


def _is_explicitly_missing(
    *,
    cfg: Mapping,
    subject: str,
    task: str,
    run: str,
) -> bool:
    """
    Return True if (subject, task, run) is explicitly marked as missing in config.
    """
    missing_runs_cfg = cfg.get("subjects", {}).get("missing_runs", {})
    subject_cfg = missing_runs_cfg.get(subject, {})
    missing_for_task = set(str(r) for r in subject_cfg.get(task, []))
    return str(run) in missing_for_task


def filter_non_existent(
  targets: TargetList,
  bids_root: Path,
  cfg: Mapping,
  task_hint: Optional[str] = None,
) -> TargetList:
  """
  Filter expanded targets by keeping only those whose BIDS source inputs exist,
  *and* removing any (subject, task, run) combos explicitly marked as missing.

  This lets you expand over (subjects × tasks × runs) without hard-coding
  special cases for subjects/tasks with fewer runs.

  Parameters
  ----------
  targets
      Either a list of target paths (strings) or a single path string.
  bids_root
      Root of the BIDS dataset (directory containing sub-*/).
  cfg
      Global config dictionary. Uses:
      cfg["subjects"]["missing_runs"][<subject>][<task>] -> list of run strings/ints
  task_hint
      Optional task label to use if the target path does not include `_task-...`.
      If both are available, the parsed task from the target wins.

  Returns
  -------
  list[str] | str
      Filtered list of targets (or the input string unchanged).

  Usage example
  -------------
      targets = expand("derived/{subject}_task-{task}_run-{run}_epo.fif", ...)
      targets = filter_non_existent(targets, bids_root=BIDS_ROOT, cfg=config)
  """
  if isinstance(targets,str):
    return targets

  final_list: List[str] = []

  for target in targets:
    m = _TARGET_KEYS_RE.search(target)
    if m is None:
      # If we cannot parse keys, keep the target so it fails loudly later.
      final_list.append(target)
      continue

    subject, parsed_task, run = m.group(1), m.group(2), m.group(3)
    task = parsed_task or (task_hint or "")

    # 1) Explicit missing runs override everything.
    if task and _is_explicitly_missing(cfg=cfg,subject=subject,task=task,run=run):
      continue

    # 2) Existence check.
    eeg_dir = bids_root / subject / "eeg"
    edf = eeg_dir / f"{subject}_task-{task}_run-{run}_eeg.edf"
    ch = eeg_dir / f"{subject}_task-{task}_run-{run}_channels.tsv"

    if edf.exists() and ch.exists():
      final_list.append(target)

  return final_list

# =============================================================================
#                           Path construction
# =============================================================================
#
# These helpers centralize how paths are built throughout the workflow.
# This avoids hardcoding directory layouts inside individual rules.
#
# If the project layout changes (e.g., moving from "results/" to "outputs/"),
# you only need to update the config.yaml and these functions.
#

def derived_path(*parts: str) -> str:
    """
    Construct a path under the derived data root.

    Parameters
    ----------
    *parts
        Path components relative to the derived data root.

    Returns
    -------
    str
        Full path as a string (Snakemake prefers string paths).

    Usage example
    -------------
        derived_path("epochs", "sub-01", "run-01", "epochs_epo.fif")
        # "data/derived/epochs/sub-01/run-01/epochs_epo.fif"
    """
    return str(Path(config["paths"]["derived_root"]) / Path(*parts))


def results_path(*parts: str) -> str:
    """
    Construct a path under the results root.

    This is intended for statistical outputs and final analysis artifacts
    (e.g., HDF5 cluster test results, summary tables).

    Parameters
    ----------
    *parts
        Path components relative to the results root.

    Returns
    -------
    str
        Full path as a string.

    Usage example
    -------------
        results_path("fooof", "exponent_results.hdf5")
        # "results/fooof/exponent_results.hdf5"
    """
    return str(Path(config["paths"]["results_root"]) / Path(*parts))


def reports_path(*parts: str) -> str:
    """
    Construct a path under the reports root.

    This is intended for human-facing outputs:
    - figures
    - tables
    - manuscript files

    Parameters
    ----------
    *parts
        Path components relative to the reports root.

    Returns
    -------
    str
        Full path as a string.

    Usage example
    -------------
        reports_path("figures", "fooof_qc.png")
        # "reports/figures/fooof_qc.png"
    """
    return str(Path(config["paths"]["reports_root"]) / Path(*parts))


def annotation_path(*parts: str) -> str:
  """
  Construct a path under the annotation root.

  Parameters
  ----------
  *parts
      Path components relative to the annotation root.

  Returns
  -------
  str
      Full path as a string.

  Usage example
  -------------
      annotation_path("ipu_v1", "sub-001_run-1_ipu.csv")
      annotations/ipu_v1/sub-001_run-1_ipu.csv
  """
  return str(Path(config["paths"]["annotation_root"]) / Path(*parts))



def bids_path(*parts: str) -> str:
    """
    Construct a path under the BIDS root.

    This should be used for all raw data access to keep a clean separation
    between:
    - raw, immutable inputs (BIDS)
    - derived, regenerable outputs (derived/)
    - final results (results/, reports/)

    Parameters
    ----------
    *parts
        Path components relative to the BIDS root.

    Returns
    -------
    str
        Full path as a string.

    Usage example
    -------------
        bids_path("sub-01", "eeg", "sub-01_task-conversation_run-01_eeg.edf")
        # "data/bids/sub-01/eeg/sub-01_task-conversation_run-01_eeg.edf"
    """
    return str(Path(config["paths"]["bids_root"]) / Path(*parts))


def check_bids_inputs(
  bids_root: Path,
  subjects: Sequence[str],
  tasks: Sequence[str],
  runs_by_task: Mapping[str, Sequence[str]],
  max_show: int = 10,
) -> None:
  from snakemake.exceptions import WorkflowError

  missing_examples = []
  missing_entire_task = []

  for subject in subjects:
    for task in tasks:
      eeg_dir = bids_root / subject / "eeg"
      runs = runs_by_task.get(task,[])

      found_any = False
      for run in runs:
        edf = eeg_dir / f"{subject}_task-{task}_run-{run}_eeg.edf"
        ch = eeg_dir / f"{subject}_task-{task}_run-{run}_channels.tsv"
        if edf.exists() and ch.exists():
          found_any = True
          break

      if not found_any:
        missing_entire_task.append(f"{subject} task={task}")
        if len(missing_entire_task) <= max_show:
          missing_examples.append(
            f"{subject} task={task} -> no runs found "
            f"(expected one of {list(runs)})"
          )

  if missing_entire_task:
    raise WorkflowError(
      "No valid runs found for some subject/task combinations:\n"
      "  - " + "\n  - ".join(missing_examples)
    )
