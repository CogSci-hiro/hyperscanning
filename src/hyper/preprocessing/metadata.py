# ==================================================================================================
#                      Core: conversational metadata (library)
# ==================================================================================================
# > Conversational metadata and event generation.
# > Trial-level metadata were constructed by aligning “self” and “partner” IPU annotation
# > tables within a fixed temporal window (±margin) around each anchor event. For each anchor IPU
# > (onset- or offset-locked), the nearest temporally adjacent partner IPU was selected, partner
# > features (duration, syllable count, rate) were appended, and a latency metric
# > (anchor time minus partner time) was computed. The resulting metadata table was saved as TSV,
# > and event timestamps were converted to an MNE-compatible events array by mapping seconds to
# > sample indices using the recording’s sampling rate and `first_samp`, then saved as a NumPy
# > `.npy` file for epoching.

import re
from pathlib import Path
from typing import Literal

import mne
import numpy as np
import pandas as pd

# ==================================================================================================
# Constants
# ==================================================================================================

DEFAULT_TSV_SEPARATOR: str = "\t"
DEFAULT_IPU_FILENAME_PATTERN: str = r"sub-(\d{3}).*run-(\d+)"

# Backward-compatibility aliases for older call sites.
DEFAULT_SEPARATOR: str = DEFAULT_TSV_SEPARATOR
DEFAULT_IPU_PATTERN: str = DEFAULT_IPU_FILENAME_PATTERN


TimeLock = Literal["onset", "offset"]
Anchor = Literal["self", "other"]

COLUMN_LIST: tuple[str, ...] = ("duration", "n_syllables", "rate")

DEFAULT_TIME_LOCK: TimeLock = "onset"
DEFAULT_ANCHOR: Anchor = "self"
DEFAULT_MARGIN_S: float = 1.0

PLACEHOLDER_ANNOTATION: str = "#"


# ==================================================================================================
# Helpers
# ==================================================================================================

def _validate_time_lock(time_lock: str) -> None:
    if time_lock not in ("onset", "offset"):
        raise ValueError(f"time_lock must be 'onset' or 'offset', got: {time_lock!r}")


def _validate_anchor(anchor: str) -> None:
    if anchor not in ("self", "other"):
        raise ValueError(f"anchor must be 'self' or 'other', got: {anchor!r}")


def _validate_margin(margin_s: float) -> None:
    if not np.isfinite(margin_s) or margin_s <= 0:
        raise ValueError(f"margin_s must be a finite positive number, got: {margin_s!r}")


def _clean_and_prefix_ipu(ipu: pd.DataFrame, *, speaker: Anchor) -> pd.DataFrame:
    """
    Remove placeholder annotations and prefix feature columns with `speaker_`.
    """
    required_cols = {"start", "end", "annotation", *COLUMN_LIST}
    missing = sorted(required_cols.difference(ipu.columns))
    if missing:
        raise ValueError(f"{speaker} IPU table missing required columns: {missing}")

    ipu_clean = ipu.loc[ipu["annotation"] != PLACEHOLDER_ANNOTATION].copy()

    rename_map = {col: f"{speaker}_{col}" for col in COLUMN_LIST}
    ipu_clean = ipu_clean.rename(columns=rename_map)

    return ipu_clean


def _combine_annotations(
    *,
    self_df: pd.DataFrame,
    other_df: pd.DataFrame,
    time_lock: TimeLock,
    anchor: Anchor,
    margin_s: float,
) -> pd.DataFrame:
    """
    Append partner features + latency to the anchor IPU table.
    """
    base_df = self_df if anchor == "self" else other_df
    ref_df = other_df if anchor == "self" else self_df

    origin_col = "start" if time_lock == "onset" else "end"
    target_col = "end" if time_lock == "onset" else "start"

    partner_prefix = "other" if anchor == "self" else "self"
    out_cols = [f"{partner_prefix}_{c}" for c in COLUMN_LIST] + ["latency"]

    def _find_adjacent(row: pd.Series, ref: pd.DataFrame) -> pd.Series:
        origin_t = float(row[origin_col])

        match = ref.loc[(origin_t - margin_s <= ref[target_col]) & (ref[target_col] < origin_t + margin_s)]
        if match.shape[0] == 0:
            return pd.Series([np.nan] * len(out_cols), index=out_cols)

        diffs = (origin_t - match[target_col]).abs()
        best_idx = diffs.idxmin()
        best = match.loc[best_idx]

        values: list[float] = []
        for col in COLUMN_LIST:
            values.append(float(best[f"{partner_prefix}_{col}"]))
        values.append(float(origin_t - float(best[target_col])))

        return pd.Series(values, index=out_cols)

    base_out = base_df.copy()
    base_out.loc[:, out_cols] = base_out.apply(_find_adjacent, axis=1, args=(ref_df,))

    return base_out


def _finalize_combined_metadata(combined: pd.DataFrame, *, time_lock: TimeLock) -> pd.DataFrame:
    """Rename/drop timing columns and compute absolute rate difference."""
    if time_lock == "onset":
        combined = combined.rename(columns={"start": "timestamp"})
        combined = combined.drop(columns=[c for c in ("tier", "end", "annotation") if c in combined.columns])
    else:
        combined = combined.rename(columns={"end": "timestamp"})
        combined = combined.drop(columns=[c for c in ("tier", "start", "annotation") if c in combined.columns])

    # Vectorized absolute difference between speaking rates.
    combined["abs_diff"] = (combined["self_rate"] - combined["other_rate"]).abs()
    return combined


def _load_partner_ipu_tables(
    *,
    self_ipu_csv_path: Path,
    ipu_pattern: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer partner path and load both IPU tables."""
    other_ipu_csv_path = infer_partner_ipu_path(self_ipu_csv_path, pattern=ipu_pattern)
    if not other_ipu_csv_path.exists():  # noqa
        raise FileNotFoundError(f"Partner IPU CSV not found: {other_ipu_csv_path}")

    self_ipu = pd.read_csv(self_ipu_csv_path)
    other_ipu = pd.read_csv(other_ipu_csv_path)
    return self_ipu, other_ipu


def _save_metadata_tsv(df: pd.DataFrame, *, output_tsv_path: Path, sep: str) -> None:
    """Write metadata to TSV."""
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_tsv_path, index=False, sep=sep)


def _build_events_from_raw_and_metadata(
    *,
    raw_fif_path: Path,
    metadata_df: pd.DataFrame,
    event_id: int,
) -> np.ndarray:
    """Create MNE events array by reading sfreq/first_samp from raw."""
    raw = mne.io.read_raw(raw_fif_path, preload=False, verbose="ERROR")
    return metadata_df_to_mne_events(
        metadata_df,
        sfreq_hz=float(raw.info["sfreq"]),
        first_samp=int(raw.first_samp),
        timestamp_col="timestamp",
        event_id=event_id,
    )


def _save_events_npy(events: np.ndarray, output_events_npy_path: Path) -> None:
    """Write events array to NPY."""
    output_events_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_events_npy_path, events)


# ==================================================================================================
# Core logic: partner inference
# ==================================================================================================

def infer_partner_id_and_run_from_ipu_path(
    ipu_path: Path,
    *,
    pattern: str = DEFAULT_IPU_FILENAME_PATTERN,
) -> tuple[str, str]:
    """
    Infer the conversation partner subject ID and run number from an IPU filename.

    This implements your legacy rule:
    - odd subject IDs pair with (id + 1)
    - even subject IDs pair with (id - 1)
    - partner is zero-padded to 3 digits

    Parameters
    ----------
    ipu_path
        Path to the "self" IPU CSV file.
    pattern
        Regex pattern with 2 capture groups:
        - group(1): subject numeric id (3 digits)
        - group(2): run number (one or more digits)

        Default matches e.g.:
          sub-001_run-3_ipu.csv
          sub-001_task-conversation_run-3_ipu.csv

    Returns
    -------
    (other_id, run)
        other_id is zero-padded (e.g. "002"), run is the run string.

    Usage example
    -------------
        other_id, run = infer_partner_id_and_run_from_ipu_path(Path("sub-001_run-3_ipu.csv"))
    """
    match = re.search(pattern, ipu_path.stem)
    if match is None:
        raise RuntimeError(f"IPU filename must match pattern {pattern!r}, got stem={ipu_path.stem!r}")

    self_id_int = int(match.group(1))
    run = str(match.group(2))

    other_id_int = self_id_int - 1 if (self_id_int % 2 == 0) else (self_id_int + 1)
    other_id = str(other_id_int).zfill(3)

    return other_id, run


def infer_partner_ipu_path(
    self_ipu_path: Path,
    *,
    pattern: str = DEFAULT_IPU_FILENAME_PATTERN,
) -> Path:
    """
    Given a self IPU path, infer the partner IPU path in the same directory.

    This follows your old naming convention:
      sub-<ID>_run-<RUN>_ipu.csv

    If your real filenames include task fields etc, the partner inference still
    works (via regex), but the partner filename template here must match what
    you actually write to disk. Adjust if needed.

    Parameters
    ----------
    self_ipu_path
        Path to self IPU CSV.
    pattern
        Regex used to infer partner id and run.

    Returns
    -------
    pathlib.Path
        Partner IPU CSV path (same directory).

    Usage example
    -------------
        other_path = infer_partner_ipu_path(Path(".../sub-001_run-3_ipu.csv"))
    """
    other_id, run = infer_partner_id_and_run_from_ipu_path(self_ipu_path, pattern=pattern)  # noqa
    return self_ipu_path.parent / f"sub-{other_id}_run-{run}_ipu.csv"


# ==================================================================================================
# Core logic: events conversion
# ==================================================================================================

def metadata_df_to_mne_events(
    metadata_df: pd.DataFrame,
    *,
    sfreq_hz: float,
    first_samp: int,
    timestamp_col: str = "timestamp",
    event_id: int = 1,
) -> np.ndarray:
    """
    Convert metadata timestamps to an MNE events array.

    Parameters
    ----------
    metadata_df
        Output of `make_metadata(...)` (must contain `timestamp_col`).
    sfreq_hz
        Sampling frequency in Hz.
    first_samp
        Raw.first_samp value.
    timestamp_col
        Column name containing timestamps in seconds.
    event_id
        Integer event code for all events.

    Returns
    -------
    numpy.ndarray
        Events array with shape (n_events, 3): [sample, 0, event_id].

    Usage example
    -------------
        raw = mne.io.read_raw_fif("raw_filt.fif", preload=False)
        events = metadata_df_to_mne_events(df, sfreq_hz=raw.info["sfreq"], first_samp=raw.first_samp)
    """
    if timestamp_col not in metadata_df.columns:
        raise ValueError(f"metadata_df must contain column {timestamp_col!r}. Found: {list(metadata_df.columns)}")

    timestamps_s = pd.to_numeric(metadata_df[timestamp_col], errors="coerce").to_numpy(dtype=float)
    timestamps_s = timestamps_s[np.isfinite(timestamps_s)]

    samples = np.rint(timestamps_s * float(sfreq_hz)).astype(int) + int(first_samp)

    events = np.zeros((samples.shape[0], 3), dtype=int)
    events[:, 0] = samples
    events[:, 2] = int(event_id)
    return events


# ==================================================================================================
# Core logic: end-to-end file entrypoint (the "heavy lifting")
# ==================================================================================================

def make_metadata_and_events_from_self_ipu(
    *,
    self_ipu_csv_path: Path,
    raw_fif_path: Path | None = None,
    raw_path: Path | None = None,
    output_tsv_path: Path,
    output_events_npy_path: Path,
    config: object,
    margin_s: float = DEFAULT_MARGIN_S,
    time_lock: Literal["onset", "offset"] = DEFAULT_TIME_LOCK,
    anchor: Literal["self", "other"] = DEFAULT_ANCHOR,
    ipu_pattern: str = DEFAULT_IPU_FILENAME_PATTERN,
    sep: str = DEFAULT_TSV_SEPARATOR,
    event_id: int = 1,
) -> None:
    """
    Create metadata TSV + MNE events NPY from a single self IPU CSV.

    This replicates the behavior of your legacy "wrapper-heavy" script:
    - infer partner IPU file from self IPU filename
    - create metadata (time-locked, anchored)
    - save TSV
    - convert timestamps to MNE events array and save NPY

    Parameters
    ----------
    self_ipu_csv_path
        Path to self IPU CSV (used to infer partner + run).
    raw_fif_path
        Preferred path to raw FIF readable by MNE.
    raw_path
        Backward-compatible alias for `raw_fif_path`.
    output_tsv_path
        Output TSV for metadata.
    output_events_npy_path
        Output .npy for events array.
    config
        ProjectConfig-like object (passed for pipeline consistency).
        Not currently used.
    margin_s, time_lock, anchor
        Passed through to `make_metadata(...)`.
    ipu_pattern
        Regex for inferring self id + run from filename.
    sep
        Separator for TSV output.
    event_id
        Event code for all events.

    Returns
    -------
    None

    Usage example
    -------------
        make_metadata_and_events_from_self_ipu(
            self_ipu_csv_path=Path("sub-001_run-3_ipu.csv"),
            raw_fif_path=Path("derived/raw_filt.fif"),
            output_tsv_path=Path("derived/metadata.tsv"),
            output_events_npy_path=Path("derived/events.npy"),
            config=cfg,
            margin_s=1.0,
        )
    """
    _ = config  # reserved for future use
    selected_raw_fif_path = raw_fif_path or raw_path
    if selected_raw_fif_path is None:
        raise ValueError("Either raw_fif_path or raw_path must be provided.")

    self_ipu, other_ipu = _load_partner_ipu_tables(
        self_ipu_csv_path=self_ipu_csv_path,
        ipu_pattern=ipu_pattern,
    )
    df = make_metadata(
        self_ipu=self_ipu,
        other_ipu=other_ipu,
        time_lock=time_lock,
        anchor=anchor,
        margin_s=margin_s,
    )
    _save_metadata_tsv(df, output_tsv_path=output_tsv_path, sep=sep)

    events = _build_events_from_raw_and_metadata(
        raw_fif_path=selected_raw_fif_path,
        metadata_df=df,
        event_id=event_id,
    )
    _save_events_npy(events, output_events_npy_path)


def make_metadata(
    self_ipu: pd.DataFrame,
    other_ipu: pd.DataFrame,
    *,
    time_lock: TimeLock = DEFAULT_TIME_LOCK,
    anchor: Anchor = DEFAULT_ANCHOR,
    margin_s: float = DEFAULT_MARGIN_S,
) -> pd.DataFrame:
    """
    Create trial-level metadata by aligning "self" and "other" IPU annotations.

    The output contains:
    - the anchor speaker's IPU rows (self or other)
    - appended partner features from the temporally adjacent IPU
    - `latency` (anchor origin time minus partner target time)
    - a `timestamp` column derived from onset/offset

    Parameters
    ----------
    self_ipu
        IPU annotation table for the "self" speaker.

        DataFrame format example
        ------------------------
        | start | end | annotation | duration | n_phonemes | n_syllables | rate |
        |------:|----:|------------|---------:|-----------:|------------:|-----:|
        |  1.20 | 2.10| hello      | 0.90     | 10         | 3           | 3.33 |
    other_ipu
        IPU annotation table for the "other" speaker (same expected columns).
    time_lock
        - "onset": anchor on `start`, match partner on `end`, timestamp = `start`
        - "offset": anchor on `end`, match partner on `start`, timestamp = `end`
    anchor
        Which speaker defines the output rows: "self" or "other".
    margin_s
        Matching window (seconds). A partner IPU is considered adjacent if its
        target time is within ±margin_s of the anchor origin time.

    Returns
    -------
    pandas.DataFrame
        Metadata DataFrame with a `timestamp` column and partner-aligned features.

        DataFrame format example
        ------------------------
        | timestamp | self_duration | self_n_syllables | other_rate | latency |
        |----------:|--------------:|-----------------:|-----------:|--------:|
        |     1.20  | 0.90          | 3                | 4.20       | 0.15    |

    Usage example
    -------------
        meta = make_metadata(self_ipu_df, other_ipu_df, time_lock="onset", anchor="self", margin_s=1.0)
    """
    _validate_time_lock(time_lock)
    _validate_anchor(anchor)
    _validate_margin(margin_s)

    self_clean = _clean_and_prefix_ipu(self_ipu, speaker="self")
    other_clean = _clean_and_prefix_ipu(other_ipu, speaker="other")

    combined = _combine_annotations(
        self_df=self_clean,
        other_df=other_clean,
        time_lock=time_lock,
        anchor=anchor,
        margin_s=margin_s,
    )
    return _finalize_combined_metadata(combined, time_lock=time_lock)
