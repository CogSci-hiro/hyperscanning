################################################################################
#                          CONVERT IPU CSV TO BIDS-LIKE ANNOTATIONS             #
################################################################################

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


################################################################################
#                                   CONSTANTS                                    #
################################################################################

TASK = "conversation"
DESCRIPTION = "ipu"
IPU_SILENCE_THRESHOLD_SECONDS = 0.200

INPUT_FILENAME_PATTERN = re.compile(
    r"^(?P<subject>sub-\d+)_run-(?P<run>\d+)_ipu\.csv$"
)

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "onset": ("onset", "start", "start_time", "tmin", "xmin", "begin"),
    "offset": ("offset", "end", "end_time", "tmax", "xmax", "stop"),
    "duration": ("duration", "dur"),
    "speaker": ("speaker", "spk", "participant", "talker"),
    "label": ("label", "text", "annotation", "ipu", "value"),
}


################################################################################
#                              FILE DISCOVERY                                    #
################################################################################

def is_hidden_path(path: Path) -> bool:
    """
    Check whether a path contains any hidden component.

    Parameters
    ----------
    path
        File or directory path.

    Returns
    -------
    bool
        True if any path component starts with '.', otherwise False.

    Usage example
    -------------
    ```python
    is_hidden = is_hidden_path(Path(".DS_Store"))
    ```
    """
    return any(part.startswith(".") for part in path.parts)


def iter_ipu_csv_files(input_directory: Path) -> list[Path]:
    """
    Find all non-hidden IPU CSV files matching the expected naming convention.

    Parameters
    ----------
    input_directory
        Directory containing source IPU CSV files.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of matching IPU CSV files.

    Usage example
    -------------
    ```python
    csv_files = iter_ipu_csv_files(Path("annotations/ipu_v1"))
    ```
    """
    csv_files: list[Path] = []

    for csv_path in input_directory.rglob("*.csv"):
        relative_path = csv_path.relative_to(input_directory)

        if is_hidden_path(relative_path):
            continue

        if INPUT_FILENAME_PATTERN.match(csv_path.name) is None:
            continue

        csv_files.append(csv_path)

    return sorted(csv_files)


def parse_subject_and_run(csv_path: Path) -> tuple[str, str]:
    """
    Parse subject and run labels from an IPU CSV filename.

    Parameters
    ----------
    csv_path
        Source CSV path.

    Returns
    -------
    tuple[str, str]
        Subject label and zero-padded run label.

    Raises
    ------
    ValueError
        If the filename does not match the expected convention.

    Usage example
    -------------
    ```python
    subject, run = parse_subject_and_run(Path("sub-001_run-1_ipu.csv"))
    ```
    """
    match = INPUT_FILENAME_PATTERN.match(csv_path.name)

    if match is None:
        raise ValueError(
            f"Filename does not match expected convention: {csv_path.name}"
        )

    subject = match.group("subject")
    run = f"{int(match.group('run')):02d}"

    return subject, run


################################################################################
#                              COLUMN NORMALIZATION                              #
################################################################################

def find_column(dataframe: pd.DataFrame, canonical_name: str) -> str | None:
    """
    Find a column in a dataframe using known aliases.

    Parameters
    ----------
    dataframe
        Input dataframe.
    canonical_name
        Canonical target column name.

    Returns
    -------
    str | None
        Matching source column name, or None if no match was found.

    Usage example
    -------------
    ```python
    source_column = find_column(dataframe, "onset")
    ```
    """
    lowercase_to_original = {column.lower(): column for column in dataframe.columns}

    for alias in COLUMN_ALIASES[canonical_name]:
        if alias.lower() in lowercase_to_original:
            return lowercase_to_original[alias.lower()]

    return None


def normalize_ipu_table(
    dataframe: pd.DataFrame,
    subject: str,
    run: str,
) -> pd.DataFrame:
    """
    Convert an IPU CSV table into a BIDS-like annotation table.

    Parameters
    ----------
    dataframe
        Raw IPU dataframe loaded from CSV.
    subject
        BIDS-style subject label, for example ``sub-001``.
    run
        Zero-padded run label, for example ``01``.

    Returns
    -------
    pandas.DataFrame
        Normalized IPU table with onset, duration, offset, speaker, and ipu_id.

    Usage example
    -------------
    ```python
    raw_table = pd.read_csv("sub-001_run-1_ipu.csv")
    ipu_table = normalize_ipu_table(raw_table, subject="sub-001", run="01")
    ```
    """
    output_table = dataframe.copy()

    onset_column = find_column(output_table, "onset")
    offset_column = find_column(output_table, "offset")
    duration_column = find_column(output_table, "duration")
    speaker_column = find_column(output_table, "speaker")
    label_column = find_column(output_table, "label")

    if onset_column is None:
        raise ValueError(
            f"Could not find onset/start column. Available columns: "
            f"{list(output_table.columns)}"
        )

    output_table = output_table.rename(columns={onset_column: "onset"})

    if offset_column is not None:
        output_table = output_table.rename(columns={offset_column: "offset"})

    if duration_column is not None:
        output_table = output_table.rename(columns={duration_column: "duration"})

    if speaker_column is not None:
        output_table = output_table.rename(columns={speaker_column: "speaker"})
    else:
        output_table["speaker"] = "self"

    if (
        label_column is not None
        and label_column not in {"onset", "offset", "duration", "speaker"}
    ):
        output_table = output_table.rename(columns={label_column: "label"})
    elif "label" not in output_table.columns:
        output_table["label"] = "IPU"

    output_table["onset"] = pd.to_numeric(output_table["onset"], errors="raise")

    if "offset" not in output_table.columns and "duration" not in output_table.columns:
        raise ValueError("Need either offset/end column or duration column.")

    if "offset" not in output_table.columns:
        output_table["duration"] = pd.to_numeric(
            output_table["duration"],
            errors="raise",
        )
        output_table["offset"] = output_table["onset"] + output_table["duration"]

    if "duration" not in output_table.columns:
        output_table["offset"] = pd.to_numeric(output_table["offset"], errors="raise")
        output_table["duration"] = output_table["offset"] - output_table["onset"]

    output_table["offset"] = pd.to_numeric(output_table["offset"], errors="raise")
    output_table["duration"] = pd.to_numeric(output_table["duration"], errors="raise")

    output_table = output_table.sort_values(["onset", "offset"]).reset_index(drop=True)

    output_table.insert(
        0,
        "ipu_id",
        [f"{subject}_run-{run}_ipu-{index + 1:04d}" for index in range(len(output_table))],
    )

    preferred_columns = [
        "onset",
        "duration",
        "offset",
        "speaker",
        "ipu_id",
        "label",
    ]

    remaining_columns = [
        column for column in output_table.columns if column not in preferred_columns
    ]

    return output_table[preferred_columns + remaining_columns]


################################################################################
#                               TEXTGRID WRITING                                 #
################################################################################

def escape_textgrid_text(value: Any) -> str:
    """
    Escape text for use in a Praat TextGrid interval label.

    Parameters
    ----------
    value
        Value to write into the TextGrid.

    Returns
    -------
    str
        Escaped TextGrid-safe string.

    Usage example
    -------------
    ```python
    label = escape_textgrid_text('speaker "A"')
    ```
    """
    return str(value).replace('"', '""')


def write_textgrid(ipu_table: pd.DataFrame, output_path: Path) -> None:
    """
    Write IPU intervals to a Praat TextGrid file.

    Parameters
    ----------
    ipu_table
        Normalized IPU table.
    output_path
        Destination TextGrid path.

    Usage example
    -------------
    ```python
    write_textgrid(ipu_table, Path("sub-001_task-conversation_run-01_desc-ipu_annotations.TextGrid"))
    ```
    """
    xmin = 0.0
    xmax = float(ipu_table["offset"].max()) if len(ipu_table) else 0.0

    lines: list[str] = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {xmin:.6f}",
        f"xmax = {xmax:.6f}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "IPU"',
        f"        xmin = {xmin:.6f}",
        f"        xmax = {xmax:.6f}",
        f"        intervals: size = {len(ipu_table)}",
    ]

    for interval_index, row in ipu_table.iterrows():
        speaker = escape_textgrid_text(row.get("speaker", ""))
        ipu_id = escape_textgrid_text(row.get("ipu_id", ""))
        interval_label = f"{speaker}|{ipu_id}" if speaker else ipu_id

        lines.extend(
            [
                f"        intervals [{interval_index + 1}]:",
                f"            xmin = {float(row['onset']):.6f}",
                f"            xmax = {float(row['offset']):.6f}",
                f'            text = "{interval_label}"',
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


################################################################################
#                                JSON SIDECAR                                    #
################################################################################

def make_json_sidecar() -> dict[str, Any]:
    """
    Create the JSON sidecar metadata for IPU annotations.

    Returns
    -------
    dict[str, Any]
        JSON-serializable metadata dictionary.

    Usage example
    -------------
    ```python
    metadata = make_json_sidecar()
    ```
    """
    return {
        "Description": "Inter-pausal unit annotations for one conversation run.",
        "AnnotationType": "interval",
        "TimeReference": (
            "All times are expressed in seconds relative to the start of the "
            "analyzable EEG conversation window for this run."
        ),
        "SourceData": "Manually corrected and time-aligned speech annotations.",
        "GeneratedBy": [
            {
                "Name": "DUET IPU annotation pipeline",
                "Description": (
                    "Inter-pausal units were derived from time-aligned speech "
                    "annotations as continuous speech intervals bounded by silence."
                ),
                "Parameters": {
                    "MinimumSilenceDuration": IPU_SILENCE_THRESHOLD_SECONDS,
                    "MinimumSilenceDurationUnits": "s",
                    "ExcludedEvents": [
                        "laughter",
                        "coughs",
                        "non-speech vocal events",
                    ],
                },
            }
        ],
        "Columns": {
            "onset": {
                "Description": "Start time of the IPU interval.",
                "Units": "s",
            },
            "duration": {
                "Description": "Duration of the IPU interval.",
                "Units": "s",
            },
            "offset": {
                "Description": "End time of the IPU interval.",
                "Units": "s",
            },
            "speaker": {
                "Description": (
                    "Speaker label for the participant who produced the IPU. "
                    "If the source file did not contain speaker labels, this is set to 'self'."
                ),
            },
            "ipu_id": {
                "Description": "Unique IPU identifier within subject and run.",
            },
            "label": {
                "Description": "Interval label for the annotation. Usually 'IPU'.",
            },
        },
    }


################################################################################
#                              CONVERSION LOGIC                                  #
################################################################################

def make_output_paths(
    output_directory: Path,
    subject: str,
    run: str,
) -> tuple[Path, Path, Path]:
    """
    Create output paths for one subject/run.

    Parameters
    ----------
    output_directory
        Root output directory for annotation derivatives.
    subject
        BIDS-style subject label.
    run
        Zero-padded run label.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        TSV path, JSON path, and TextGrid path.

    Usage example
    -------------
    ```python
    tsv_path, json_path, textgrid_path = make_output_paths(
        Path("derivatives/annotations"),
        subject="sub-001",
        run="01",
    )
    ```
    """
    subject_output_directory = output_directory / subject
    output_basename = (
        f"{subject}_task-{TASK}_run-{run}_desc-{DESCRIPTION}_annotations"
    )

    tsv_path = subject_output_directory / f"{output_basename}.tsv"
    json_path = subject_output_directory / f"{output_basename}.json"
    textgrid_path = subject_output_directory / f"{output_basename}.TextGrid"

    return tsv_path, json_path, textgrid_path


def convert_one_file(csv_path: Path, output_directory: Path) -> None:
    """
    Convert one IPU CSV file into TSV, JSON sidecar, and Praat TextGrid.

    Parameters
    ----------
    csv_path
        Source IPU CSV path.
    output_directory
        Root output directory for annotation derivatives.

    Usage example
    -------------
    ```python
    convert_one_file(
        Path("sub-001_run-1_ipu.csv"),
        Path("derivatives/annotations"),
    )
    ```
    """
    subject, run = parse_subject_and_run(csv_path)
    tsv_path, json_path, textgrid_path = make_output_paths(
        output_directory=output_directory,
        subject=subject,
        run=run,
    )

    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    raw_table = pd.read_csv(csv_path)
    ipu_table = normalize_ipu_table(raw_table, subject=subject, run=run)

    ipu_table.to_csv(tsv_path, sep="\t", index=False, na_rep="n/a")

    metadata = make_json_sidecar()
    json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_textgrid(ipu_table, textgrid_path)

    print(f"Wrote: {tsv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {textgrid_path}")


################################################################################
#                                COMMAND LINE                                    #
################################################################################

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.

    Usage example
    -------------
    ```bash
    python convert_ipu_annotations.py \
        --input-dir /Users/hiro/Projects/active/diapix-annotations/EEG/annotations/ipu_v1 \
        --output-dir /Users/hiro/Projects/active/openneuro-duet/derivatives/annotations
    ```
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert IPU CSV annotation files into BIDS-like TSV, JSON sidecar, "
            "and Praat TextGrid files."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing source files named like sub-001_run-1_ipu.csv.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root output directory, usually derivatives/annotations.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Convert all matching IPU CSV files in an input directory.

    Usage example
    -------------
    ```bash
    python convert_ipu_annotations.py \
        --input-dir /Users/hiro/Projects/active/diapix-annotations/EEG/annotations/ipu_v1 \
        --output-dir /Users/hiro/Projects/active/openneuro-duet/derivatives/annotations
    ```
    """
    arguments = parse_arguments()

    input_directory = arguments.input_dir.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()

    if not input_directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")

    if not input_directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_directory}")

    csv_files = iter_ipu_csv_files(input_directory)

    if not csv_files:
        print(f"No matching IPU CSV files found in: {input_directory}")
        return

    print(f"Found {len(csv_files)} IPU CSV file(s).")

    for csv_path in csv_files:
        convert_one_file(csv_path=csv_path, output_directory=output_directory)


if __name__ == "__main__":
    main()