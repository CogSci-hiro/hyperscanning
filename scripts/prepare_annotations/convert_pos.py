################################################################################
#                    CONVERT POS FEATURE FILES TO ANNOTATION DERIVATIVES         #
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

INPUT_FILENAME_PATTERN = re.compile(
    r"^(?P<subject>sub-\d+)_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d+)_"
    r"desc-(?P<role>[A-Za-z0-9]+)_pos_features\.tsv$"
)


################################################################################
#                              FILE DISCOVERY                                    #
################################################################################

def is_hidden_path(path: Path) -> bool:
    """
    Check whether a path contains hidden components.

    Parameters
    ----------
    path
        File or directory path.

    Returns
    -------
    bool
        Whether any path component starts with a dot.

    Usage example
    -------------
    ```python
    is_hidden = is_hidden_path(Path(".DS_Store"))
    ```
    """
    return any(part.startswith(".") for part in path.parts)


def iter_pos_tsv_files(input_directory: Path) -> list[Path]:
    """
    Find all non-hidden POS TSV files matching the expected naming convention.

    Parameters
    ----------
    input_directory
        Directory containing source POS feature TSV files.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of matching POS TSV files.

    Usage example
    -------------
    ```python
    files = iter_pos_tsv_files(Path("features/events/pos"))
    ```
    """
    tsv_files: list[Path] = []

    for tsv_path in input_directory.rglob("*.tsv"):
        relative_path = tsv_path.relative_to(input_directory)

        if is_hidden_path(relative_path):
            continue

        if INPUT_FILENAME_PATTERN.match(tsv_path.name) is None:
            continue

        tsv_files.append(tsv_path)

    return sorted(tsv_files)


def parse_pos_filename(tsv_path: Path) -> tuple[str, str, str, str]:
    """
    Parse subject, task, run, and role from a POS feature filename.

    Parameters
    ----------
    tsv_path
        Source POS TSV path.

    Returns
    -------
    tuple[str, str, str, str]
        Subject label, task label, zero-padded run label, and source role.

    Usage example
    -------------
    ```python
    subject, task, run, role = parse_pos_filename(
        Path("sub-004_task-conversation_run-3_desc-other_pos_features.tsv")
    )
    ```
    """
    match = INPUT_FILENAME_PATTERN.match(tsv_path.name)

    if match is None:
        raise ValueError(f"Unexpected POS filename: {tsv_path.name}")

    subject = match.group("subject")
    task = match.group("task")
    run = f"{int(match.group('run')):02d}"
    role = match.group("role")

    return subject, task, run, role


################################################################################
#                              TABLE NORMALIZATION                               #
################################################################################

def add_offset_if_needed(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the POS annotation table has onset, duration, and offset columns.

    Parameters
    ----------
    dataframe
        POS table.

    Returns
    -------
    pandas.DataFrame
        Table with onset, duration, and offset columns.

    Usage example
    -------------
    ```python
    table = add_offset_if_needed(table)
    ```
    """
    output_table = dataframe.copy()

    if "onset" not in output_table.columns:
        raise ValueError(
            f"POS table must contain an onset column. "
            f"Available columns: {list(output_table.columns)}"
        )

    if "duration" not in output_table.columns:
        if "offset" in output_table.columns:
            output_table["duration"] = (
                pd.to_numeric(output_table["offset"], errors="raise")
                - pd.to_numeric(output_table["onset"], errors="raise")
            )
        elif "end" in output_table.columns:
            output_table["duration"] = (
                pd.to_numeric(output_table["end"], errors="raise")
                - pd.to_numeric(output_table["onset"], errors="raise")
            )
        else:
            raise ValueError(
                "POS table must contain either duration, offset, or end."
            )

    if "offset" not in output_table.columns:
        if "end" in output_table.columns:
            output_table["offset"] = pd.to_numeric(output_table["end"], errors="raise")
        else:
            output_table["offset"] = (
                pd.to_numeric(output_table["onset"], errors="raise")
                + pd.to_numeric(output_table["duration"], errors="raise")
            )

    output_table["onset"] = pd.to_numeric(output_table["onset"], errors="raise")
    output_table["duration"] = pd.to_numeric(output_table["duration"], errors="raise")
    output_table["offset"] = pd.to_numeric(output_table["offset"], errors="raise")

    return output_table


def make_pos_annotation_table(
    dataframe: pd.DataFrame,
    subject: str,
    run: str,
    role: str,
) -> pd.DataFrame:
    """
    Convert a POS feature table into a POS annotation table.

    Parameters
    ----------
    dataframe
        Source POS feature table.
    subject
        BIDS-style subject label.
    run
        Zero-padded run label.
    role
        Source role from the original filename, for example ``self`` or ``other``.

    Returns
    -------
    pandas.DataFrame
        POS annotation table.

    Usage example
    -------------
    ```python
    table = make_pos_annotation_table(
        dataframe,
        subject="sub-004",
        run="03",
        role="other",
    )
    ```
    """
    output_table = add_offset_if_needed(dataframe)

    if "token" not in output_table.columns:
        raise ValueError(
            f"POS table must contain a token column. "
            f"Available columns: {list(output_table.columns)}"
        )

    if "source_interval_id" in output_table.columns:
        output_table = output_table.rename(
            columns={"source_interval_id": "word_id"}
        )
    elif "word_id" not in output_table.columns:
        output_table.insert(
            0,
            "word_id",
            [
                f"{subject}_run-{run}_word-{index + 1:05d}"
                for index in range(len(output_table))
            ],
        )

    output_table["source_role"] = role

    if "pos_model" not in output_table.columns:
        output_table["pos_model"] = "stanza"

    if "pos_lang" not in output_table.columns:
        output_table["pos_lang"] = "fr"

    if "mapping_status" not in output_table.columns:
        output_table["mapping_status"] = "n/a"

    if "lemma" not in output_table.columns:
        output_table["lemma"] = "n/a"

    if "upos" not in output_table.columns:
        output_table["upos"] = "n/a"

    if "xpos" not in output_table.columns:
        output_table["xpos"] = "n/a"

    if "morph" not in output_table.columns:
        output_table["morph"] = "n/a"

    output_table = output_table.sort_values(["onset", "offset"]).reset_index(drop=True)

    preferred_columns = [
        "onset",
        "duration",
        "offset",
        "word_id",
        "speaker",
        "source_role",
        "token",
        "lemma",
        "upos",
        "xpos",
        "morph",
        "mapping_status",
        "mapping_note",
        "pos_model",
        "pos_lang",
        "stanza_version",
    ]

    existing_preferred_columns = [
        column for column in preferred_columns if column in output_table.columns
    ]

    remaining_columns = [
        column
        for column in output_table.columns
        if column not in existing_preferred_columns
    ]

    return output_table[existing_preferred_columns + remaining_columns]


################################################################################
#                               TEXTGRID WRITING                                 #
################################################################################

def escape_textgrid_text(value: Any) -> str:
    """
    Escape text for use in a Praat TextGrid label.

    Parameters
    ----------
    value
        Value to write into a TextGrid interval.

    Returns
    -------
    str
        Escaped TextGrid-safe string.

    Usage example
    -------------
    ```python
    escaped = escape_textgrid_text('a "quoted" token')
    ```
    """
    return str(value).replace('"', '""')


def make_pos_textgrid_label(row: pd.Series) -> str:
    """
    Make a compact TextGrid label for one POS row.

    Parameters
    ----------
    row
        POS annotation row.

    Returns
    -------
    str
        TextGrid label containing token and POS information.

    Usage example
    -------------
    ```python
    label = make_pos_textgrid_label(row)
    ```
    """
    token = row.get("token", "")
    upos = row.get("upos", "")
    lemma = row.get("lemma", "")

    label_parts = [str(token)]

    if pd.notna(upos) and str(upos) not in {"", "n/a", "nan"}:
        label_parts.append(str(upos))

    if pd.notna(lemma) and str(lemma) not in {"", "n/a", "nan"}:
        label_parts.append(str(lemma))

    return "|".join(label_parts)


def write_pos_textgrid(annotation_table: pd.DataFrame, output_path: Path) -> None:
    """
    Write POS annotations to a Praat TextGrid.

    Parameters
    ----------
    annotation_table
        POS annotation table.
    output_path
        Destination TextGrid path.

    Usage example
    -------------
    ```python
    write_pos_textgrid(pos_table, Path("sub-004_task-conversation_run-03_desc-pos_annotations.TextGrid"))
    ```
    """
    xmin = 0.0
    xmax = float(annotation_table["offset"].max()) if len(annotation_table) else 0.0

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
        '        name = "POS"',
        f"        xmin = {xmin:.6f}",
        f"        xmax = {xmax:.6f}",
        f"        intervals: size = {len(annotation_table)}",
    ]

    for interval_index, row in annotation_table.iterrows():
        interval_label = escape_textgrid_text(make_pos_textgrid_label(row))

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

def load_source_json(json_path: Path) -> dict[str, Any]:
    """
    Load the original POS feature JSON sidecar when available.

    Parameters
    ----------
    json_path
        Source JSON path.

    Returns
    -------
    dict[str, Any]
        Parsed JSON, or an empty dictionary if the file does not exist.

    Usage example
    -------------
    ```python
    source_metadata = load_source_json(source_json_path)
    ```
    """
    if not json_path.exists():
        return {}

    return json.loads(json_path.read_text(encoding="utf-8"))


def make_pos_json_sidecar(
    source_metadata: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    """
    Create JSON metadata for POS annotations.

    Parameters
    ----------
    source_metadata
        Original POS feature JSON sidecar.
    role
        Source role from the filename, for example ``self`` or ``other``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable metadata dictionary.

    Usage example
    -------------
    ```python
    metadata = make_pos_json_sidecar(source_metadata, role="other")
    ```
    """
    source_columns = source_metadata.get("Columns", {})

    metadata: dict[str, Any] = {
        "Description": (
            "Part-of-speech annotations mapped onto word-level alignment rows "
            "for one conversation run."
        ),
        "AnnotationType": "interval",
        "TimeReference": (
            "All times are expressed in seconds relative to the start of the "
            "analyzable EEG conversation window for this run."
        ),
        "SourceData": (
            "Token-level POS feature files generated from aligned token tables "
            "and converted to annotation derivatives."
        ),
        "SourceRole": role,
        "GeneratedBy": [
            {
                "Name": "Stanza",
                "Description": (
                    "French POS tagging, lemmatization, and morphology annotation."
                ),
                "Version": (
                    source_metadata.get("Generation", {})
                    .get("ExtractionLibraryVersion")
                ),
            },
            {
                "Name": "DUET POS annotation conversion script",
                "Description": (
                    "Converted POS feature TSV/JSON files into BIDS-like annotation "
                    "TSV, JSON sidecar, and Praat TextGrid files. Original aligned "
                    "token rows were preserved."
                ),
            },
        ],
        "Method": source_metadata.get("Method", {}),
        "QualityControl": source_metadata.get("QualityControl", {}),
        "Columns": {
            "onset": {
                "Description": "Start time of the token interval.",
                "Units": "s",
            },
            "duration": {
                "Description": "Duration of the token interval.",
                "Units": "s",
            },
            "offset": {
                "Description": "End time of the token interval.",
                "Units": "s",
            },
            "word_id": {
                "Description": (
                    "Identifier of the aligned word/token row. This is inherited "
                    "from source_interval_id when available."
                ),
            },
            "speaker": source_columns.get(
                "speaker",
                {
                    "Description": "Speaker identifier from the aligned token source table.",
                },
            ),
            "source_role": {
                "Description": (
                    "Role encoded in the source POS filename, for example self or other."
                ),
            },
            "token": source_columns.get(
                "token",
                {
                    "Description": "Surface token from the aligned source table.",
                },
            ),
            "lemma": source_columns.get(
                "lemma",
                {
                    "Description": "Lemma emitted by Stanza when available.",
                },
            ),
            "upos": source_columns.get(
                "upos",
                {
                    "Description": "Universal Dependencies UPOS tag assigned by Stanza.",
                },
            ),
            "xpos": source_columns.get(
                "xpos",
                {
                    "Description": "Language-specific POS tag emitted by Stanza.",
                },
            ),
            "morph": source_columns.get(
                "morph",
                {
                    "Description": "Universal Dependencies morphological feature bundle.",
                },
            ),
            "mapping_status": source_columns.get(
                "mapping_status",
                {
                    "Description": (
                        "Diagnostic status describing how Stanza tokenization was "
                        "reconciled with the aligned token row."
                    ),
                },
            ),
            "mapping_note": source_columns.get(
                "mapping_note",
                {
                    "Description": "Human-readable reconciliation note for non-exact mappings.",
                },
            ),
            "pos_model": source_columns.get(
                "pos_model",
                {
                    "Description": "POS annotation software family used for the row.",
                },
            ),
            "pos_lang": source_columns.get(
                "pos_lang",
                {
                    "Description": "Language code used to load the Stanza model.",
                },
            ),
            "stanza_version": source_columns.get(
                "stanza_version",
                {
                    "Description": "Installed Stanza package version used for annotation.",
                },
            ),
        },
    }

    return metadata


################################################################################
#                              OUTPUT PATHS                                      #
################################################################################

def make_output_paths(
    output_directory: Path,
    subject: str,
    task: str,
    run: str,
) -> tuple[Path, Path, Path]:
    """
    Create output paths for one POS annotation file set.

    Parameters
    ----------
    output_directory
        Root output directory for annotation derivatives.
    subject
        BIDS-style subject label.
    task
        Task label.
    run
        Zero-padded run label.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        TSV, JSON, and TextGrid paths.

    Usage example
    -------------
    ```python
    paths = make_output_paths(Path("derivatives/annotations"), "sub-004", "conversation", "03")
    ```
    """
    subject_output_directory = output_directory / subject
    output_basename = f"{subject}_task-{task}_run-{run}_desc-pos_annotations"

    tsv_path = subject_output_directory / f"{output_basename}.tsv"
    json_path = subject_output_directory / f"{output_basename}.json"
    textgrid_path = subject_output_directory / f"{output_basename}.TextGrid"

    return tsv_path, json_path, textgrid_path


################################################################################
#                              CONVERSION LOGIC                                  #
################################################################################

def convert_one_file(tsv_path: Path, output_directory: Path) -> None:
    """
    Convert one POS feature TSV/JSON pair into annotation files.

    Parameters
    ----------
    tsv_path
        Source POS feature TSV file.
    output_directory
        Root output directory for annotation derivatives.

    Usage example
    -------------
    ```python
    convert_one_file(
        Path("sub-004_task-conversation_run-3_desc-other_pos_features.tsv"),
        Path("derivatives/annotations"),
    )
    ```
    """
    subject, task, run, role = parse_pos_filename(tsv_path)

    source_json_path = tsv_path.with_suffix(".json")
    source_metadata = load_source_json(source_json_path)

    raw_table = pd.read_csv(tsv_path, sep="\t", na_values=["n/a"])
    annotation_table = make_pos_annotation_table(
        dataframe=raw_table,
        subject=subject,
        run=run,
        role=role,
    )

    output_tsv_path, output_json_path, output_textgrid_path = make_output_paths(
        output_directory=output_directory,
        subject=subject,
        task=task,
        run=run,
    )

    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)

    annotation_table.to_csv(
        output_tsv_path,
        sep="\t",
        index=False,
        na_rep="n/a",
    )

    metadata = make_pos_json_sidecar(
        source_metadata=source_metadata,
        role=role,
    )

    output_json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_pos_textgrid(
        annotation_table=annotation_table,
        output_path=output_textgrid_path,
    )

    print(f"Wrote: {output_tsv_path}")
    print(f"Wrote: {output_json_path}")
    print(f"Wrote: {output_textgrid_path}")


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
    python convert_pos_annotations.py \
        --input-dir /Users/hiro/Datasets/working/hyperscanning-derived/features/events/pos \
        --output-dir /Users/hiro/Datasets/DUET-root/derivatives/annotations
    ```
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert POS feature TSV/JSON files into POS annotation TSV, JSON, "
            "and Praat TextGrid files."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing files named like sub-004_task-conversation_run-3_desc-other_pos_features.tsv.",
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
    Convert all matching POS feature files in an input directory.

    Usage example
    -------------
    ```bash
    python convert_pos_annotations.py \
        --input-dir /Users/hiro/Datasets/working/hyperscanning-derived/features/events/pos \
        --output-dir /Users/hiro/Datasets/DUET-root/derivatives/annotations
    ```
    """
    arguments = parse_arguments()

    input_directory = arguments.input_dir.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()

    if not input_directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")

    if not input_directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_directory}")

    tsv_files = iter_pos_tsv_files(input_directory)

    if not tsv_files:
        print(f"No matching POS TSV files found in: {input_directory}")
        return

    print(f"Found {len(tsv_files)} POS TSV file(s).")

    for tsv_path in tsv_files:
        convert_one_file(tsv_path=tsv_path, output_directory=output_directory)


if __name__ == "__main__":
    main()