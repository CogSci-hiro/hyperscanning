################################################################################
#                 CONVERT SPPAS PALIGN CSV TO WORD AND PHONEME ANNOTATIONS      #
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
    r"^(?P<subject>sub-\d+)_run-(?P<run>\d+)_palign\.csv$"
)

WORD_TIER_NAMES = {
    "TokensAlign",
    "TokenAlign",
    "WordsAlign",
    "WordAlign",
    "tokens",
    "words",
    "word",
}

PHONEME_TIER_NAMES = {
    "PhonAlign",
    "PhonesAlign",
    "PhonemesAlign",
    "phonemes",
    "phones",
    "phoneme",
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "onset": ("onset", "start", "start_time", "tmin", "xmin", "begin"),
    "offset": ("offset", "end", "end_time", "tmax", "xmax", "stop"),
    "duration": ("duration", "dur"),
    "tier": ("tier", "tier_name", "name", "annotation_tier"),
    "label": ("label", "text", "annotation", "value", "token", "phoneme"),
}

XSAMPA_MULTI_CHARACTER_SYMBOLS: dict[str, str] = {
    "a~": "ɑ̃",
    "e~": "ɛ̃",
    "o~": "ɔ̃",
    "9~": "œ̃",
    "E~": "ɛ̃",
    "O~": "ɔ̃",
    "A~": "ɑ̃",
    "N": "ŋ",
    "S": "ʃ",
    "Z": "ʒ",
    "J": "ɲ",
    "R": "ʁ",
    "E": "ɛ",
    "O": "ɔ",
    "2": "ø",
    "9": "œ",
    "@": "ə",
}

XSAMPA_SINGLE_CHARACTER_SYMBOLS: dict[str, str] = {
    "a": "a",
    "b": "b",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "g",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "p": "p",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "y": "y",
    "z": "z",
}


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


def iter_palign_csv_files(input_directory: Path) -> list[Path]:
    """
    Find all non-hidden palign CSV files matching the expected naming convention.

    Parameters
    ----------
    input_directory
        Directory containing source palign CSV files.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of matching palign CSV files.

    Usage example
    -------------
    ```python
    files = iter_palign_csv_files(Path("annotations/palign_v1"))
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
    Parse subject and run labels from a palign filename.

    Parameters
    ----------
    csv_path
        Source CSV path.

    Returns
    -------
    tuple[str, str]
        Subject label and zero-padded run label.

    Usage example
    -------------
    ```python
    subject, run = parse_subject_and_run(Path("sub-001_run-1_palign.csv"))
    ```
    """
    match = INPUT_FILENAME_PATTERN.match(csv_path.name)

    if match is None:
        raise ValueError(f"Unexpected filename: {csv_path.name}")

    subject = match.group("subject")
    run = f"{int(match.group('run')):02d}"

    return subject, run


################################################################################
#                              COLUMN HELPERS                                    #
################################################################################

def find_column(dataframe: pd.DataFrame, canonical_name: str) -> str | None:
    """
    Find a source column using known aliases.

    Parameters
    ----------
    dataframe
        Input dataframe.
    canonical_name
        Canonical column name.

    Returns
    -------
    str | None
        Matching column name, or None.

    Usage example
    -------------
    ```python
    onset_column = find_column(dataframe, "onset")
    ```
    """
    lowercase_to_original = {column.lower(): column for column in dataframe.columns}

    for alias in COLUMN_ALIASES[canonical_name]:
        if alias.lower() in lowercase_to_original:
            return lowercase_to_original[alias.lower()]

    return None


def normalize_interval_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize onset, offset, duration, tier, and label columns.

    Parameters
    ----------
    dataframe
        Source palign dataframe.

    Returns
    -------
    pandas.DataFrame
        Dataframe with canonical interval columns.

    Usage example
    -------------
    ```python
    table = normalize_interval_columns(raw_table)
    ```
    """
    output_table = dataframe.copy()

    onset_column = find_column(output_table, "onset")
    offset_column = find_column(output_table, "offset")
    duration_column = find_column(output_table, "duration")
    tier_column = find_column(output_table, "tier")
    label_column = find_column(output_table, "label")

    if onset_column is None:
        raise ValueError(
            f"Could not find onset/start column. Available columns: "
            f"{list(output_table.columns)}"
        )

    if label_column is None:
        raise ValueError(
            f"Could not find label/text column. Available columns: "
            f"{list(output_table.columns)}"
        )

    output_table = output_table.rename(columns={onset_column: "onset"})
    output_table = output_table.rename(columns={label_column: "label"})

    if offset_column is not None:
        output_table = output_table.rename(columns={offset_column: "offset"})

    if duration_column is not None:
        output_table = output_table.rename(columns={duration_column: "duration"})

    if tier_column is not None:
        output_table = output_table.rename(columns={tier_column: "tier"})
    else:
        output_table["tier"] = ""

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

    output_table["label"] = output_table["label"].fillna("").astype(str)
    output_table["tier"] = output_table["tier"].fillna("").astype(str)

    return output_table


def read_palign_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read an SPPAS palign CSV file.

    The source palign files are expected to contain four columns without a
    header row:

    tier,onset,offset,label

    Parameters
    ----------
    csv_path
        Path to the source palign CSV file.

    Returns
    -------
    pandas.DataFrame
        Palign table with canonical source columns.

    Usage example
    -------------
    ```python
    table = read_palign_csv(Path("sub-001_run-1_palign.csv"))
    ```
    """
    table = pd.read_csv(
        csv_path,
        header=None,
        names=["tier", "onset", "offset", "label"],
    )

    return table


################################################################################
#                            TIER SPLITTING                                      #
################################################################################

def select_tier_rows(
    dataframe: pd.DataFrame,
    tier_names: set[str],
    fallback_label: str,
) -> pd.DataFrame:
    """
    Select rows belonging to a word or phoneme tier.

    Parameters
    ----------
    dataframe
        Normalized palign dataframe.
    tier_names
        Accepted tier names.
    fallback_label
        Human-readable label used in error messages.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe.

    Usage example
    -------------
    ```python
    words = select_tier_rows(table, WORD_TIER_NAMES, "word")
    ```
    """
    normalized_tier_names = {name.lower() for name in tier_names}
    tier_values = dataframe["tier"].str.lower()

    selected_table = dataframe[tier_values.isin(normalized_tier_names)].copy()

    if selected_table.empty:
        available_tiers = sorted(dataframe["tier"].dropna().unique().tolist())
        raise ValueError(
            f"Could not find {fallback_label} tier. "
            f"Expected one of {sorted(tier_names)}. "
            f"Available tiers: {available_tiers}"
        )

    return selected_table.sort_values(["onset", "offset"]).reset_index(drop=True)


################################################################################
#                            TOKEN AND PHONEME CLEANING                          #
################################################################################

def clean_sppas_token(token: Any) -> str:
    """
    Convert an SPPAS token into a cleaner human-readable token.

    This keeps the original token in the TSV, but creates a normalized version
    for easier reuse. The function replaces SPPAS underscore joins with spaces
    and strips common alignment whitespace.

    Parameters
    ----------
    token
        Original token label.

    Returns
    -------
    str
        Cleaned token.

    Usage example
    -------------
    ```python
    clean_token = clean_sppas_token("du_coup")
    ```
    """
    if pd.isna(token):
        return ""

    clean_token = str(token).strip()

    if clean_token in {"", "#", "*", "@"}:
        return clean_token

    clean_token = clean_token.replace("_", " ")
    clean_token = clean_token.replace("’", "'")
    clean_token = re.sub(r"\s+", " ", clean_token)

    return clean_token.strip()


def xsampa_to_ipa(label: Any) -> str:
    """
    Convert a French-oriented XSAMPA phoneme label to IPA.

    The converter preserves unknown symbols rather than failing. This is
    intentional because SPPAS labels may contain special symbols, pauses, or
    dataset-specific markers.

    Parameters
    ----------
    label
        XSAMPA phoneme label.

    Returns
    -------
    str
        IPA phoneme label.

    Usage example
    -------------
    ```python
    ipa = xsampa_to_ipa("S")
    ```
    """
    if pd.isna(label):
        return ""

    source_label = str(label).strip()

    if source_label in {"", "#", "*", "@"}:
        return source_label

    output_symbols: list[str] = []
    index = 0

    multi_character_keys = sorted(
        XSAMPA_MULTI_CHARACTER_SYMBOLS,
        key=len,
        reverse=True,
    )

    while index < len(source_label):
        matched_symbol = None

        for xsampa_symbol in multi_character_keys:
            if source_label.startswith(xsampa_symbol, index):
                matched_symbol = xsampa_symbol
                break

        if matched_symbol is not None:
            output_symbols.append(XSAMPA_MULTI_CHARACTER_SYMBOLS[matched_symbol])
            index += len(matched_symbol)
            continue

        character = source_label[index]
        output_symbols.append(
            XSAMPA_SINGLE_CHARACTER_SYMBOLS.get(character, character)
        )
        index += 1

    return "".join(output_symbols)


################################################################################
#                            WORD AND PHONEME TABLES                             #
################################################################################

def make_word_table(
    raw_table: pd.DataFrame,
    subject: str,
    run: str,
) -> pd.DataFrame:
    """
    Create a word-level annotation table from palign data.

    Parameters
    ----------
    raw_table
        Raw palign dataframe.
    subject
        BIDS-style subject label.
    run
        Zero-padded run label.

    Returns
    -------
    pandas.DataFrame
        Word-level annotation table.

    Usage example
    -------------
    ```python
    word_table = make_word_table(raw_table, subject="sub-001", run="01")
    ```
    """
    normalized_table = normalize_interval_columns(raw_table)
    word_table = select_tier_rows(
        dataframe=normalized_table,
        tier_names=WORD_TIER_NAMES,
        fallback_label="word",
    )

    word_table = word_table.rename(columns={"label": "token"})
    word_table["clean_token"] = word_table["token"].map(clean_sppas_token)

    word_table.insert(
        0,
        "word_id",
        [f"{subject}_run-{run}_word-{index + 1:05d}" for index in range(len(word_table))],
    )

    preferred_columns = [
        "onset",
        "duration",
        "offset",
        "word_id",
        "token",
        "clean_token",
        "tier",
    ]

    remaining_columns = [
        column for column in word_table.columns if column not in preferred_columns
    ]

    return word_table[preferred_columns + remaining_columns]


def make_phoneme_table(
    raw_table: pd.DataFrame,
    subject: str,
    run: str,
) -> pd.DataFrame:
    """
    Create a phoneme-level annotation table from palign data.

    Parameters
    ----------
    raw_table
        Raw palign dataframe.
    subject
        BIDS-style subject label.
    run
        Zero-padded run label.

    Returns
    -------
    pandas.DataFrame
        Phoneme-level annotation table.

    Usage example
    -------------
    ```python
    phoneme_table = make_phoneme_table(raw_table, subject="sub-001", run="01")
    ```
    """
    normalized_table = normalize_interval_columns(raw_table)
    phoneme_table = select_tier_rows(
        dataframe=normalized_table,
        tier_names=PHONEME_TIER_NAMES,
        fallback_label="phoneme",
    )

    phoneme_table = phoneme_table.rename(columns={"label": "phoneme_xsampa"})
    phoneme_table["phoneme_ipa"] = phoneme_table["phoneme_xsampa"].map(xsampa_to_ipa)

    phoneme_table.insert(
        0,
        "phoneme_id",
        [
            f"{subject}_run-{run}_phoneme-{index + 1:05d}"
            for index in range(len(phoneme_table))
        ],
    )

    preferred_columns = [
        "onset",
        "duration",
        "offset",
        "phoneme_id",
        "phoneme_xsampa",
        "phoneme_ipa",
        "tier",
    ]

    remaining_columns = [
        column for column in phoneme_table.columns if column not in preferred_columns
    ]

    return phoneme_table[preferred_columns + remaining_columns]


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
        Escaped string.

    Usage example
    -------------
    ```python
    escaped = escape_textgrid_text('a "quoted" token')
    ```
    """
    return str(value).replace('"', '""')


def write_interval_textgrid(
    annotation_table: pd.DataFrame,
    output_path: Path,
    tier_name: str,
    label_column: str,
) -> None:
    """
    Write an interval annotation table to a Praat TextGrid.

    Parameters
    ----------
    annotation_table
        Annotation table with onset and offset columns.
    output_path
        Destination TextGrid path.
    tier_name
        Name of the TextGrid interval tier.
    label_column
        Column to use as interval text.

    Usage example
    -------------
    ```python
    write_interval_textgrid(word_table, path, "Words", "clean_token")
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
        f'        name = "{escape_textgrid_text(tier_name)}"',
        f"        xmin = {xmin:.6f}",
        f"        xmax = {xmax:.6f}",
        f"        intervals: size = {len(annotation_table)}",
    ]

    for interval_index, row in annotation_table.iterrows():
        interval_label = escape_textgrid_text(row.get(label_column, ""))

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
#                                JSON SIDECARS                                   #
################################################################################

def make_word_json_sidecar() -> dict[str, Any]:
    """
    Create JSON metadata for word-level annotations.

    Returns
    -------
    dict[str, Any]
        JSON-serializable metadata dictionary.

    Usage example
    -------------
    ```python
    metadata = make_word_json_sidecar()
    ```
    """
    return {
        "Description": "Word-level time-aligned annotations for one conversation run.",
        "AnnotationType": "interval",
        "TimeReference": (
            "All times are expressed in seconds relative to the start of the "
            "analyzable EEG conversation window for this run."
        ),
        "SourceData": "SPPAS palign output derived from manually corrected transcripts.",
        "GeneratedBy": [
            {
                "Name": "SPPAS",
                "Description": "Forced alignment of manually corrected speech transcripts.",
            },
            {
                "Name": "DUET palign conversion script",
                "Description": (
                    "Rows from the SPPAS token alignment tier were converted to "
                    "BIDS-like TSV annotations. Original SPPAS tokens were preserved "
                    "and cleaned human-readable tokens were added."
                ),
            },
        ],
        "Columns": {
            "onset": {
                "Description": "Start time of the word interval.",
                "Units": "s",
            },
            "duration": {
                "Description": "Duration of the word interval.",
                "Units": "s",
            },
            "offset": {
                "Description": "End time of the word interval.",
                "Units": "s",
            },
            "word_id": {
                "Description": "Unique word identifier within subject and run.",
            },
            "token": {
                "Description": "Original token label from SPPAS.",
            },
            "clean_token": {
                "Description": (
                    "Human-readable token derived from the SPPAS token. "
                    "Underscore-joined forms are converted to space-separated forms."
                ),
            },
            "tier": {
                "Description": "Original SPPAS tier name.",
            },
        },
    }


def make_phoneme_json_sidecar() -> dict[str, Any]:
    """
    Create JSON metadata for phoneme-level annotations.

    Returns
    -------
    dict[str, Any]
        JSON-serializable metadata dictionary.

    Usage example
    -------------
    ```python
    metadata = make_phoneme_json_sidecar()
    ```
    """
    return {
        "Description": "Phoneme-level time-aligned annotations for one conversation run.",
        "AnnotationType": "interval",
        "TimeReference": (
            "All times are expressed in seconds relative to the start of the "
            "analyzable EEG conversation window for this run."
        ),
        "SourceData": "SPPAS palign output derived from manually corrected transcripts.",
        "GeneratedBy": [
            {
                "Name": "SPPAS",
                "Description": "Forced alignment of manually corrected speech transcripts.",
            },
            {
                "Name": "DUET palign conversion script",
                "Description": (
                    "Rows from the SPPAS phoneme alignment tier were converted to "
                    "BIDS-like TSV annotations. Original XSAMPA labels were preserved "
                    "and IPA labels were added."
                ),
            },
        ],
        "Columns": {
            "onset": {
                "Description": "Start time of the phoneme interval.",
                "Units": "s",
            },
            "duration": {
                "Description": "Duration of the phoneme interval.",
                "Units": "s",
            },
            "offset": {
                "Description": "End time of the phoneme interval.",
                "Units": "s",
            },
            "phoneme_id": {
                "Description": "Unique phoneme identifier within subject and run.",
            },
            "phoneme_xsampa": {
                "Description": "Original XSAMPA phoneme label from SPPAS.",
            },
            "phoneme_ipa": {
                "Description": (
                    "IPA conversion of phoneme_xsampa. Unknown or special symbols "
                    "are preserved."
                ),
            },
            "tier": {
                "Description": "Original SPPAS tier name.",
            },
        },
    }


################################################################################
#                              OUTPUT PATHS                                      #
################################################################################

def make_output_paths(
    output_directory: Path,
    subject: str,
    run: str,
    description: str,
) -> tuple[Path, Path, Path]:
    """
    Create output paths for one subject/run/description.

    Parameters
    ----------
    output_directory
        Root output directory for annotation derivatives.
    subject
        BIDS-style subject label.
    run
        Zero-padded run label.
    description
        Annotation description, for example ``words`` or ``phonemes``.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        TSV, JSON, and TextGrid paths.

    Usage example
    -------------
    ```python
    paths = make_output_paths(Path("derivatives/annotations"), "sub-001", "01", "words")
    ```
    """
    subject_output_directory = output_directory / subject
    output_basename = (
        f"{subject}_task-{TASK}_run-{run}_desc-{description}_annotations"
    )

    tsv_path = subject_output_directory / f"{output_basename}.tsv"
    json_path = subject_output_directory / f"{output_basename}.json"
    textgrid_path = subject_output_directory / f"{output_basename}.TextGrid"

    return tsv_path, json_path, textgrid_path


################################################################################
#                              CONVERSION LOGIC                                  #
################################################################################

def write_annotation_set(
    annotation_table: pd.DataFrame,
    metadata: dict[str, Any],
    tsv_path: Path,
    json_path: Path,
    textgrid_path: Path,
    textgrid_tier_name: str,
    textgrid_label_column: str,
) -> None:
    """
    Write TSV, JSON sidecar, and TextGrid for one annotation table.

    Parameters
    ----------
    annotation_table
        Annotation dataframe.
    metadata
        JSON sidecar metadata.
    tsv_path
        Destination TSV path.
    json_path
        Destination JSON path.
    textgrid_path
        Destination TextGrid path.
    textgrid_tier_name
        Name of the TextGrid tier.
    textgrid_label_column
        Column used as interval labels in TextGrid.

    Usage example
    -------------
    ```python
    write_annotation_set(
        word_table,
        make_word_json_sidecar(),
        tsv_path,
        json_path,
        textgrid_path,
        "Words",
        "clean_token",
    )
    ```
    """
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    annotation_table.to_csv(tsv_path, sep="\t", index=False, na_rep="n/a")

    json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_interval_textgrid(
        annotation_table=annotation_table,
        output_path=textgrid_path,
        tier_name=textgrid_tier_name,
        label_column=textgrid_label_column,
    )

    print(f"Wrote: {tsv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {textgrid_path}")


def convert_one_file(csv_path: Path, output_directory: Path) -> None:
    """
    Convert one palign CSV into word and phoneme annotation files.

    Parameters
    ----------
    csv_path
        Source palign CSV path.
    output_directory
        Root output directory for annotation derivatives.

    Usage example
    -------------
    ```python
    convert_one_file(
        Path("sub-001_run-1_palign.csv"),
        Path("derivatives/annotations"),
    )
    ```
    """
    subject, run = parse_subject_and_run(csv_path)
    raw_table = read_palign_csv(csv_path)

    word_table = make_word_table(raw_table, subject=subject, run=run)
    phoneme_table = make_phoneme_table(raw_table, subject=subject, run=run)

    word_tsv_path, word_json_path, word_textgrid_path = make_output_paths(
        output_directory=output_directory,
        subject=subject,
        run=run,
        description="words",
    )

    phoneme_tsv_path, phoneme_json_path, phoneme_textgrid_path = make_output_paths(
        output_directory=output_directory,
        subject=subject,
        run=run,
        description="phonemes",
    )

    write_annotation_set(
        annotation_table=word_table,
        metadata=make_word_json_sidecar(),
        tsv_path=word_tsv_path,
        json_path=word_json_path,
        textgrid_path=word_textgrid_path,
        textgrid_tier_name="Words",
        textgrid_label_column="clean_token",
    )

    write_annotation_set(
        annotation_table=phoneme_table,
        metadata=make_phoneme_json_sidecar(),
        tsv_path=phoneme_tsv_path,
        json_path=phoneme_json_path,
        textgrid_path=phoneme_textgrid_path,
        textgrid_tier_name="Phonemes",
        textgrid_label_column="phoneme_ipa",
    )


################################################################################
#                                COMMAND LINE                                    #
################################################################################

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.

    Usage example
    -------------
    ```bash
    python convert_palign_annotations.py \
        --input-dir /Users/hiro/Projects/active/diapix-annotations/EEG/annotations/palign_v1 \
        --output-dir /Users/hiro/Datasets/DUET-root/derivatives/annotations
    ```
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert SPPAS palign CSV files into word and phoneme annotation "
            "TSV, JSON, and TextGrid files."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing files named like sub-001_run-1_palign.csv.",
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
    Convert all matching palign CSV files in an input directory.

    Usage example
    -------------
    ```bash
    python convert_palign_annotations.py \
        --input-dir /Users/hiro/Projects/active/diapix-annotations/EEG/annotations/palign_v1 \
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

    csv_files = iter_palign_csv_files(input_directory)

    if not csv_files:
        print(f"No matching palign CSV files found in: {input_directory}")
        return

    print(f"Found {len(csv_files)} palign CSV file(s).")

    for csv_path in csv_files:
        convert_one_file(csv_path=csv_path, output_directory=output_directory)


if __name__ == "__main__":
    main()