"""Stanza-backed French POS extraction aligned to existing token rows."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_STANZA_LANGUAGE: str = "fr"
DEFAULT_STANZA_PROCESSORS: str = "tokenize,pos,lemma"
DEFAULT_POS_MODEL_NAME: str = "stanza"
DEFAULT_POS_MODEL_SOURCE: str = "Stanford NLP Stanza pretrained models"
DEFAULT_FEATURE_NAME: str = "pos"
MISSING_VALUE: str = "n/a"
SPACELESS_PUNCTUATION_PREFIXES: tuple[str, ...] = ("'", "’", "-", ")", "]", "}", ".", ",", ";", ":", "!", "?", "%")
SPACELESS_PUNCTUATION_SUFFIXES: tuple[str, ...] = ("(", "[", "{", "«", "“", '"')
PREFERRED_TEXT_COLUMNS: tuple[str, ...] = ("token", "word", "label")
APOSTROPHE_TRANSLATION: dict[int, str] = str.maketrans({
    ord("’"): "'",
    ord("ʼ"): "'",
    ord("`"): "'",
    ord("´"): "'",
})
SPPAS_TOKEN_TEXT_OVERRIDES: dict[str, str] = {
    "est-ce_que": "est-ce que",
}


@dataclass(frozen=True, slots=True)
class StanzaPosConfig:
    """Configuration for token-aligned Stanza POS extraction.

    Parameters
    ----------
    enabled
        Whether Stanza POS extraction should be considered active in workflow config.
    language
        Stanza language code.
    processors
        Comma-separated Stanza processor list.
    resources_dir
        Optional local Stanza resources directory.
    allow_download
        Whether missing resources may be downloaded automatically.
    preserve_unmapped_rows
        Must remain True for this project so exported rows stay 1:1 with aligned tokens.
    fail_on_mapping_error
        If True, raise an error when any row is not mapped exactly.
    """

    enabled: bool = True
    language: str = DEFAULT_STANZA_LANGUAGE
    processors: str = DEFAULT_STANZA_PROCESSORS
    resources_dir: Path | None = None
    allow_download: bool = True
    preserve_unmapped_rows: bool = True
    fail_on_mapping_error: bool = False


@dataclass(frozen=True, slots=True)
class StanzaRuntimeInfo:
    """Resolved Stanza runtime metadata used in exports."""

    model_name: str
    model_source: str
    stanza_version: str | None
    resources_dir: Path
    processors: tuple[str, ...]
    language: str


@dataclass(frozen=True, slots=True)
class StanzaWordAnnotation:
    """One flattened Stanza word annotation."""

    text: str
    upos: str | None
    xpos: str | None
    feats: str | None
    lemma: str | None


@dataclass(frozen=True, slots=True)
class StanzaTokenAnnotation:
    """One Stanza surface token plus its underlying word annotations."""

    text: str
    words: tuple[StanzaWordAnnotation, ...]


@dataclass(frozen=True, slots=True)
class StanzaPosResult:
    """Tabular POS output plus reproducibility metadata."""

    event_table: pd.DataFrame
    runtime: StanzaRuntimeInfo
    text_column: str
    reconstructed_text: str


def default_stanza_resources_dir() -> Path:
    """Return the project-default Stanza cache directory."""
    env_value = os.environ.get("HYPER_STANZA_RESOURCES_DIR")
    if env_value:
        return Path(env_value).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "hyper" / "stanza"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "hyper" / "stanza"
    return Path.home() / ".cache" / "hyper" / "stanza"


def _resolved_resources_dir(resources_dir: Path | None) -> Path:
    """Resolve the active resources directory."""
    resolved = default_stanza_resources_dir() if resources_dir is None else Path(resources_dir)
    return resolved.expanduser().resolve()


def _installed_stanza_version() -> str | None:
    """Return the installed Stanza version when available."""
    try:
        return version("stanza")
    except PackageNotFoundError:
        return None


def load_stanza_pos_pipeline(config: StanzaPosConfig) -> tuple[Any, StanzaRuntimeInfo]:
    """Load and validate a Stanza pipeline for POS annotation."""
    if not config.preserve_unmapped_rows:
        raise ValueError(
            "stanza_pos.preserve_unmapped_rows must remain true because released POS rows "
            "must preserve 1:1 alignment with the input token table."
        )

    try:
        import stanza
        from stanza.pipeline.core import DownloadMethod
    except ImportError as exc:
        raise RuntimeError(
            "Stanza POS extraction requires the optional dependency `stanza`. "
            f"The current interpreter is Python {sys.version.split()[0]}. "
            "Install project dependencies in an environment where `stanza` is available "
            "before invoking `hyper pos-tags`."
        ) from exc

    resources_dir = _resolved_resources_dir(config.resources_dir)
    resources_dir.mkdir(parents=True, exist_ok=True)
    processors = tuple(part.strip() for part in str(config.processors).split(",") if part.strip())
    download_method = (
        DownloadMethod.DOWNLOAD_RESOURCES if config.allow_download else DownloadMethod.REUSE_RESOURCES
    )

    try:
        pipeline = stanza.Pipeline(
            lang=str(config.language),
            dir=str(resources_dir),
            processors=",".join(processors),
            download_method=download_method,
            logging_level="WARN",
            use_gpu=False,
        )
    except Exception as exc:  # noqa: BLE001 - deliberate boundary for dependency failures
        if config.allow_download:
            raise RuntimeError(
                "Unable to initialize the Stanza French POS pipeline. "
                f"Automatic model download or reuse failed in {resources_dir}. "
                "Ensure network access is available for the first run, or point "
                "`features.stanza_pos.resources_dir` / `--resources-dir` to a populated Stanza cache. "
                f"Original error: {exc}"
            ) from exc
        raise RuntimeError(
            "Unable to initialize the Stanza French POS pipeline from local resources. "
            f"No usable Stanza resources were found in {resources_dir}, and automatic download is disabled. "
            "Populate that directory first or re-run with `allow_download: true`. "
            f"Original error: {exc}"
        ) from exc

    runtime = StanzaRuntimeInfo(
        model_name=DEFAULT_POS_MODEL_NAME,
        model_source=DEFAULT_POS_MODEL_SOURCE,
        stanza_version=_installed_stanza_version(),
        resources_dir=resources_dir,
        processors=processors,
        language=str(config.language),
    )
    return pipeline, runtime


def _normalize_surface(value: object) -> str:
    """Normalize token text for conservative sequence matching."""
    if pd.isna(value):
        return ""
    text = str(value).strip().translate(APOSTROPHE_TRANSLATION)
    lowered = text.lower()
    text = SPPAS_TOKEN_TEXT_OVERRIDES.get(lowered, text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", "", text)
    return text


def _normalize_token_for_stanza_text(value: object) -> str:
    """Normalize one aligned token string before reconstructing Stanza text."""
    if pd.isna(value):
        return ""
    text = str(value).strip().translate(APOSTROPHE_TRANSLATION)
    lowered = text.lower()
    text = SPPAS_TOKEN_TEXT_OVERRIDES.get(lowered, text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _resolve_text_column(token_table: pd.DataFrame) -> str:
    """Choose the canonical text column from the aligned input table."""
    for column in PREFERRED_TEXT_COLUMNS:
        if column in token_table.columns:
            return column
    available = ", ".join(str(col) for col in token_table.columns)
    raise ValueError(
        "Aligned token table must include one of the text columns "
        f"{PREFERRED_TEXT_COLUMNS!r}; available columns: {available}"
    )


def _join_tokens_for_stanza(token_values: list[str]) -> str:
    """Reconstruct deterministic text from aligned token rows."""
    pieces: list[str] = []
    for raw_token in token_values:
        token = _normalize_token_for_stanza_text(raw_token)
        if not token:
            continue
        if not pieces:
            pieces.append(token)
            continue
        previous = pieces[-1]
        if (
            token.startswith(SPACELESS_PUNCTUATION_PREFIXES)
            or previous.endswith(SPACELESS_PUNCTUATION_SUFFIXES)
            or previous.endswith("'")
            or previous.endswith("’")
        ):
            pieces[-1] = previous + token
        else:
            pieces.append(token)
    return " ".join(pieces)


def _flatten_stanza_tokens(doc: Any) -> list[StanzaTokenAnnotation]:
    """Convert a Stanza document into a flat list of surface-token annotations."""
    flattened: list[StanzaTokenAnnotation] = []
    for sentence in getattr(doc, "sentences", ()):
        for token in getattr(sentence, "tokens", ()):
            token_words: list[StanzaWordAnnotation] = []
            for word in getattr(token, "words", ()):
                token_words.append(
                    StanzaWordAnnotation(
                        text=str(getattr(word, "text", "")),
                        upos=_optional_text(getattr(word, "upos", None)),
                        xpos=_optional_text(getattr(word, "xpos", None)),
                        feats=_optional_text(getattr(word, "feats", None)),
                        lemma=_optional_text(getattr(word, "lemma", None)),
                    )
                )
            flattened.append(
                StanzaTokenAnnotation(
                    text=str(getattr(token, "text", "")),
                    words=tuple(token_words),
                )
            )
    return flattened


def _optional_text(value: object) -> str | None:
    """Convert optional Stanza attributes into stripped strings."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _null_annotation_payload() -> dict[str, object]:
    """Return a consistent null POS payload."""
    return {
        "upos": pd.NA,
        "xpos": pd.NA,
        "morph": pd.NA,
        "lemma": pd.NA,
    }


def _annotation_payload(annotation: StanzaWordAnnotation) -> dict[str, object]:
    """Convert a Stanza word annotation into tabular columns."""
    return {
        "upos": annotation.upos if annotation.upos is not None else pd.NA,
        "xpos": annotation.xpos if annotation.xpos is not None else pd.NA,
        "morph": annotation.feats if annotation.feats is not None else pd.NA,
        "lemma": annotation.lemma if annotation.lemma is not None else pd.NA,
    }


def _token_annotation_payload(annotation: StanzaTokenAnnotation) -> dict[str, object]:
    """Convert a Stanza surface token into tabular columns when 1:1 mapping holds."""
    if len(annotation.words) != 1:
        raise ValueError("A token annotation payload requires exactly one underlying Stanza word.")
    return _annotation_payload(annotation.words[0])


def _join_annotation_values(values: list[str | None]) -> object:
    """Join non-empty annotation values into a deterministic composite string."""
    normalized_values = [value for value in values if value is not None and str(value).strip()]
    if len(normalized_values) == 0:
        return pd.NA
    return "+".join(str(value) for value in normalized_values)


def _composite_annotation_payload(words: list[StanzaWordAnnotation]) -> dict[str, object]:
    """Build a deterministic composite POS payload for one token split into multiple words."""
    return {
        "upos": _join_annotation_values([word.upos for word in words]),
        "xpos": _join_annotation_values([word.xpos for word in words]),
        "morph": _join_annotation_values([word.feats for word in words]),
        "lemma": _join_annotation_values([word.lemma for word in words]),
    }


def _collect_group_words(stanza_tokens: list[StanzaTokenAnnotation], start_index: int, stop_index: int) -> list[StanzaWordAnnotation]:
    """Flatten underlying words across a contiguous Stanza token span."""
    collected: list[StanzaWordAnnotation] = []
    for token_index in range(start_index, stop_index):
        collected.extend(stanza_tokens[token_index].words)
    return collected


def _assign_group_status(
    annotations: list[dict[str, object]],
    start_index: int,
    stop_index: int,
    *,
    status: str,
    note: str,
    payload: dict[str, object] | None = None,
) -> None:
    """Fill a contiguous range of output rows with a conservative mismatch marker."""
    for index in range(start_index, stop_index):
        annotations[index] = {
            **(_null_annotation_payload() if payload is None else payload),
            "mapping_status": status,
            "mapping_note": note,
        }


def _build_mapping_annotations(
    token_values: list[str],
    stanza_tokens: list[StanzaTokenAnnotation],
) -> list[dict[str, object]]:
    """Map flattened Stanza words back to the aligned token sequence."""
    annotations = [
        {
            **_null_annotation_payload(),
            "mapping_status": "unmapped_input",
            "mapping_note": "No Stanza annotation was matched to this token row.",
        }
        for _ in token_values
    ]

    normalized_tokens = [_normalize_surface(value) for value in token_values]
    normalized_stanza_tokens = [_normalize_surface(token.text) for token in stanza_tokens]

    token_index = 0
    stanza_index = 0
    while token_index < len(token_values) and stanza_index < len(stanza_tokens):
        token_surface = normalized_tokens[token_index]
        if not token_surface:
            annotations[token_index] = {
                **_null_annotation_payload(),
                "mapping_status": "missing_token_text",
                "mapping_note": "Token row has empty text after normalization.",
            }
            token_index += 1
            continue

        stanza_surface = normalized_stanza_tokens[stanza_index]
        if token_surface == stanza_surface:
            if len(stanza_tokens[stanza_index].words) == 1:
                annotations[token_index] = {
                    **_token_annotation_payload(stanza_tokens[stanza_index]),
                    "mapping_status": "exact",
                    "mapping_note": pd.NA,
                }
            else:
                annotations[token_index] = {
                    **_composite_annotation_payload(list(stanza_tokens[stanza_index].words)),
                    "mapping_status": "stanza_split",
                    "mapping_note": (
                        "One aligned token corresponds to multiple Stanza words; "
                        "composite POS values were preserved on the original token row."
                    ),
                }
            token_index += 1
            stanza_index += 1
            continue

        token_stop = token_index
        stanza_stop = stanza_index
        token_concat = token_surface
        stanza_concat = stanza_surface

        while token_concat != stanza_concat:
            if len(token_concat) <= len(stanza_concat) and token_stop + 1 < len(normalized_tokens):
                token_stop += 1
                token_concat += normalized_tokens[token_stop]
                continue
            if stanza_stop + 1 < len(normalized_stanza_tokens):
                stanza_stop += 1
                stanza_concat += normalized_stanza_tokens[stanza_stop]
                continue
            if token_stop + 1 < len(normalized_tokens):
                token_stop += 1
                token_concat += normalized_tokens[token_stop]
                continue
            break

        if token_concat == stanza_concat and token_concat:
            token_span = token_stop - token_index + 1
            stanza_span = stanza_stop - stanza_index + 1
            if token_span == 1 and stanza_span > 1:
                composite_words = _collect_group_words(stanza_tokens, stanza_index, stanza_stop + 1)
                status = "stanza_split"
                note = (
                    "One aligned token corresponds to multiple Stanza words; "
                    "composite POS values were preserved on the original token row."
                )
                payload = _composite_annotation_payload(composite_words)
            elif token_span > 1 and stanza_span == 1:
                status = "stanza_merge"
                note = (
                    "Multiple aligned tokens correspond to one Stanza word; "
                    "POS values were left empty to preserve the original token rows."
                )
                payload = None
            else:
                status = "group_mismatch"
                note = (
                    "Aligned token group and Stanza word group share the same normalized surface "
                    "but not a 1:1 word mapping; POS values were left empty."
                )
                payload = None
            _assign_group_status(
                annotations,
                token_index,
                token_stop + 1,
                status=status,
                note=note,
                payload=payload,
            )
            token_index = token_stop + 1
            stanza_index = stanza_stop + 1
            continue

        annotations[token_index] = {
            **_null_annotation_payload(),
            "mapping_status": "unmapped_input",
            "mapping_note": (
                "Token could not be reconciled with the current Stanza word sequence; "
                "original aligned row was preserved without POS values."
            ),
        }
        token_index += 1

    while token_index < len(token_values):
        if normalized_tokens[token_index]:
            annotations[token_index] = {
                **_null_annotation_payload(),
                "mapping_status": "unmapped_input",
                "mapping_note": (
                    "No remaining Stanza word was available for this token row; "
                    "original aligned row was preserved without POS values."
                ),
            }
        else:
            annotations[token_index] = {
                **_null_annotation_payload(),
                "mapping_status": "missing_token_text",
                "mapping_note": "Token row has empty text after normalization.",
            }
        token_index += 1

    return annotations


def annotate_aligned_token_pos(
    token_table: pd.DataFrame,
    *,
    nlp: Any,
    runtime: StanzaRuntimeInfo,
    fail_on_mapping_error: bool = False,
) -> StanzaPosResult:
    """Annotate an aligned token table with Stanza POS fields.

    Parameters
    ----------
    token_table
        Subject/run token table whose rows define the released alignment.
    nlp
        Initialized Stanza pipeline or a test double implementing ``__call__``.
    runtime
        Resolved Stanza runtime information stored in outputs.
    fail_on_mapping_error
        Whether to raise if any row is not mapped exactly.
    """

    text_column = _resolve_text_column(token_table)
    token_strings = token_table[text_column].fillna("").astype(str).tolist()
    reconstructed_text = _join_tokens_for_stanza(token_strings)
    doc = nlp(reconstructed_text)
    stanza_tokens = _flatten_stanza_tokens(doc)
    annotations = _build_mapping_annotations(token_strings, stanza_tokens)

    output = token_table.copy()
    if "onset" not in output.columns and "start" in output.columns:
        output["onset"] = pd.to_numeric(output["start"], errors="coerce")
    if "duration" not in output.columns:
        if "start" in output.columns and "end" in output.columns:
            output["duration"] = pd.to_numeric(output["end"], errors="coerce") - pd.to_numeric(
                output["start"], errors="coerce"
            )
        else:
            output["duration"] = pd.NA
    if "source_interval_id" not in output.columns:
        if "_source_row_index" in output.columns:
            output["source_interval_id"] = output["_source_row_index"].astype(str)
        elif "annotation_index" in output.columns:
            output["source_interval_id"] = output["annotation_index"].astype(str)
        else:
            output["source_interval_id"] = output.index.astype(str)

    annotation_frame = pd.DataFrame(annotations)
    output["upos"] = annotation_frame["upos"]
    output["xpos"] = annotation_frame["xpos"]
    output["morph"] = annotation_frame["morph"]
    output["lemma"] = annotation_frame["lemma"]
    output["pos_model"] = runtime.model_name
    output["stanza_version"] = runtime.stanza_version if runtime.stanza_version is not None else pd.NA
    output["pos_lang"] = runtime.language
    output["mapping_status"] = annotation_frame["mapping_status"]
    output["mapping_note"] = annotation_frame["mapping_note"]

    if fail_on_mapping_error and not output["mapping_status"].eq("exact").all():
        mismatch_count = int((~output["mapping_status"].eq("exact")).sum())
        raise ValueError(
            "Stanza token reconciliation produced non-exact mappings for "
            f"{mismatch_count} row(s). Inspect `mapping_status` and `mapping_note`."
        )

    return StanzaPosResult(
        event_table=output,
        runtime=runtime,
        text_column=text_column,
        reconstructed_text=reconstructed_text,
    )


def extract_stanza_pos_features(
    token_table: pd.DataFrame,
    config: StanzaPosConfig,
    *,
    nlp: Any | None = None,
    runtime: StanzaRuntimeInfo | None = None,
) -> StanzaPosResult:
    """Run Stanza POS extraction for one aligned token table."""
    active_nlp = nlp
    active_runtime = runtime
    if active_nlp is None or active_runtime is None:
        active_nlp, active_runtime = load_stanza_pos_pipeline(config)
    return annotate_aligned_token_pos(
        token_table,
        nlp=active_nlp,
        runtime=active_runtime,
        fail_on_mapping_error=config.fail_on_mapping_error,
    )


def build_pos_sidecar_payload(
    *,
    result: StanzaPosResult,
    feature_name: str,
    feature_file_path: Path,
    input_token_path: Path,
    subject: str,
    run: str,
    source_subject: str | None,
    source_role: str | None,
) -> dict[str, Any]:
    """Create a publication-oriented JSON sidecar for the POS derivative."""
    event_table = result.event_table
    missing_value_count = int(event_table.isna().sum().sum())
    mapping_counts = {
        str(key): int(value)
        for key, value in event_table["mapping_status"].value_counts(dropna=False).items()
    }
    speaker_series = event_table.get("speaker", pd.Series(dtype=object))
    speaker_levels = sorted({str(value) for value in speaker_series.dropna().unique()})

    columns: dict[str, dict[str, Any]] = {}
    for column_name in event_table.columns:
        if column_name == "onset":
            columns[column_name] = {
                "Description": "Token onset time relative to run onset.",
                "Units": "s",
                "DType": "float",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "duration":
            columns[column_name] = {
                "Description": "Token duration.",
                "Units": "s",
                "DType": "float",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "speaker":
            columns[column_name] = {
                "Description": "Speaker identifier from the aligned token source table.",
                "Units": None,
                "DType": "string",
                "Levels": speaker_levels or None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == result.text_column:
            columns[column_name] = {
                "Description": "Surface token from the aligned source table used as the release-facing row definition.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "upos":
            columns[column_name] = {
                "Description": "Universal Dependencies UPOS tag assigned by Stanza when exact token-word mapping succeeded.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "xpos":
            columns[column_name] = {
                "Description": "Language-specific POS tag exposed by Stanza when available.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "morph":
            columns[column_name] = {
                "Description": "Universal Dependencies morphological feature bundle from Stanza.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "lemma":
            columns[column_name] = {
                "Description": "Lemma emitted by Stanza when available.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "mapping_status":
            columns[column_name] = {
                "Description": "Diagnostic status describing how Stanza tokenization was reconciled with the aligned token row.",
                "Units": None,
                "DType": "string",
                "Levels": [
                    "exact",
                    "stanza_split",
                    "stanza_merge",
                    "group_mismatch",
                    "unmapped_input",
                    "missing_token_text",
                ],
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "mapping_note":
            columns[column_name] = {
                "Description": "Human-readable reconciliation note for non-exact mappings.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "source_interval_id":
            columns[column_name] = {
                "Description": "Stable identifier of the aligned input row used to generate this derivative row.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "pos_model":
            columns[column_name] = {
                "Description": "POS annotation software family used for the row.",
                "Units": None,
                "DType": "string",
                "Levels": [result.runtime.model_name],
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "stanza_version":
            columns[column_name] = {
                "Description": "Installed Stanza package version used for annotation.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }
        elif column_name == "pos_lang":
            columns[column_name] = {
                "Description": "Language code used to load the Stanza model.",
                "Units": None,
                "DType": "string",
                "Levels": [result.runtime.language],
                "MissingValues": MISSING_VALUE,
            }
        else:
            columns[column_name] = {
                "Description": "Column preserved from the aligned input token table.",
                "Units": None,
                "DType": "string",
                "Levels": None,
                "MissingValues": MISSING_VALUE,
            }

    return {
        "FeatureName": f"{feature_name}_features",
        "FeatureLabel": "Token-level POS features",
        "FeatureType": "event",
        "Description": (
            "French token-level POS, XPOS, morphology, and lemma annotations "
            "derived with Stanza and attached to the original aligned token rows."
        ),
        "TimeBase": "seconds_from_run_onset",
        "Delimiter": "tab",
        "RowDefinition": (
            "One row represents one original aligned token row for the selected "
            "subject and run."
        ),
        "Columns": columns,
        "Source": {
            "Modality": "annotation",
            "InputFiles": [str(input_token_path)],
            "InputTimeBase": "seconds",
            "AnnotationSource": "Aligned token table used as the release-facing ground truth.",
        },
        "Generation": {
            "SoftwareName": "hyperscanning",
            "SoftwareVersion": None,
            "CodeURL": None,
            "ExtractionLibrary": result.runtime.model_name,
            "ExtractionLibraryVersion": result.runtime.stanza_version,
            "ComputationDate": datetime.now(UTC).isoformat(),
        },
        "Method": {
            "Algorithm": "Stanza POS tagging with conservative token-to-word reconciliation",
            "AlgorithmReference": "https://stanfordnlp.github.io/stanza/",
            "Parameters": {
                "Language": result.runtime.language,
                "TagSet": "Universal Dependencies UPOS",
                "ModelName": result.runtime.model_name,
                "ModelSource": result.runtime.model_source,
                "Processors": list(result.runtime.processors),
                "ResourcesDir": str(result.runtime.resources_dir),
                "TokenizationPolicy": (
                    "Stanza tokenization is used internally only; released rows "
                    "preserve the aligned token table."
                ),
                "AlignmentPolicy": (
                    "The aligned token table is the ground truth and row count is "
                    "preserved exactly."
                ),
                "MappingPolicy": (
                    "Only exact one-to-one token-word matches receive POS values; "
                    "grouped or unmatched cases retain the original row with null "
                    "POS fields and diagnostics."
                ),
            },
            "Preprocessing": [
                "Aligned tokens were reconstructed into deterministic text before Stanza annotation.",
                "No released row was retokenized or reordered.",
            ],
            "Postprocessing": [
                "Stanza word annotations were reconciled back onto the original token rows.",
            ],
            "FilteringRules": [],
            "AggregationRules": [],
        },
        "QualityControl": {
            "NumRows": int(event_table.shape[0]),
            "NumMissingValues": missing_value_count,
            "ExcludedCount": 0,
            "CategoryCounts": {"mapping_status": mapping_counts},
            "TimingSorted": (
                bool(event_table["onset"].is_monotonic_increasing)
                if "onset" in event_table.columns
                else None
            ),
        },
        "FeatureFile": {
            "Path": str(feature_file_path),
            "Format": "TSV",
        },
        "Notes": [
            f"Subject={subject}, run={run}, source_subject={source_subject}, source_role={source_role}.",
            (
                "This derivative preserves the aligned token table row-for-row even "
                "when Stanza tokenization differs internally."
            ),
            f"Reconstructed text length before annotation: {len(result.reconstructed_text)} characters.",
        ],
    }
