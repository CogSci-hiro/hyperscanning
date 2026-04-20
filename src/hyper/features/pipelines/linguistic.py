"""Pipeline wrappers for linguistic feature derivatives."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd

from hyper.features.pipelines.acoustics import ALIGNMENT_EVENT_COLUMNS, infer_dyad_index_and_speaker
from hyper.features.linguistic.pos import (
    DEFAULT_FEATURE_NAME,
    MISSING_VALUE,
    StanzaPosConfig,
    StanzaPosResult,
    StanzaRuntimeInfo,
    build_pos_sidecar_payload,
    extract_stanza_pos_features,
)

JSON_INDENT_SPACES: int = 2
POS_PROGRESS_TOTAL_STEPS: int = 4
FUNCTION_WORD_UPOS: frozenset[str] = frozenset({"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"})
CONTENT_WORD_UPOS: frozenset[str] = frozenset({"ADJ", "ADV", "INTJ", "NOUN", "NUM", "PROPN", "VERB"})


def _progress_context(*, enabled: bool, description: str):
    """Return a small tqdm progress bar or a no-op context manager."""
    if not enabled:
        return nullcontext(None)

    from tqdm import tqdm

    return tqdm(total=POS_PROGRESS_TOTAL_STEPS, desc=description, unit="step")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable UTF-8 JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=JSON_INDENT_SPACES, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_word_label_column(table: pd.DataFrame) -> str | None:
    """Return the preferred surface-form column for event labels."""
    for column_name in ("token", "word", "label"):
        if column_name in table.columns:
            return column_name
    return None


def _split_upos_tags(value: object) -> tuple[str, ...]:
    """Split a UPOS payload into deterministic component tags."""
    if pd.isna(value):
        return ()
    return tuple(tag.strip() for tag in str(value).split("+") if tag.strip())


def _word_class_membership(upos_value: object, *, word_class: str) -> bool:
    """Return whether one UPOS payload belongs to the requested word class."""
    upos_tags = _split_upos_tags(upos_value)
    if not upos_tags:
        return False

    allowed_tags = FUNCTION_WORD_UPOS if word_class == "function" else CONTENT_WORD_UPOS
    return all(tag in allowed_tags for tag in upos_tags)


def _load_subject_run_tokens(
    *,
    tokens_path: Path,
    subject: str,
    run: str,
    exclude_labels: tuple[str, ...],
) -> pd.DataFrame:
    """Load the canonical aligned token rows for one subject and run."""
    _, speaker = infer_dyad_index_and_speaker(subject)
    token_df = pd.read_csv(tokens_path)

    missing_columns = [column for column in ("run", "speaker") if column not in token_df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Input token table is missing required columns: {missing_text}")

    filtered = token_df.loc[
        (token_df["run"].astype(str) == str(run))
        & (token_df["speaker"].astype(str) == speaker)
    ].copy()
    if "token" in filtered.columns and exclude_labels:
        filtered = filtered.loc[~filtered["token"].astype(str).isin(set(exclude_labels))].copy()

    filtered = filtered.reset_index(drop=False).rename(columns={"index": "_source_row_index"})
    return filtered


def run_token_pos_pipeline(
    *,
    tokens_path: Path,
    subject: str,
    run: str,
    output_tsv_path: Path,
    output_sidecar_path: Path,
    config: StanzaPosConfig,
    feature_name: str = DEFAULT_FEATURE_NAME,
    exclude_labels: tuple[str, ...] = (),
    source_subject: str | None = None,
    source_role: str | None = None,
    nlp: Any | None = None,
    runtime: StanzaRuntimeInfo | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Generate a token-aligned Stanza POS derivative and write it to disk."""
    progress_description = f"POS {feature_name} {subject} run-{run}"
    with _progress_context(enabled=show_progress, description=progress_description) as progress:
        token_table = _load_subject_run_tokens(
            tokens_path=tokens_path,
            subject=subject,
            run=run,
            exclude_labels=exclude_labels,
        )
        if progress is not None:
            progress.update(1)

        result = extract_stanza_pos_features(
            token_table,
            config,
            nlp=nlp,
            runtime=runtime,
        )
        if progress is not None:
            progress.update(1)

        output_table = result.event_table.copy()
        if "_source_row_index" in output_table.columns:
            output_table["source_interval_id"] = output_table["_source_row_index"].astype(str)
            output_table = output_table.drop(columns="_source_row_index")
        output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
        output_table.to_csv(output_tsv_path, sep="\t", index=False, na_rep=MISSING_VALUE)
        if progress is not None:
            progress.update(1)

        cleaned_result = StanzaPosResult(
            event_table=output_table,
            runtime=result.runtime,
            text_column=result.text_column,
            reconstructed_text=result.reconstructed_text,
        )
        sidecar_payload = build_pos_sidecar_payload(
            result=cleaned_result,
            feature_name=feature_name,
            feature_file_path=output_tsv_path,
            input_token_path=tokens_path,
            subject=subject,
            run=str(run),
            source_subject=source_subject,
            source_role=source_role,
        )
        _write_json(output_sidecar_path, sidecar_payload)
        if progress is not None:
            progress.update(1)

    return output_table


def run_word_class_event_pipeline(
    *,
    pos_features_path: Path,
    subject: str,
    run: str,
    word_class: str,
    output_tsv_path: Path,
    output_sidecar_path: Path,
    feature_name: str,
    source_subject: str | None = None,
    source_role: str | None = None,
) -> pd.DataFrame:
    """Filter token-level POS rows into binary onset-coded word-class events."""
    if word_class not in {"function", "content"}:
        raise ValueError(f"word_class must be 'function' or 'content', got {word_class!r}.")

    _, speaker = infer_dyad_index_and_speaker(subject)
    pos_df = pd.read_csv(pos_features_path, sep="\t")
    required_columns = {"onset", "upos"}
    missing_columns = sorted(required_columns.difference(pos_df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"POS feature table is missing required columns: {missing_text}")

    filtered = pos_df.loc[
        (pos_df["upos"].map(lambda value: _word_class_membership(value, word_class=word_class)))
        & (pos_df["speaker"].astype(str) == speaker)
    ].copy() if "speaker" in pos_df.columns else pos_df.loc[
        pos_df["upos"].map(lambda value: _word_class_membership(value, word_class=word_class))
    ].copy()

    label_column = _resolve_word_label_column(filtered)
    label_values = (
        filtered[label_column].astype(str)
        if label_column is not None
        else pd.Series([""] * len(filtered), index=filtered.index, dtype="string")
    )
    source_interval_values = (
        filtered["source_interval_id"].astype(str)
        if "source_interval_id" in filtered.columns
        else filtered.index.astype(str)
    )
    duration_values = (
        pd.to_numeric(filtered["duration"], errors="coerce")
        if "duration" in filtered.columns
        else pd.Series(pd.NA, index=filtered.index, dtype="Float64")
    )

    event_table = pd.DataFrame(
        {
            "onset_seconds": pd.to_numeric(filtered["onset"], errors="coerce"),
            "duration_seconds": duration_values,
            "label": label_values,
            "speaker": speaker,
            "source_subject": source_subject if source_subject is not None else subject,
            "source_role": source_role,
            "source_interval_id": source_interval_values,
        },
        columns=ALIGNMENT_EVENT_COLUMNS,
    )
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    event_table.to_csv(output_tsv_path, sep="\t", index=False, na_rep=MISSING_VALUE)

    allowed_upos = sorted(FUNCTION_WORD_UPOS if word_class == "function" else CONTENT_WORD_UPOS)
    _write_json(
        output_sidecar_path,
        {
            "metadata": {
                "feature_name": feature_name,
                "alignment_target": "event_onsets",
                "source_pos_path": str(pos_features_path),
                "source_tier": "pos",
                "subject": subject,
                "run": str(run),
                "speaker": speaker,
                "word_class": word_class,
                "allowed_upos": allowed_upos,
                "source_subject": source_subject if source_subject is not None else subject,
                "source_role": source_role,
                "shape": list(event_table.shape),
            }
        },
    )
    return event_table
