"""Regression tests for speaker-aware acoustic event export wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hyper.features.pipelines.acoustics import run_alignment_event_pipeline, run_token_event_pipeline


def test_run_alignment_event_pipeline_records_source_role_and_inferred_speaker(tmp_path: Path) -> None:
    """Alignment-derived event exports should make the self/other source explicit."""
    alignment_path = tmp_path / "phones.csv"
    alignment_path.write_text(
        "\n".join(
            [
                '"PhonAlign",0.0,0.1,"a"',
                '"PhonAlign",0.1,0.2,"t"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_tsv = tmp_path / "self_phonemes.tsv"
    output_json = tmp_path / "self_phonemes.json"

    event_table = run_alignment_event_pipeline(
        alignment_path=alignment_path,
        tier_name="PhonAlign",
        output_tsv_path=output_tsv,
        output_sidecar_path=output_json,
        feature_name="self_phonemes",
        source_subject="sub-007",
        source_role="self",
    )

    assert list(event_table["speaker"].unique()) == ["A"]
    assert list(event_table["source_subject"].unique()) == ["sub-007"]
    assert list(event_table["source_role"].unique()) == ["self"]

    written = pd.read_csv(output_tsv, sep="\t")
    assert list(written["speaker"].unique()) == ["A"]
    sidecar = json.loads(output_json.read_text(encoding="utf-8"))
    assert sidecar["metadata"]["source_subject"] == "sub-007"
    assert sidecar["metadata"]["source_role"] == "self"


def test_run_token_event_pipeline_filters_to_requested_subject_speaker(tmp_path: Path) -> None:
    """Token exports should keep only the speaker tied to the requested subject."""
    tokens_path = tmp_path / "dyad-001_tokens.csv"
    tokens_path.write_text(
        "\n".join(
            [
                "run,token,speaker,start,end",
                "1,hello,A,0.0,0.2",
                "1,world,B,0.3,0.5",
                "2,skip,A,0.0,0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_tsv = tmp_path / "other_tokens.tsv"
    output_json = tmp_path / "other_tokens.json"

    event_table = run_token_event_pipeline(
        tokens_path=tokens_path,
        subject="sub-002",
        run="1",
        output_tsv_path=output_tsv,
        output_sidecar_path=output_json,
        feature_name="other_tokens",
        source_subject="sub-002",
        source_role="other",
    )

    assert event_table["label"].tolist() == ["world"]
    assert event_table["speaker"].tolist() == ["B"]
    assert event_table["source_subject"].tolist() == ["sub-002"]
    assert event_table["source_role"].tolist() == ["other"]
