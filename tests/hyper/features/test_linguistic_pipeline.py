"""Regression tests for linguistic event export wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hyper.features.pipelines.linguistic import run_word_class_event_pipeline


def test_run_word_class_event_pipeline_filters_function_and_content_words(tmp_path: Path) -> None:
    """Word-class exports should retain only rows whose UPOS tags fully match the requested class."""
    pos_path = tmp_path / "self_pos.tsv"
    pd.DataFrame(
        [
            {
                "onset": 0.1,
                "duration": 0.2,
                "token": "le",
                "upos": "DET",
                "speaker": "A",
                "source_interval_id": "0",
            },
            {
                "onset": 0.4,
                "duration": 0.3,
                "token": "chat",
                "upos": "NOUN",
                "speaker": "A",
                "source_interval_id": "1",
            },
            {
                "onset": 0.8,
                "duration": 0.2,
                "token": "du",
                "upos": "ADP+DET",
                "speaker": "A",
                "source_interval_id": "2",
            },
            {
                "onset": 1.1,
                "duration": 0.2,
                "token": "va",
                "upos": "VERB",
                "speaker": "A",
                "source_interval_id": "3",
            },
            {
                "onset": 1.6,
                "duration": 0.2,
                "token": "mixte",
                "upos": "DET+NOUN",
                "speaker": "A",
                "source_interval_id": "4",
            },
            {
                "onset": 2.0,
                "duration": 0.2,
                "token": "toi",
                "upos": "PRON",
                "speaker": "B",
                "source_interval_id": "5",
            },
        ]
    ).to_csv(pos_path, sep="\t", index=False)

    function_tsv = tmp_path / "self_function.tsv"
    function_json = tmp_path / "self_function.json"
    function_table = run_word_class_event_pipeline(
        pos_features_path=pos_path,
        subject="sub-001",
        run="1",
        word_class="function",
        output_tsv_path=function_tsv,
        output_sidecar_path=function_json,
        feature_name="self_function_words",
        source_subject="sub-001",
        source_role="self",
    )

    assert function_table["label"].tolist() == ["le", "du"]
    assert function_table["speaker"].tolist() == ["A", "A"]
    assert function_table["onset_seconds"].tolist() == [0.1, 0.8]

    content_tsv = tmp_path / "self_content.tsv"
    content_json = tmp_path / "self_content.json"
    content_table = run_word_class_event_pipeline(
        pos_features_path=pos_path,
        subject="sub-001",
        run="1",
        word_class="content",
        output_tsv_path=content_tsv,
        output_sidecar_path=content_json,
        feature_name="self_content_words",
        source_subject="sub-001",
        source_role="self",
    )

    assert content_table["label"].tolist() == ["chat", "va"]
    assert content_table["source_interval_id"].tolist() == ["1", "3"]

    function_sidecar = json.loads(function_json.read_text(encoding="utf-8"))
    assert function_sidecar["metadata"]["word_class"] == "function"
    assert function_sidecar["metadata"]["source_pos_path"] == str(pos_path)
