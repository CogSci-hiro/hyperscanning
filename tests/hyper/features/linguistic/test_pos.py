"""Unit tests for Stanza-backed token-aligned POS extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hyper.features.linguistic.pos import (
    StanzaPosConfig,
    StanzaRuntimeInfo,
    annotate_aligned_token_pos,
)
from hyper.features.pipelines.linguistic import run_token_pos_pipeline


@dataclass(frozen=True)
class _FakeWord:
    text: str
    upos: str | None = None
    xpos: str | None = None
    feats: str | None = None
    lemma: str | None = None


@dataclass(frozen=True)
class _FakeSentence:
    words: list[_FakeWord]
    tokens: list[object] | None = None


@dataclass(frozen=True)
class _FakeDoc:
    sentences: list[_FakeSentence]


@dataclass(frozen=True)
class _FakeToken:
    text: str
    words: list[_FakeWord]


class _FakePipeline:
    """Small Stanza-like callable used for deterministic tests."""

    def __init__(self, words: list[_FakeWord]) -> None:
        self.words = words
        self.last_text: str | None = None

    def __call__(self, text: str) -> _FakeDoc:
        self.last_text = text
        tokens = [_FakeToken(text=word.text, words=[word]) for word in self.words]
        return _FakeDoc(sentences=[_FakeSentence(words=self.words, tokens=tokens)])


class _FakePipelineWithTokens:
    """Stanza-like callable that lets tests control token/word divergence."""

    def __init__(self, tokens: list[_FakeToken]) -> None:
        self.tokens = tokens
        self.last_text: str | None = None

    def __call__(self, text: str) -> _FakeDoc:
        self.last_text = text
        flat_words = [word for token in self.tokens for word in token.words]
        return _FakeDoc(sentences=[_FakeSentence(words=flat_words, tokens=self.tokens)])


def _runtime() -> StanzaRuntimeInfo:
    """Return stable runtime metadata for tests."""
    return StanzaRuntimeInfo(
        model_name="stanza",
        model_source="test-double",
        stanza_version="test-1.0",
        resources_dir=Path("/tmp/stanza"),
        processors=("tokenize", "pos", "lemma"),
        language="fr",
    )


def test_annotate_aligned_token_pos_preserves_rows_and_timing() -> None:
    """Exact mappings should preserve source rows and timing columns."""
    token_table = pd.DataFrame(
        {
            "run": ["1", "1"],
            "speaker": ["A", "A"],
            "start": [0.0, 0.5],
            "end": [0.4, 0.9],
            "token": ["Bonjour", "monde"],
            "trial_id": [11, 12],
        }
    )
    fake_nlp = _FakePipeline(
        [
            _FakeWord("Bonjour", upos="INTJ", xpos="I", feats="Polite=Form", lemma="bonjour"),
            _FakeWord("monde", upos="NOUN", xpos="NC", feats="Gender=Masc|Number=Sing", lemma="monde"),
        ]
    )

    result = annotate_aligned_token_pos(
        token_table,
        nlp=fake_nlp,
        runtime=_runtime(),
    )

    assert list(result.event_table["token"]) == ["Bonjour", "monde"]
    assert list(result.event_table["trial_id"]) == [11, 12]
    assert list(result.event_table["onset"]) == [0.0, 0.5]
    assert list(result.event_table["duration"]) == [0.4, 0.4]
    assert list(result.event_table["upos"]) == ["INTJ", "NOUN"]
    assert list(result.event_table["lemma"]) == ["bonjour", "monde"]
    assert result.event_table["mapping_status"].tolist() == ["exact", "exact"]
    assert fake_nlp.last_text == "Bonjour monde"


def test_annotate_aligned_token_pos_marks_grouped_mismatches_without_dropping_rows() -> None:
    """Non-1:1 reconciliation should keep rows and emit diagnostics."""
    token_table = pd.DataFrame(
        {
            "run": ["1", "1", "1"],
            "speaker": ["A", "A", "A"],
            "start": [0.0, 0.2, 0.4],
            "end": [0.1, 0.3, 0.6],
            "token": ["aujourd", "'", "hui"],
        }
    )
    fake_nlp = _FakePipeline(
        [_FakeWord("aujourd'hui", upos="ADV", xpos="ADV", feats=None, lemma="aujourd'hui")]
    )

    result = annotate_aligned_token_pos(
        token_table,
        nlp=fake_nlp,
        runtime=_runtime(),
    )

    assert len(result.event_table) == 3
    assert result.event_table["mapping_status"].tolist() == [
        "stanza_merge",
        "stanza_merge",
        "stanza_merge",
    ]
    assert result.event_table["upos"].isna().all()
    assert result.event_table["mapping_note"].notna().all()


def test_annotate_aligned_token_pos_resynchronizes_after_multiword_token() -> None:
    """French multi-word tokens should not cause downstream reconciliation drift."""
    token_table = pd.DataFrame(
        {
            "run": ["1", "1", "1", "1"],
            "speaker": ["A", "A", "A", "A"],
            "start": [0.0, 0.2, 0.4, 0.6],
            "end": [0.1, 0.3, 0.5, 0.7],
            "token": ["du", "garçon", "qui", "cherche"],
        }
    )
    fake_nlp = _FakePipelineWithTokens(
        [
            _FakeToken(
                text="du",
                words=[
                    _FakeWord("de", upos="ADP", xpos="P", feats=None, lemma="de"),
                    _FakeWord("le", upos="DET", xpos="DET", feats=None, lemma="le"),
                ],
            ),
            _FakeToken(text="garçon", words=[_FakeWord("garçon", upos="NOUN", xpos="NC", feats=None, lemma="garçon")]),
            _FakeToken(text="qui", words=[_FakeWord("qui", upos="PRON", xpos="PRO", feats=None, lemma="qui")]),
            _FakeToken(text="cherche", words=[_FakeWord("cherche", upos="VERB", xpos="V", feats=None, lemma="chercher")]),
        ]
    )

    result = annotate_aligned_token_pos(
        token_table,
        nlp=fake_nlp,
        runtime=_runtime(),
    )

    assert result.event_table["mapping_status"].tolist() == ["stanza_split", "exact", "exact", "exact"]
    assert result.event_table.loc[0, "upos"] == "ADP+DET"
    assert result.event_table.loc[0, "lemma"] == "de+le"
    assert result.event_table.loc[1:, "upos"].tolist() == ["NOUN", "PRON", "VERB"]


def test_annotate_aligned_token_pos_preserves_composite_pos_for_split_surface_token() -> None:
    """One aligned token split by Stanza should keep a composite POS label."""
    token_table = pd.DataFrame(
        {
            "run": ["1"],
            "speaker": ["A"],
            "start": [0.0],
            "end": [0.5],
            "token": ["c'est"],
        }
    )
    fake_nlp = _FakePipelineWithTokens(
        [
            _FakeToken(text="c'", words=[_FakeWord("c'", upos="PRON", xpos="PRO", feats=None, lemma="ce")]),
            _FakeToken(text="est", words=[_FakeWord("est", upos="AUX", xpos="V", feats=None, lemma="être")]),
        ]
    )

    result = annotate_aligned_token_pos(
        token_table,
        nlp=fake_nlp,
        runtime=_runtime(),
    )

    assert result.event_table.loc[0, "mapping_status"] == "stanza_split"
    assert result.event_table.loc[0, "upos"] == "PRON+AUX"
    assert result.event_table.loc[0, "lemma"] == "ce+être"


def test_annotate_aligned_token_pos_normalizes_sppas_compound_before_stanza() -> None:
    """Known SPPAS compound tokens should be expanded before Stanza annotation."""
    token_table = pd.DataFrame(
        {
            "run": ["1"],
            "speaker": ["A"],
            "start": [0.0],
            "end": [0.5],
            "token": ["est-ce_que"],
        }
    )
    fake_nlp = _FakePipelineWithTokens(
        [
            _FakeToken(text="est", words=[_FakeWord("est", upos="AUX", xpos="V", feats=None, lemma="être")]),
            _FakeToken(text="-ce", words=[_FakeWord("-ce", upos="PRON", xpos="PRO", feats=None, lemma="ce")]),
            _FakeToken(text="que", words=[_FakeWord("que", upos="SCONJ", xpos="CS", feats=None, lemma="que")]),
        ]
    )

    result = annotate_aligned_token_pos(
        token_table,
        nlp=fake_nlp,
        runtime=_runtime(),
    )

    assert fake_nlp.last_text == "est-ce que"
    assert result.event_table.loc[0, "mapping_status"] == "stanza_split"
    assert result.event_table.loc[0, "upos"] == "AUX+PRON+SCONJ"


def test_run_token_pos_pipeline_writes_sidecar_and_preserves_filtered_rows(tmp_path: Path) -> None:
    """Pipeline wrapper should filter by subject/run, write TSV, and emit sidecar metadata."""
    tokens_path = tmp_path / "dyad-001_tokens.csv"
    pd.DataFrame(
        {
            "run": ["1", "1", "1"],
            "speaker": ["A", "A", "B"],
            "start": [0.0, 0.3, 0.5],
            "end": [0.2, 0.6, 0.8],
            "token": ["Salut", "toi", "bonjour"],
            "word": ["Salut", "toi", "bonjour"],
        }
    ).to_csv(tokens_path, index=False)

    out_tsv = tmp_path / "out" / "self_pos.tsv"
    out_json = tmp_path / "out" / "self_pos.json"
    fake_nlp = _FakePipeline(
        [
            _FakeWord("Salut", upos="INTJ", xpos="I", feats=None, lemma="salut"),
            _FakeWord("toi", upos="PRON", xpos="PRO", feats="Person=2", lemma="toi"),
        ]
    )

    output = run_token_pos_pipeline(
        tokens_path=tokens_path,
        subject="sub-001",
        run="1",
        output_tsv_path=out_tsv,
        output_sidecar_path=out_json,
        config=StanzaPosConfig(),
        feature_name="self_pos",
        nlp=fake_nlp,
        runtime=_runtime(),
    )

    written = pd.read_csv(out_tsv, sep="\t")
    sidecar = json.loads(out_json.read_text(encoding="utf-8"))

    assert len(output) == 2
    assert len(written) == 2
    assert "source_interval_id" in written.columns
    assert "_source_row_index" not in written.columns
    assert sidecar["FeatureName"] == "self_pos_features"
    assert sidecar["Method"]["Parameters"]["TagSet"] == "Universal Dependencies UPOS"
    assert sidecar["Generation"]["ExtractionLibraryVersion"] == "test-1.0"
    assert sidecar["QualityControl"]["NumRows"] == 2


def test_run_token_pos_pipeline_can_disable_progress(tmp_path: Path) -> None:
    """Pipeline wrapper should still write outputs when progress is disabled."""
    tokens_path = tmp_path / "dyad-001_tokens.csv"
    pd.DataFrame(
        {
            "run": ["1"],
            "speaker": ["A"],
            "start": [0.0],
            "end": [0.2],
            "token": ["Salut"],
        }
    ).to_csv(tokens_path, index=False)

    out_tsv = tmp_path / "out" / "self_pos.tsv"
    out_json = tmp_path / "out" / "self_pos.json"
    fake_nlp = _FakePipeline([_FakeWord("Salut", upos="INTJ", xpos="I", feats=None, lemma="salut")])

    output = run_token_pos_pipeline(
        tokens_path=tokens_path,
        subject="sub-001",
        run="1",
        output_tsv_path=out_tsv,
        output_sidecar_path=out_json,
        config=StanzaPosConfig(),
        nlp=fake_nlp,
        runtime=_runtime(),
        show_progress=False,
    )

    assert len(output) == 1
    assert out_tsv.exists()
    assert out_json.exists()
