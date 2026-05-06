"""Regression tests for speaker-aware acoustic event export wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from duet.features.acoustic.common import ContinuousFeatureMetadata, ContinuousFeatureResult
from duet.features.pipelines.acoustics import (
    run_alignment_event_pipeline,
    run_envelope_pipeline,
    run_token_event_pipeline,
    zero_waveform_for_ipu_silences,
)


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


def test_zero_waveform_for_ipu_silences_masks_hash_intervals(tmp_path: Path) -> None:
    """Silence-labelled IPU rows should zero out the corresponding waveform span."""
    ipu_path = tmp_path / "sub-001_run-1_ipu.csv"
    pd.DataFrame(
        {
            "start": [0.0, 0.2, 0.6],
            "end": [0.2, 0.6, 1.0],
            "annotation": ["speech", "#", "speech"],
        }
    ).to_csv(ipu_path, index=False)

    waveform = np.arange(10, dtype=np.float32)
    masked_waveform, metadata = zero_waveform_for_ipu_silences(
        waveform,
        audio_sampling_rate_hz=10,
        ipu_path=ipu_path,
    )

    np.testing.assert_array_equal(masked_waveform[:2], waveform[:2])
    np.testing.assert_array_equal(masked_waveform[2:6], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(masked_waveform[6:], waveform[6:])
    assert metadata["silence_interval_count"] == 1
    assert metadata["silence_sample_count"] == 4


def test_run_envelope_pipeline_masks_silence_before_extraction(monkeypatch, tmp_path: Path) -> None:
    """Envelope extraction should receive silence-masked audio when IPU rows mark '#' spans."""
    waveform = np.arange(10, dtype=np.float32)
    captured: dict[str, np.ndarray] = {}

    ipu_path = tmp_path / "sub-001_run-1_ipu.csv"
    pd.DataFrame(
        {
            "start": [0.0, 0.3, 0.7],
            "end": [0.3, 0.7, 1.0],
            "annotation": ["speech", "#", "speech"],
        }
    ).to_csv(ipu_path, index=False)

    monkeypatch.setattr(
        "duet.features.pipelines.acoustics.load_audio_waveform",
        lambda audio_path: (waveform.copy(), 10),
    )

    def _fake_extract(**kwargs):
        captured["waveform"] = np.asarray(kwargs["waveform"], dtype=np.float32)
        return ContinuousFeatureResult(
            raw_time_seconds=np.array([0.0, 0.1], dtype=np.float32),
            raw_values=np.array([1.0, 2.0], dtype=np.float32),
            eeg_time_seconds=np.array([0.0, 0.5], dtype=np.float32),
            eeg_values=np.array([3.0, 4.0], dtype=np.float32),
            metadata=ContinuousFeatureMetadata(
                feature_name="envelope",
                audio_sampling_rate_hz=10,
                eeg_sampling_rate_hz=2.0,
                eeg_sample_count=2,
                extraction_parameters={},
                voxatlas_version=None,
                voxatlas_function="fake",
                resampling_method="linear",
                alignment_target="eeg_samples",
                units="amplitude",
                shape=(2,),
                notes=[],
            ),
        )

    monkeypatch.setattr("duet.features.pipelines.acoustics.extract_envelope_feature", _fake_extract)

    output_values = tmp_path / "envelope.npy"
    output_sidecar = tmp_path / "envelope.json"
    run_envelope_pipeline(
        audio_path=tmp_path / "audio.wav",
        eeg_sampling_rate_hz=2.0,
        eeg_sample_count=2,
        output_values_path=output_values,
        output_sidecar_path=output_sidecar,
        ipu_path=ipu_path,
    )

    np.testing.assert_array_equal(captured["waveform"][:3], waveform[:3])
    np.testing.assert_array_equal(captured["waveform"][3:7], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(captured["waveform"][7:], waveform[7:])

    sidecar = json.loads(output_sidecar.read_text(encoding="utf-8"))
    assert sidecar["metadata"]["silence_masking"]["applied"] is True
    assert sidecar["metadata"]["silence_masking"]["silence_interval_count"] == 1
