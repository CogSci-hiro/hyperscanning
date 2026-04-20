"""Unit tests for vowel-centered formant event extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hyper.features.acoustic import formants as mod
from hyper.features.acoustic.formants import (
    FormantEventExtractionConfig,
    extract_vowel_formant_events,
    load_vowel_intervals_from_palign_csv,
    load_vowel_intervals_from_textgrid,
)


def test_load_vowel_intervals_from_textgrid_filters_to_vowels(tmp_path: Path) -> None:
    """TextGrid loading should retain vowel intervals and skip non-vowels."""
    textgrid_path = tmp_path / "phones.TextGrid"
    textgrid_path.write_text(
        "\n".join(
            [
                'item [1]:',
                '    name = "phones"',
                "    intervals [1]:",
                "        xmin = 0",
                "        xmax = 0.1",
                '        text = "a"',
                "    intervals [2]:",
                "        xmin = 0.1",
                "        xmax = 0.2",
                '        text = "t"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    intervals = load_vowel_intervals_from_textgrid(str(textgrid_path), tier_name="phones", speaker="A")

    assert len(intervals) == 1
    assert intervals[0].vowel_label == "a"
    assert intervals[0].speaker == "A"


def test_load_vowel_intervals_from_palign_csv_filters_to_vowels(tmp_path: Path) -> None:
    """palign CSV loading should retain only vowel rows from the requested tier."""
    csv_path = tmp_path / "phones.csv"
    csv_path.write_text(
        "\n".join(
            [
                '"PhonAlign",0.0,0.1,"a"',
                '"PhonAlign",0.1,0.2,"t"',
                '"OtherTier",0.2,0.3,"e"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    intervals = load_vowel_intervals_from_palign_csv(csv_path, tier_name="PhonAlign", speaker="A")

    assert len(intervals) == 1
    assert intervals[0].vowel_label == "a"
    assert intervals[0].speaker == "A"


def test_vowel_median_computation_uses_mocked_tracks(monkeypatch) -> None:
    """Formant medians should be computed per vowel interval from VoxAtlas tracks."""
    mocked_tracks = pd.DataFrame(
        [
            {"phoneme_id": "1", "F1": 300.0, "F2": 2200.0},
            {"phoneme_id": "1", "F1": 340.0, "F2": 2100.0},
        ]
    )
    monkeypatch.setattr(mod, "_compute_interval_formant_tracks", lambda **kwargs: mocked_tracks)

    from hyper.features.acoustic.formants import VowelInterval

    result = extract_vowel_formant_events(
        waveform=np.zeros(1600, dtype=np.float32),
        audio_sampling_rate_hz=16000,
        vowel_intervals=[VowelInterval(interval_id="1", onset_seconds=0.0, offset_seconds=0.08, vowel_label="a", speaker="A")],
        config=FormantEventExtractionConfig(),
    )

    row = result.event_table.iloc[0]
    assert row["extraction_status"] == "ok"
    assert float(row["f1_median_hz"]) == 320.0
    assert float(row["f2_median_hz"]) == 2150.0
    assert result.metadata.voxatlas_function.endswith("_compute_interval_formant_tracks")


def test_failed_formant_estimation_is_marked_in_output(monkeypatch) -> None:
    """Intervals with no stable formants should be marked instead of crashing."""
    monkeypatch.setattr(mod, "_compute_interval_formant_tracks", lambda **kwargs: pd.DataFrame(columns=["phoneme_id", "F1", "F2"]))

    from hyper.features.acoustic.formants import VowelInterval

    result = extract_vowel_formant_events(
        waveform=np.zeros(1600, dtype=np.float32),
        audio_sampling_rate_hz=16000,
        vowel_intervals=[VowelInterval(interval_id="1", onset_seconds=0.0, offset_seconds=0.08, vowel_label="a")],
        config=FormantEventExtractionConfig(),
    )

    row = result.event_table.iloc[0]
    assert row["extraction_status"] == "failed"
    assert "stable" in row["notes"]


def test_formant_sidecar_metadata_is_complete(monkeypatch) -> None:
    """Metadata should expose extraction parameters and output shape."""
    monkeypatch.setattr(mod, "_compute_interval_formant_tracks", lambda **kwargs: pd.DataFrame(columns=["phoneme_id", "F1", "F2"]))

    from hyper.features.acoustic.formants import VowelInterval

    result = extract_vowel_formant_events(
        waveform=np.zeros(800, dtype=np.float32),
        audio_sampling_rate_hz=16000,
        vowel_intervals=[VowelInterval(interval_id="1", onset_seconds=0.0, offset_seconds=0.02, vowel_label="a")],
        config=FormantEventExtractionConfig(min_interval_duration_seconds=0.03),
    )

    assert result.metadata.audio_sampling_rate_hz == 16000
    assert "min_interval_duration_seconds" in result.metadata.extraction_parameters
    assert result.metadata.shape == tuple(result.event_table.shape)
