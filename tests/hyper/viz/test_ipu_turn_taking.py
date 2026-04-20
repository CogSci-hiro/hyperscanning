"""Tests for IPU turn-taking summary helpers."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from hyper.viz import ipu_turn_taking as mod


def _ipu_table(rows: list[tuple[float, float]]) -> pd.DataFrame:
    """Build a minimal IPU table for tests."""
    return pd.DataFrame(
        {
            "start": [start for start, _ in rows],
            "end": [end for _, end in rows],
            "duration": [end - start for start, end in rows],
        }
    )


def test_segments_from_ipus_partitions_timeline_into_expected_categories() -> None:
    """Segments should label A, B, overlap, and silence correctly."""
    a_table = _ipu_table([(0.0, 1.0), (3.0, 4.0)])
    b_table = _ipu_table([(0.5, 2.0)])

    segments = mod._segments_from_ipus(a_table, b_table)

    assert [(segment.start, segment.end, segment.category) for segment in segments] == [
        (0.0, 0.5, "A"),
        (0.5, 1.0, "overlap"),
        (1.0, 2.0, "B"),
        (2.0, 3.0, "silence"),
        (3.0, 4.0, "A"),
    ]


def test_cumulative_path_ignores_silence_and_accumulates_overlap_for_both_axes() -> None:
    """Cumulative traces should move right, up, or diagonally by segment class."""
    segments = [
        mod.TurnSegment(start=0.0, end=1.0, category="A"),
        mod.TurnSegment(start=1.0, end=3.0, category="overlap"),
        mod.TurnSegment(start=3.0, end=4.5, category="silence"),
        mod.TurnSegment(start=4.5, end=5.0, category="B"),
    ]

    x_values, y_values = mod._cumulative_path(segments)

    assert np.allclose(x_values, np.array([0.0, 1.0, 3.0, 3.0]))
    assert np.allclose(y_values, np.array([0.0, 0.0, 2.0, 2.5]))
