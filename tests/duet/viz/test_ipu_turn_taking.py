"""Tests for IPU turn-taking summary helpers."""

from __future__ import annotations

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from duet.viz import ipu_turn_taking as mod


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


def test_tune_ipu_summary_layout_scales_text_and_removes_left_xticks() -> None:
    """Layout tuning should rescale text and remove right-panel x ticks."""
    fig, axes = plt.subplots(1, 3)
    for index, axis in enumerate(axes):
        axis.set_title(f"Panel {index}")
        axis.set_xlabel("X label")
        axis.set_ylabel("Y label")
        axis.plot([0.0, 1.0], [0.0, 1.0], label="Example")
        axis.legend(title="Legend")

    fig.canvas.draw()
    initial_title_size = axes[0].title.get_fontsize()
    initial_xlabel_size = axes[0].xaxis.label.get_fontsize()
    initial_legend_size = axes[0].get_legend().get_texts()[0].get_fontsize()

    mod._tune_ipu_summary_layout(fig, axes)

    assert axes[0].title.get_fontsize() == initial_title_size * mod.FONT_SCALE * mod.TITLE_SCALE
    assert axes[0].xaxis.label.get_fontsize() == initial_xlabel_size * mod.FONT_SCALE * mod.AXIS_LABEL_SCALE
    assert axes[0].get_legend().get_texts()[0].get_fontsize() == initial_legend_size * mod.FONT_SCALE * mod.LEGEND_SCALE
    assert axes[0].get_xticklabels()[0].get_fontsize() == axes[1].get_yticklabels()[0].get_fontsize()
    assert axes[0].get_yticklabels()[0].get_fontsize() == axes[1].get_yticklabels()[0].get_fontsize()
    assert axes[1].get_xticklabels()[0].get_fontsize() == axes[1].get_yticklabels()[0].get_fontsize()
    assert len(axes[2].get_xticks()) == 0
    assert len(axes[0].get_xticks()) > 0

    plt.close(fig)


def test_add_panel_labels_places_expected_ipu_annotations() -> None:
    """IPU summary panels should receive the expected corner labels."""
    fig, axes = plt.subplots(1, 3)

    mod._add_panel_labels(axes)

    assert [axis.texts[0].get_text() for axis in axes] == ["(A)", "(B)", "(C)"]
    assert [axis.texts[0].get_position() for axis in axes] == list(mod.PANEL_LABEL_POSITIONS)

    plt.close(fig)
