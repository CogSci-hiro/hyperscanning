"""Unit tests for pure IPU derivation logic from token intervals.

These tests validate core conversational segmentation semantics independent of
file I/O and CLI wiring.
"""

import pytest

from hyper.annotations.palign_core import (
    Interval,
    _is_token_in_ipu,
    apply_min_ipu_and_render_full_tier,
    build_ipu_segments_from_tokens,
)


def test_is_token_in_ipu_respects_special_flags() -> None:
    """Special labels should be governed by include_* flags."""
    assert _is_token_in_ipu("hello", include_laughter=False, include_noise=False, include_filled_pause=False)
    assert not _is_token_in_ipu("#", include_laughter=True, include_noise=True, include_filled_pause=True)
    assert _is_token_in_ipu("@", include_laughter=True, include_noise=False, include_filled_pause=False)
    assert not _is_token_in_ipu("@", include_laughter=False, include_noise=False, include_filled_pause=False)


def test_build_ipu_segments_merges_short_silences() -> None:
    """Adjacent speech runs should merge when separating gap is below threshold."""
    tokens = [
        Interval(0.0, 0.5, "a"),
        Interval(0.5, 0.6, "#"),
        Interval(0.6, 1.0, "b"),
    ]

    segments = build_ipu_segments_from_tokens(tokens, min_silence_s=0.2)

    assert segments == [(0.0, 1.0)]


def test_build_ipu_segments_rejects_negative_silence() -> None:
    """Silence threshold is a duration and must be non-negative."""
    with pytest.raises(ValueError, match="min_silence_s"):
        build_ipu_segments_from_tokens([], min_silence_s=-0.1)


def test_apply_min_ipu_and_render_full_tier_covers_full_timeline() -> None:
    """Rendered tier should alternate silence/IPU and cover [t_start, t_end]."""
    out = apply_min_ipu_and_render_full_tier(
        t_start=0.0,
        t_end=2.0,
        ipu_segments=[(0.5, 1.0), (1.5, 1.52)],
        min_ipu_s=0.05,
    )

    assert out[0].text == "#"
    assert out[1].text == "IPU"
    assert out[-1].end == 2.0
    # Short second IPU is dropped, so trailing silence starts at 1.0.
    assert out[-1].start == 1.0
