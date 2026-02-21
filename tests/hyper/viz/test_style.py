"""Dedicated tests for visualization style module."""

from hyper.viz.style import Style


def test_style_is_frozen_dataclass_like_in_usage() -> None:
    """Style should provide stable defaults for downstream plotting calls."""
    style = Style()
    assert style.fontsize == 12
