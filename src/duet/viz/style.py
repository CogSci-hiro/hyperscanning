from dataclasses import dataclass

import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt

REPORT_CONTEXT = "paper"
REPORT_THEME_FONT_SCALE = 1.15
REPORT_FONT_SCALE = 2.0
REPORT_X_TICK_LABEL_SCALE = 0.7
REPORT_Y_TICK_LABEL_SCALE = 1.0
REPORT_THEME_RC = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#303030",
    "axes.linewidth": 0.8,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "grid.color": "#d8d8d8",
    "grid.linewidth": 0.7,
    "xtick.color": "#202020",
    "ytick.color": "#202020",
    "axes.labelcolor": "#202020",
    "axes.titleweight": "semibold",
}


@dataclass(frozen=True, slots=True)
class Style:
    """
    Global visualization style choices.

    Usage example
    -------------
        style = Style()
        print(style.fontsize)
    """

    # Base text size used for axes labels/ticks unless overridden by caller.
    fontsize: int = 12


def apply_report_style(*, rc: dict[str, object] | None = None) -> None:
    """Apply the shared report theme used across summary figures."""
    theme_rc = dict(REPORT_THEME_RC)
    if rc:
        theme_rc.update(rc)
    sns.set_theme(
        style="whitegrid",
        context=REPORT_CONTEXT,
        font_scale=REPORT_THEME_FONT_SCALE,
        rc=theme_rc,
    )


def scale_figure_text(figure: plt.Figure, *, scale: float = REPORT_FONT_SCALE) -> None:
    """Scale all text artists in a figure by a constant factor."""
    if not hasattr(figure, "findobj"):
        return
    for text in figure.findobj(matplotlib.text.Text):
        if not hasattr(text, "get_fontsize") or not hasattr(text, "set_fontsize"):
            continue
        fontsize = text.get_fontsize()
        if fontsize is not None:
            text.set_fontsize(float(fontsize) * float(scale))


def scale_tick_labels(
    axis: plt.Axes,
    *,
    x_scale: float = REPORT_X_TICK_LABEL_SCALE,
    y_scale: float = REPORT_Y_TICK_LABEL_SCALE,
) -> None:
    """Scale axis tick labels after figure-wide text scaling."""
    if hasattr(axis, "get_xticklabels"):
        for tick in axis.get_xticklabels():
            fontsize = tick.get_fontsize()
            if fontsize is not None:
                tick.set_fontsize(float(fontsize) * float(x_scale))
    if hasattr(axis, "get_yticklabels"):
        for tick in axis.get_yticklabels():
            fontsize = tick.get_fontsize()
            if fontsize is not None:
                tick.set_fontsize(float(fontsize) * float(y_scale))
