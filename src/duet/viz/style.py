# ==================================================================================================
#                               Plot styling
# ==================================================================================================
#
# Minimal styling container for plotting modules. Keep this as a dataclass so
# defaults remain explicit and easy to override in a controlled way.

from dataclasses import dataclass


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
