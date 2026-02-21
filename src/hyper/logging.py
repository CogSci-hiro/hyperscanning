# ==================================================================================================
#                                   Logging
# ==================================================================================================
#
# Intentionally tiny logging bootstrap used by CLI entrypoints/scripts.
# We keep this separate so every command gets consistent formatting without
# duplicating `basicConfig(...)` snippets across modules.

import logging

def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure global logging once.

    Parameters
    ----------
    level
        Logging level.

    Usage example
    -------------
        configure_logging()
        logging.getLogger(__name__).info("hello")
    """
    # Configure root logger once. If called repeatedly, Python logging keeps
    # existing handlers unless `force=True` is used (we do not force-reset here
    # to avoid disrupting embedding environments).
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
