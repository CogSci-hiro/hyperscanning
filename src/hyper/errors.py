"""
Shared error-handling primitives for pipeline-style execution.

This module standardizes two execution modes:
1) Debug mode: fail fast and re-raise exceptions immediately.
2) Run mode: capture structured failure details and return them to the caller.

Keeping this policy centralized avoids ad-hoc try/except blocks in each command
module and makes failures easier to log, inspect, and test.
"""

import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar

# ==================================================================================================
#                                   TYPES
# ==================================================================================================

T = TypeVar("T")


@dataclass(frozen=True)
class ErrorPolicy:
    """
    Error handling policy.

    Attributes
    ----------
    debug : bool
        If True, exceptions are re-raised (fail-fast).
        If False, exceptions are logged and the pipeline continues.
    log_path : Path
        Where to write logs (file logger).
    """

    debug: bool
    log_path: Path


@dataclass(frozen=True)
class StepFailure:
    """
    Structured failure record for non-debug runs.

    Attributes
    ----------
    step : str
        Name of the step that failed.
    context : dict[str, Any]
        Useful metadata (paper_id, file paths, parameters, etc.).
    exc_type : str
        Exception class name.
    message : str
        Exception message.
    traceback : str
        Full traceback.
    timestamp_utc : str
        ISO timestamp.
    """

    step: str
    context: dict[str, Any]
    exc_type: str
    message: str
    traceback: str
    timestamp_utc: str


@dataclass(frozen=True)
class StepResult(Generic[T]):
    """
    Result wrapper: either value or failure.

    Usage example
    -------------
        result = run_step(policy, "parse_pdf", {"paper_id": "p1"}, parse_pdf, pdf_path)
        if result.failure is not None:
            # handle failure
            ...
        else:
            text = result.value
    """

    value: Optional[T]
    failure: Optional[StepFailure]


# ==================================================================================================
#                                   HELPERS
# ==================================================================================================

def make_logger(*, log_path: Path) -> logging.Logger:
    """
    Return a file-backed logger used by pipeline steps.

    The function is idempotent for a given path: it avoids attaching duplicate
    handlers when called repeatedly in long-running processes or tests.
    """
    # Ensure the log directory exists before creating the file handler.
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times.
    # This keeps log output single-lined instead of duplicated N times.
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ==================================================================================================
#                                   CORE LOGIC
# ==================================================================================================

def run_step(
    policy: ErrorPolicy,
    step: str,
    context: dict[str, Any],
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> StepResult[T]:
    """
    Run a pipeline step with policy-controlled error handling.

    In debug mode, re-raises exceptions to halt immediately.
    In run mode, logs the failure and returns StepResult(value=None, failure=...).

    Usage example
    -------------
        policy = ErrorPolicy(debug=False, log_path=Path("run.log"))
        res = run_step(policy, "load_data", {"subject": "sub-001"}, load_data, path)
        if res.failure:
            # continue / skip
            pass
    """
    # Build/get the pipeline logger once per invocation boundary.
    logger = make_logger(log_path=policy.log_path)

    try:
        # Run the wrapped step and return its value on success.
        value = func(*args, **kwargs)
        return StepResult(value=value, failure=None)
    except Exception as exc:  # noqa: BLE001 (intentional: boundary catch)
        # Capture full traceback text first; this is useful for both logs and
        # structured diagnostics returned to orchestrators/callers.
        tb = traceback.format_exc()
        failure = StepFailure(
            step=step,
            context=context,
            exc_type=type(exc).__name__,
            message=str(exc),
            traceback=tb,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        # Human-readable log line + JSON payload (best of both worlds)
        logger.error("%s failed | %s: %s", step, failure.exc_type, failure.message)
        logger.error("context=%s", json.dumps(context, ensure_ascii=False))
        logger.error("traceback=%s", tb)

        if policy.debug:
            # Debug mode should halt at first failure with original exception.
            raise

        # Non-debug mode returns a structured failure so callers can continue
        # batch execution while retaining detailed failure metadata.
        return StepResult(value=None, failure=failure)
