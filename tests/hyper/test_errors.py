"""Tests for policy-driven error handling helper functions."""

from pathlib import Path

import pytest

from hyper.errors import ErrorPolicy, run_step


def _ok_step(x: int, y: int) -> int:
    """Simple helper for success path tests."""
    return x + y


def _fail_step() -> int:
    """Simple helper for failure path tests."""
    raise RuntimeError("boom")


def test_run_step_returns_value_on_success(tmp_path: Path) -> None:
    """Successful execution should populate `value` and no `failure`."""
    policy = ErrorPolicy(debug=False, log_path=tmp_path / "run.log")

    result = run_step(policy, "add", {"case": "ok"}, _ok_step, 1, 2)

    assert result.value == 3
    assert result.failure is None


def test_run_step_captures_failure_when_not_debug(tmp_path: Path) -> None:
    """Non-debug mode should capture structured failure metadata."""
    log_path = tmp_path / "run.log"
    policy = ErrorPolicy(debug=False, log_path=log_path)

    result = run_step(policy, "explode", {"case": "fail"}, _fail_step)

    assert result.value is None
    assert result.failure is not None
    assert result.failure.step == "explode"
    assert result.failure.exc_type == "RuntimeError"
    assert log_path.exists()


def test_run_step_reraises_in_debug_mode(tmp_path: Path) -> None:
    """Debug mode should preserve fail-fast behavior."""
    policy = ErrorPolicy(debug=True, log_path=tmp_path / "run.log")

    with pytest.raises(RuntimeError, match="boom"):
        run_step(policy, "explode", {"case": "debug"}, _fail_step)
