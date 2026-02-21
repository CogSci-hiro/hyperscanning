"""Tests for global logging bootstrap behavior."""

import logging
from typing import Any

from hyper.logging import configure_logging


def test_configure_logging_calls_basic_config_with_expected_arguments(monkeypatch) -> None:
    """Bootstrap should delegate to `logging.basicConfig` with expected args."""
    captured: dict[str, Any] = {}

    def _fake_basic_config(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", _fake_basic_config)

    configure_logging(level=logging.DEBUG)

    assert captured["level"] == logging.DEBUG
    assert "%(asctime)s" in captured["format"]
