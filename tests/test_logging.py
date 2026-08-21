"""Left-side ENGAGED / DORMANT badge on log lines."""

from __future__ import annotations

import logging

import pytest

from glad.logging import GladFormatter, set_engagement


@pytest.fixture(autouse=True)
def _reset_engagement() -> None:
    set_engagement(False)
    yield
    set_engagement(False)


def _record(message: str = "hello") -> logging.LogRecord:
    return logging.LogRecord(
        name="glad.orchestrator",
        level=logging.INFO,
        pathname="orchestrator.py",
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_dormant_badge_is_on_the_left_without_color() -> None:
    set_engagement(False)
    formatted = GladFormatter(color=False).format(_record())
    assert " DORMANT " in formatted
    assert formatted.index(" DORMANT ") < formatted.index("hello")
    assert "\033[" not in formatted


def test_engaged_badge_is_on_the_left_and_green_when_colored() -> None:
    set_engagement(True)
    formatted = GladFormatter(color=True).format(_record())
    assert "ENGAGED" in formatted
    assert formatted.index("ENGAGED") < formatted.index("hello")
    assert "38;2;140;255;160" in formatted
    assert "DORMANT" not in formatted


def test_engagement_badge_is_shared_across_loggers() -> None:
    """Gemini's receive loop and inbound audio are different tasks; the
    badge must not be task-local or 'Gemini said' lines stay DORMANT."""
    set_engagement(True)
    other = logging.LogRecord(
        name="glad.live.session",
        level=logging.INFO,
        pathname="session.py",
        lineno=1,
        msg="Gemini said: Hello",
        args=(),
        exc_info=None,
    )
    formatted = GladFormatter(color=False).format(other)
    assert " ENGAGED " in formatted
    assert "DORMANT" not in formatted
