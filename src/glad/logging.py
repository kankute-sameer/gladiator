"""Central Glad logging: one configure call, one get_logger everywhere.

Call `configure_logging()` once at process start (see `glad.main`). Modules
should only do:

    from glad.logging import get_logger
    logger = get_logger(__name__)

Do not call `logging.basicConfig` from feature code.
"""

from __future__ import annotations

import logging
import os
import sys
import time

_CONFIGURED = False

_DEFAULT_DATEFMT = "%H:%M:%S"

# Left-column badge: ENGAGED (glowing green) or DORMANT (red).
# Module-level, not a ContextVar — inbound websockets and the Gemini
# receive loop are different asyncio tasks, and each task would otherwise
# keep a stale copy (Gemini talking while the badge still said DORMANT).
_engaged = False

_RESET = "\033[0m"
_DIM = "\033[90m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[90m",
    logging.INFO: "\033[36m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}
# Bold bright-green on deep green — reads as a glow in most terminals.
_ENGAGED_BADGE = "\033[1;38;2;140;255;160;48;2;0;70;25m ENGAGED \033[0m"
# Bold white on deep red.
_DORMANT_BADGE = "\033[1;38;2;255;230;230;48;2;150;10;20m DORMANT \033[0m"
_PLAIN_ENGAGED = " ENGAGED "
_PLAIN_DORMANT = " DORMANT "


def set_engagement(engaged: bool) -> None:
    """Drive the left-side ENGAGED / DORMANT badge on every log line."""
    global _engaged
    _engaged = engaged


def _want_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stderr.isatty()


class GladFormatter(logging.Formatter):
    """Timestamp, colored level, then a left-side engagement badge."""

    def __init__(self, *, color: bool, datefmt: str = _DEFAULT_DATEFMT) -> None:
        super().__init__(datefmt=datefmt)
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        created = self.formatTime(record, self.datefmt)
        level = f"{record.levelname:<7}"
        name = record.name
        message = record.getMessage()
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{record.exc_text}"
        badge = self._badge()
        if self._color:
            created = f"{_DIM}{created}{_RESET}"
            level_color = _LEVEL_COLORS.get(record.levelno, "")
            level = f"{level_color}{level}{_RESET}" if level_color else level
            name = f"{_DIM}{name}{_RESET}"
        return f"{created} {level} {badge} [{name}] {message}"

    def _badge(self) -> str:
        engaged = _engaged
        if not self._color:
            return _PLAIN_ENGAGED if engaged else _PLAIN_DORMANT
        return _ENGAGED_BADGE if engaged else _DORMANT_BADGE


def configure_logging(level: str | int = "INFO") -> None:
    """Idempotent root setup for the Glad process.

    Safe to call more than once; later calls only adjust the root level.
    """
    global _CONFIGURED
    resolved = _resolve_level(level)

    root = logging.getLogger()
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(GladFormatter(color=_want_color()))
        root.handlers.clear()
        root.addHandler(handler)
        # Keep third-party noise down unless the process is in DEBUG.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("google_genai").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
        _CONFIGURED = True

    root.setLevel(resolved)
    # Our package always follows the configured level.
    logging.getLogger("glad").setLevel(resolved)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Prefer `get_logger(__name__)` at module scope."""
    return logging.getLogger(name)


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ValueError(f"Unknown log level: {level!r}")
    return resolved


class LogThrottle:
    """Rate-limit identical or high-frequency warnings (e.g. audio frame drops).

    Usage::

        _drop_throttle = LogThrottle(interval_s=2.0)

        if _drop_throttle.should_log():
            logger.warning(
                "Dropping frames (%d since last log)",
                _drop_throttle.take_count(),
            )
    """

    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self._last_at = 0.0
        self.count = 0

    def should_log(self) -> bool:
        """Record one occurrence; return True when a log line should be emitted."""
        self.count += 1
        now = time.monotonic()
        if now - self._last_at >= self.interval_s:
            self._last_at = now
            return True
        return False

    def reset(self) -> None:
        self.count = 0
        self._last_at = 0.0

    def take_count(self) -> int:
        """Return and clear the occurrence count since the last take."""
        n = self.count
        self.count = 0
        return n
