"""Central Glad logging: one configure call, one get_logger everywhere.

Call `configure_logging()` once at process start (see `glad.main`). Modules
should only do:

    from glad.logging import get_logger
    logger = get_logger(__name__)

Do not call `logging.basicConfig` from feature code.
"""

from __future__ import annotations

import logging
import sys
import time

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | int = "INFO") -> None:
    """Idempotent root setup for the Glad process.

    Safe to call more than once; later calls only adjust the root level.
    """
    global _CONFIGURED
    resolved = _resolve_level(level)

    root = logging.getLogger()
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
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
