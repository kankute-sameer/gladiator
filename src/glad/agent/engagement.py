"""Derived engagement: DORMANT unless a recent timestamp says otherwise.

No stored boolean -- a skipped `stay_engaged` expires on its own. Failure
always resolves toward silence (recoverable by the wake word).
"""

from __future__ import annotations

import time
from collections.abc import Callable

DEFAULT_TTL_S = 10.0
DEFAULT_HARD_CAP_S = 120.0


class EngagementState:
    """TTL + hard-cap engagement. `is_engaged()` is computed from timestamps."""

    def __init__(
        self,
        *,
        ttl_s: float = DEFAULT_TTL_S,
        hard_cap_s: float = DEFAULT_HARD_CAP_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_s = ttl_s
        self.hard_cap_s = hard_cap_s
        self._clock = clock
        self.last_engaged_at: float | None = None
        self.engaged_by: str | None = None
        self.hard_cap_started_at: float | None = None
        self.last_close_reason: str | None = None

    def is_engaged(self, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        if self.last_engaged_at is None:
            return False
        if now - self.last_engaged_at >= self.ttl_s:
            return False
        if self.hard_cap_started_at is not None and now - self.hard_cap_started_at >= self.hard_cap_s:
            return False
        return True

    def extend(self, reason: str, now: float | None = None) -> bool:
        """Refresh the TTL. Returns True if this call opened engagement."""
        now = self._clock() if now is None else now
        opened = not self.is_engaged(now)
        if opened:
            self.hard_cap_started_at = now
            self.engaged_by = reason
        self.last_engaged_at = now
        return opened

    def hold(self, now: float | None = None) -> None:
        """Refresh the TTL without changing the trigger. Used while the
        user is talking or Glad is waiting to answer, so those seconds
        do not count as a silent follow-up timeout."""
        if self.last_engaged_at is None:
            return
        self.last_engaged_at = self._clock() if now is None else now

    def dismiss(self, reason: str) -> bool:
        """Expire immediately. Returns True if a span was open (even if TTL
        had already elapsed but not yet polled)."""
        had_span = self.last_engaged_at is not None
        self.last_close_reason = reason if had_span else self.last_close_reason
        self.last_engaged_at = None
        self.hard_cap_started_at = None
        self.engaged_by = None
        return had_span

    def poll_expiry(self, now: float | None = None) -> str | None:
        """If a live span just expired, clear it and return ttl_expired | hard_cap."""
        now = self._clock() if now is None else now
        if self.last_engaged_at is None or self.is_engaged(now):
            return None
        reason = (
            "hard_cap"
            if self.hard_cap_started_at is not None and now - self.hard_cap_started_at >= self.hard_cap_s
            else "ttl_expired"
        )
        self.last_close_reason = reason
        self.last_engaged_at = None
        self.hard_cap_started_at = None
        self.engaged_by = None
        return reason
