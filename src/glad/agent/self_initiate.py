"""Rate-limiting state for asking the next script question in a natural gap,
so an ignored question does not retrigger immediately."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SelfInitiate:
    gap_s: float
    cooldown_s: float
    enabled: bool = True
    _floor_free_since: float | None = None
    _last_fired_at: float = float("-inf")

    def ready(self, now: float, *, floor_busy: bool) -> bool:
        """True once the floor has been free for `gap_s` and `cooldown_s`
        has elapsed since this last fired."""
        if floor_busy:
            self._floor_free_since = None
            return False
        if self._floor_free_since is None:
            self._floor_free_since = now
            return False
        if now - self._floor_free_since < self.gap_s:
            return False
        return now - self._last_fired_at >= self.cooldown_s

    def mark_fired(self, now: float) -> None:
        self._last_fired_at = now
