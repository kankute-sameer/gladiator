"""Who is speaking, from per-participant RMS the orchestrator already has.

Debounce stops overlapping speech from flapping the declared speaker.
The orchestrator rate-limits how often that change is sent to Live.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

DEBOUNCE_S = 0.3


class SpeakerTracker:
    """Highest RMS above `threshold` must hold for `debounce_s` to win."""

    def __init__(
        self,
        *,
        threshold: float,
        debounce_s: float = DEBOUNCE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold = threshold
        self.debounce_s = debounce_s
        self._clock = clock
        self._leader_id: int | None = None
        self._leader_since: float | None = None
        self._speaker_id: int | None = None

    def current_speaker(self) -> int | None:
        return self._speaker_id

    def update(self, levels: Mapping[int, float], now: float | None = None) -> int | None:
        """Feed latest RMS by participant id. Returns the new speaker id
        on change, otherwise None. Quiet frames do not clear the speaker."""
        now = self._clock() if now is None else now
        leader: int | None = None
        best = self.threshold
        for participant_id, level in levels.items():
            if level >= best:
                best = level
                leader = participant_id
        if leader is None:
            self._leader_id = None
            self._leader_since = None
            return None
        if leader != self._leader_id:
            self._leader_id = leader
            self._leader_since = now
            return None
        if self._leader_since is None or now - self._leader_since < self.debounce_s:
            return None
        if leader == self._speaker_id:
            return None
        self._speaker_id = leader
        return leader
