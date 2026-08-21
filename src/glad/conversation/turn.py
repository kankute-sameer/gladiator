"""Floor, engagement TTL, dormant transcript buffer, and speaker tracking.

DORMANT speech never opens a Live activity window. Wake word moves
AMBIENT -> WAKE_PENDING; the silence gap then fires WAKE (a text turn).
Once ENGAGED, ordinary speech opens and closes audio windows, and barge-in
still interrupts Glad.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum


class FloorState(str, Enum):
    AMBIENT = "ambient"
    WAKE_PENDING = "wake_pending"
    SPEAKING = "speaking"
    YIELDED = "yielded"


class FloorAction(str, Enum):
    NONE = "none"
    OPEN_WINDOW = "open_window"
    CLOSE_WINDOW = "close_window"
    WAKE = "wake"


@dataclass(slots=True)
class FloorControl:
    """Pure state machine -- no I/O, no clock of its own. Callers pass in
    `now` (monotonic seconds) and act on the returned `FloorAction`.

    `_silence_since` defaults to "-inf" (already silent) so a quiet "Glad?"
    that never crossed the energy detector can still fire after the gap.
    """

    gap_threshold_s: float = 1.2
    endpoint_gap_s: float = 1.2
    state: FloorState = FloorState.AMBIENT
    _speech_active: bool = False
    _silence_since: float = float("-inf")

    def on_speech_started(self, now: float, *, discovery_mode: bool) -> FloorAction:
        """`discovery_mode` here means Glad is currently ENGAGED and may
        take the floor on ordinary speech (no wake word)."""
        self._speech_active = True
        if discovery_mode and self.state is FloorState.AMBIENT:
            self.state = FloorState.SPEAKING
            return FloorAction.OPEN_WINDOW
        return FloorAction.NONE

    def on_barge_in(self, now: float) -> FloorAction:
        """User talked over Glad. Always signal activity so Gemini stops."""
        self._speech_active = True
        self.state = FloorState.SPEAKING
        return FloorAction.OPEN_WINDOW

    def on_speech_ended(self, now: float) -> None:
        self._speech_active = False
        self._silence_since = now

    def on_wake_matched(self, now: float) -> bool:
        """Stage-2-accepted wake word. True if this reached WAKE_PENDING."""
        if self.state is FloorState.AMBIENT:
            self.state = FloorState.WAKE_PENDING
            return True
        return False

    def tick(self, now: float) -> FloorAction:
        if self._speech_active:
            return FloorAction.NONE
        elapsed = now - self._silence_since

        if self.state is FloorState.WAKE_PENDING and elapsed >= self.gap_threshold_s:
            self.state = FloorState.SPEAKING
            return FloorAction.WAKE

        if self.state is FloorState.SPEAKING and elapsed >= self.endpoint_gap_s:
            self.state = FloorState.AMBIENT
            return FloorAction.CLOSE_WINDOW

        return FloorAction.NONE

    def on_interrupted(self, *, listening: bool = False) -> None:
        if listening:
            self.state = FloorState.SPEAKING
            return
        self.state = FloorState.YIELDED
        self.state = FloorState.AMBIENT

    def on_turn_complete(self, *, listening: bool = False) -> None:
        if listening:
            return
        if self.state is FloorState.SPEAKING:
            self.state = FloorState.AMBIENT


class EngagementState:
    """ENGAGED after wake until go_dormant. No timer."""

    def __init__(self) -> None:
        self.engaged_by: str | None = None
        self.last_close_reason: str | None = None

    def is_engaged(self, now: float | None = None) -> bool:
        return self.engaged_by is not None

    def extend(self, reason: str, now: float | None = None) -> bool:
        opened = self.engaged_by is None
        if opened:
            self.engaged_by = reason
        return opened

    def dismiss(self, reason: str) -> bool:
        had_span = self.engaged_by is not None
        if had_span:
            self.last_close_reason = reason
        self.engaged_by = None
        return had_span


@dataclass(frozen=True, slots=True)
class AmbientUtterance:
    speaker: str
    text: str
    ts: float


@dataclass(slots=True)
class AmbientBuffer:
    """Transcripts collected while DORMANT, handed to Gemini on wake."""

    _utterances: list[AmbientUtterance] = field(default_factory=list)

    def add(self, speaker: str, text: str, ts: float) -> None:
        text = text.strip()
        if text:
            self._utterances.append(AmbientUtterance(speaker=speaker, text=text, ts=ts))

    def __len__(self) -> int:
        return len(self._utterances)

    def flush(self) -> str | None:
        if not self._utterances:
            return None
        lines = [
            "[Said while you were not in the conversation. Treat any answers "
            "to your discovery questions here as if they were just said "
            "aloud -- call record_answer for them.]"
        ]
        for utterance in self._utterances:
            lines.append(f"{utterance.speaker}: {utterance.text}")
        self._utterances.clear()
        return "\n".join(lines)


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
