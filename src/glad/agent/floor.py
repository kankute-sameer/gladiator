"""Floor control: "may I speak now", separate from wake word detection's
"was I addressed" (`glad.agent.wakeword`).

Four states:
  AMBIENT       -- idle, no window open, listening only.
  WAKE_PENDING  -- a wake word matched; waiting for the room to go quiet
                   for `gap_threshold_s` before taking the floor.
  SPEAKING      -- a window is open / a turn is in flight.
  YIELDED       -- transient: a barge-in interrupted SPEAKING. The actual
                   flush happens in `orchestrator.run()`'s `is_interrupted`
                   handling; this class only reflects the state back.

DISCOVERY vs AMBIENT mode (`glad.agent.mode`) is a flag passed into
`on_speech_started`, not a fifth state: in discovery mode, speech takes
the floor directly; in ambient mode, only a wake word can move
AMBIENT -> WAKE_PENDING, and only the post-wake silence gap moves
WAKE_PENDING -> SPEAKING.

An empty window (opened and closed with no real content) produces an
audible, hallucinated reply, so the ambient path never opens a raw-audio
window at all -- `WAKE` means "hand the model a text turn", not
"open a window for live audio".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FloorState(str, Enum):
    AMBIENT = "ambient"
    WAKE_PENDING = "wake_pending"
    SPEAKING = "speaking"
    YIELDED = "yielded"


class FloorAction(str, Enum):
    NONE = "none"
    OPEN_WINDOW = "open_window"  # discovery mode: start capturing live participant audio
    CLOSE_WINDOW = "close_window"  # discovery mode: endpointing silence gap elapsed
    WAKE = "wake"  # ambient mode: gap elapsed post-wake-word -- inject buffered text, no window


@dataclass(slots=True)
class FloorControl:
    """Pure state machine -- no I/O, no clock of its own. Callers pass in
    `now` (monotonic seconds) and act on the returned `FloorAction`.

    `_silence_since` defaults to "-inf" (always already silent) rather than
    `None`: a wake word can arrive before we've ever seen a speech/silence
    edge (e.g. a quiet "Glad?" the energy detector never crossed threshold
    on), and treating that as "nothing to wait for" is correct.
    """

    gap_threshold_s: float = 1.2
    endpoint_gap_s: float = 0.5
    state: FloorState = FloorState.AMBIENT
    _speech_active: bool = False
    _silence_since: float = float("-inf")

    def on_speech_started(self, now: float, *, discovery_mode: bool) -> FloorAction:
        """Call whenever speech-energy transitions from silent to active."""
        self._speech_active = True
        if discovery_mode and self.state is FloorState.AMBIENT:
            self.state = FloorState.SPEAKING
            return FloorAction.OPEN_WINDOW
        return FloorAction.NONE

    def on_barge_in(self, now: float) -> FloorAction:
        """User talked over Glad. Always signal activity so Gemini stops
        generating."""
        self._speech_active = True
        self.state = FloorState.SPEAKING
        return FloorAction.OPEN_WINDOW

    def on_speech_ended(self, now: float) -> None:
        """Call whenever speech-energy transitions from active to silent."""
        self._speech_active = False
        self._silence_since = now

    def on_wake_matched(self, now: float) -> bool:
        """A stage-2-accepted wake word arrived. Returns whether this
        reached WAKE_PENDING -- False if the floor was already pending
        or taken."""
        if self.state is FloorState.AMBIENT:
            self.state = FloorState.WAKE_PENDING
            return True
        return False

    def tick(self, now: float) -> FloorAction:
        """Call periodically to check elapsed silence against the current
        state's threshold. No-op while speech is active."""
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
        """Barge-in during SPEAKING; the flush itself happens in
        `orchestrator.run()`. Keep SPEAKING if a listen window is open so
        `tick()` can still send ActivityEnd. Text-only turns go AMBIENT."""
        if listening:
            self.state = FloorState.SPEAKING
            return
        self.state = FloorState.YIELDED
        self.state = FloorState.AMBIENT

    def on_turn_complete(self, *, listening: bool = False) -> None:
        """The model finished a turn. Guards against staying SPEAKING when
        the turn was text-only (no window for `tick()` to close). An open
        listen window must stay SPEAKING or ActivityEnd never fires."""
        if listening:
            return
        if self.state is FloorState.SPEAKING:
            self.state = FloorState.AMBIENT
