"""Buffers ambient-mode transcript text so it can be handed to the model as
context on the next turn Glad was going to take anyway (a discovery window
opening, or a wake word firing).

Audio streamed with no activity window open is not retained as model
context, but the Recall transcript text keeps arriving regardless -- this
buffer is what makes volunteered ambient answers reachable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AmbientUtterance:
    speaker: str
    text: str
    ts: float


@dataclass(slots=True)
class AmbientBuffer:
    """One per session. Only meant to be fed while mode is AMBIENT --
    discovery mode already gets participant speech via a live audio
    window, so buffering it too would just duplicate it."""

    _utterances: list[AmbientUtterance] = field(default_factory=list)

    def add(self, speaker: str, text: str, ts: float) -> None:
        text = text.strip()
        if text:
            self._utterances.append(AmbientUtterance(speaker=speaker, text=text, ts=ts))

    def __len__(self) -> int:
        return len(self._utterances)

    def flush(self) -> str | None:
        """Return the buffered utterances as one context string and clear
        the buffer. None if empty -- callers must not inject an empty
        turn, since every turn, empty or not, produces an audible reply."""
        if not self._utterances:
            return None
        lines = [
            "[Ambient context: said while you were not actively asking a "
            "question, so you have not heard it yet. Treat any answers to "
            "your discovery questions here as if they were just said aloud "
            "-- call record_answer for them.]"
        ]
        for utterance in self._utterances:
            lines.append(f"{utterance.speaker}: {utterance.text}")
        self._utterances.clear()
        return "\n".join(lines)
