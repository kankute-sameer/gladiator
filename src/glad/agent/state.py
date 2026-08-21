"""Session-scoped conversation state: what's been answered so far.

No global state -- the orchestrator holds one `SessionState` per bot
(`dict[session_id, SessionState]`), so two concurrent bots never share, or
even see, each other's answers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from glad.agent.engagement import EngagementState
from glad.agent.script import Question, QuestionSet


@dataclass(slots=True)
class Participant:
    id: int
    name: str
    is_host: bool | None
    joined_at: float
    left_at: float | None = None


@dataclass(slots=True)
class Roster:
    """Who has been in this meeting. Departed people stay, with `left_at` set."""

    _by_id: dict[int, Participant] = field(default_factory=dict)
    _clock: Callable[[], float] = field(default=time.time, repr=False)

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, participant_id: int) -> Participant | None:
        return self._by_id.get(participant_id)

    def present(self) -> list[Participant]:
        return [p for p in self._by_id.values() if p.left_at is None]

    def all(self) -> list[Participant]:
        return list(self._by_id.values())

    def join(self, participant_id: int, name: str, is_host: bool | None, now: float | None = None) -> Participant:
        now = self._clock() if now is None else now
        existing = self._by_id.get(participant_id)
        if existing is None:
            participant = Participant(
                id=participant_id, name=name, is_host=is_host, joined_at=now
            )
            self._by_id[participant_id] = participant
            return participant
        existing.name = name or existing.name
        existing.is_host = is_host if is_host is not None else existing.is_host
        existing.left_at = None
        return existing

    def leave(self, participant_id: int, name: str, is_host: bool | None, now: float | None = None) -> Participant:
        now = self._clock() if now is None else now
        existing = self._by_id.get(participant_id)
        if existing is None:
            participant = Participant(
                id=participant_id, name=name, is_host=is_host, joined_at=now, left_at=now
            )
            self._by_id[participant_id] = participant
            return participant
        if name:
            existing.name = name
        if is_host is not None:
            existing.is_host = is_host
        existing.left_at = now
        return existing

    def update(self, participant_id: int, name: str, is_host: bool | None, now: float | None = None) -> Participant:
        existing = self._by_id.get(participant_id)
        if existing is None:
            return self.join(participant_id, name, is_host, now)
        if name:
            existing.name = name
        if is_host is not None:
            existing.is_host = is_host
        return existing

    def note(self, participant_id: int, name: str, is_host: bool | None = None) -> bool:
        """Mark this person present. True if they were not already in the call."""
        existing = self.get(participant_id)
        newly = existing is None or existing.left_at is not None
        self.join(participant_id, name, is_host)
        return newly


@dataclass(frozen=True, slots=True)
class Answer:
    question_id: str
    value: str
    revision: int
    recorded_at: float
    participant_id: int | None = None
    participant_name: str | None = None


@dataclass(slots=True)
class SessionState:
    session_id: str
    question_set: QuestionSet
    answers: dict[str, Answer] = field(default_factory=dict)
    engagement: EngagementState = field(default_factory=EngagementState)
    roster: Roster = field(default_factory=Roster)

    @property
    def question_set_id(self) -> str:
        return self.question_set.id

    def record(
        self,
        question_id: str,
        value: str,
        *,
        participant_id: int | None = None,
        participant_name: str | None = None,
    ) -> Answer:
        """Record one answer. Last write wins: a second call for the same
        `question_id` overwrites in place and bumps `revision` rather than
        creating a second entry. Raises `ValueError` for an unknown
        `question_id`."""
        if self.question_set.get(question_id) is None:
            raise ValueError(f"Unknown question id: {question_id!r}")

        previous = self.answers.get(question_id)
        revision = previous.revision + 1 if previous is not None else 1
        answer = Answer(
            question_id=question_id,
            value=value,
            revision=revision,
            recorded_at=time.time(),
            participant_id=participant_id,
            participant_name=participant_name,
        )
        self.answers[question_id] = answer
        return answer

    def remaining(self) -> list[Question]:
        """Questions with no recorded answer yet, in question-set order."""
        return [q for q in self.question_set.questions if q.id not in self.answers]
