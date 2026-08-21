"""Question set, roster, answers, and derived discovery/ambient mode."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from glad.conversation.turn import EngagementState

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_QUESTION_SETS_DIR = Path(__file__).resolve().parents[3] / "question_sets"


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class QuestionSet:
    id: str
    version: int
    questions: tuple[Question, ...]

    def get(self, question_id: str) -> Question | None:
        for question in self.questions:
            if question.id == question_id:
                return question
        return None

    def namespaced_id(self, question_id: str) -> str:
        return f"{self.id}.{question_id}"


def load_question_set(name: str) -> QuestionSet:
    path = _QUESTION_SETS_DIR / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _parse(raw, source=str(path))


def _parse(raw: Any, *, source: str) -> QuestionSet:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: question set must be a YAML mapping, got {type(raw).__name__}")

    set_id = raw.get("id")
    if not set_id or not isinstance(set_id, str):
        raise ValueError(f"{source}: question set is missing a string 'id'")

    version = raw.get("version")
    if not isinstance(version, int):
        raise ValueError(f"{source}: question set {set_id!r} is missing an integer 'version'")

    raw_questions = raw.get("questions")
    if not raw_questions:
        raise ValueError(f"{source}: question set {set_id!r} has no questions")

    questions: list[Question] = []
    seen_counts: dict[str, int] = {}
    invalid_ids: list[str] = []
    for entry in raw_questions:
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: question set {set_id!r} has a non-mapping question entry")
        qid = entry.get("id")
        text = entry.get("text")
        if not qid or not isinstance(qid, str):
            raise ValueError(f"{source}: question set {set_id!r} has a question with no string 'id'")
        if not text or not isinstance(text, str):
            raise ValueError(f"{source}: question set {set_id!r} question {qid!r} has no string 'text'")

        if not _ID_PATTERN.match(qid):
            invalid_ids.append(qid)
        seen_counts[qid] = seen_counts.get(qid, 0) + 1
        questions.append(Question(id=qid, text=text))

    if invalid_ids:
        raise ValueError(
            f"{source}: question set {set_id!r} has invalid question id(s) "
            f"(must match {_ID_PATTERN.pattern!r}): {sorted(set(invalid_ids))}"
        )

    duplicate_ids = sorted(qid for qid, count in seen_counts.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"{source}: question set {set_id!r} has duplicate question id(s): {duplicate_ids}")

    return QuestionSet(id=set_id, version=version, questions=tuple(questions))


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
        return [q for q in self.question_set.questions if q.id not in self.answers]


class Mode(str, Enum):
    DISCOVERY = "discovery"
    AMBIENT = "ambient"


def outstanding_question(state: SessionState) -> Question | None:
    remaining = state.remaining()
    return remaining[0] if remaining else None


def derive_mode(state: SessionState) -> Mode:
    return Mode.DISCOVERY if outstanding_question(state) is not None else Mode.AMBIENT
