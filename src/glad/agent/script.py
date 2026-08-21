"""Loads and validates a discovery question set from YAML.

The PROMPT (see `glad.agent.prompt`) drives which questions get asked, when,
and how they're phrased -- this module only defines what the valid
questions *are*, and guarantees their ids are safe to use as a Gemini
tool-call enum and as stable dashboard/log keys. No ordering or
state-machine logic belongs here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

# repo_root/question_sets -- four parents up from src/glad/agent/script.py.
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
        """f"{set_id}.{question_id}" -- stable across question text/order
        edits, for use in logs and dashboards, not for the tool enum
        (Gemini is given the plain id)."""
        return f"{self.id}.{question_id}"


def load_question_set(name: str) -> QuestionSet:
    """Load and validate `question_sets/<name>.yaml`. Raises `ValueError`
    for a duplicate id, malformed id, or empty question list -- anything
    a bot must never start with."""
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
