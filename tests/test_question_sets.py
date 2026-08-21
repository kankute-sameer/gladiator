"""Every shipped file in question_sets/ must load without error."""

from __future__ import annotations

from pathlib import Path

import pytest

from glad.conversation.session import load_question_set

_QUESTION_SETS_DIR = Path(__file__).resolve().parents[1] / "question_sets"
_SET_NAMES = sorted(p.stem for p in _QUESTION_SETS_DIR.glob("*.yaml"))


def test_at_least_one_question_set_shipped() -> None:
    assert _SET_NAMES, "expected at least one file under question_sets/"


@pytest.mark.parametrize("name", _SET_NAMES)
def test_question_set_loads(name: str) -> None:
    question_set = load_question_set(name)
    assert question_set.id
    assert question_set.questions
