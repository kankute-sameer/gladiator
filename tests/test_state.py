"""Tests for glad.agent.state: recording answers and session isolation."""

from __future__ import annotations

import pytest

from glad.conversation.session import Question, QuestionSet, SessionState

_QUESTION_SET = QuestionSet(
    id="test_set",
    version=1,
    questions=(
        Question(id="budget", text="What's your budget?"),
        Question(id="timeline", text="What's your timeline?"),
        Question(id="team_size", text="How big is the team?"),
    ),
)


def _state(session_id: str = "session-1") -> SessionState:
    return SessionState(session_id=session_id, question_set=_QUESTION_SET)


def test_record_then_rerecord_overwrites_and_bumps_revision() -> None:
    state = _state()

    first = state.record("budget", "around $10k")
    assert first.revision == 1
    assert len(state.answers) == 1

    second = state.record("budget", "actually more like $50k")
    assert second.revision == 2
    assert state.answers["budget"].value == "actually more like $50k"
    # Still one entry, not a second one for the same question.
    assert len(state.answers) == 1


def test_unknown_question_id_raises() -> None:
    state = _state()

    with pytest.raises(ValueError, match="Unknown question id"):
        state.record("not_a_real_question", "whatever")


def test_remaining_shrinks_as_answers_come_in() -> None:
    state = _state()
    assert [q.id for q in state.remaining()] == ["budget", "timeline", "team_size"]

    state.record("timeline", "next quarter")
    assert [q.id for q in state.remaining()] == ["budget", "team_size"]

    state.record("budget", "$10k")
    state.record("team_size", "5 people")
    assert state.remaining() == []


def test_two_session_states_are_independent() -> None:
    state_a = _state("session-a")
    state_b = _state("session-b")

    state_a.record("budget", "$10k")

    assert "budget" in state_a.answers
    assert state_b.answers == {}
    assert [q.id for q in state_b.remaining()] == ["budget", "timeline", "team_size"]


def test_record_stores_participant_and_last_write_wins() -> None:
    state = _state()
    first = state.record("budget", "$10k", participant_id=1, participant_name="Alice")
    assert first.participant_id == 1
    assert first.participant_name == "Alice"

    second = state.record("budget", "$50k", participant_id=2, participant_name="Bob")
    assert second.revision == 2
    assert second.participant_id == 2
    assert state.answers["budget"].value == "$50k"
