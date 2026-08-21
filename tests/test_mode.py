"""Tests for glad.agent.mode: mode is derived, never stored."""

from __future__ import annotations

from glad.conversation.session import (
    Mode,
    Question,
    QuestionSet,
    SessionState,
    derive_mode,
    outstanding_question,
)

_QUESTION_SET = QuestionSet(
    id="test_set",
    version=1,
    questions=(
        Question(id="budget", text="What's your budget?"),
        Question(id="timeline", text="What's your timeline?"),
    ),
)


def _state() -> SessionState:
    return SessionState(session_id="s1", question_set=_QUESTION_SET)


def test_discovery_while_any_question_unanswered() -> None:
    state = _state()
    assert outstanding_question(state) is not None
    assert derive_mode(state) is Mode.DISCOVERY

    state.record("budget", "$10k")
    assert outstanding_question(state).id == "timeline"
    assert derive_mode(state) is Mode.DISCOVERY


def test_ambient_once_every_question_answered() -> None:
    state = _state()
    state.record("budget", "$10k")
    state.record("timeline", "next quarter")

    assert outstanding_question(state) is None
    assert derive_mode(state) is Mode.AMBIENT


def test_mode_is_derived_not_cached() -> None:
    """Answering a question flips the mode on the very next check -- there
    is no stored flag to fall out of sync."""
    state = _state()
    state.record("budget", "$10k")
    state.record("timeline", "next quarter")
    assert derive_mode(state) is Mode.AMBIENT

    # A correction (last-write-wins) doesn't change remaining()'s emptiness.
    state.record("budget", "actually $20k")
    assert derive_mode(state) is Mode.AMBIENT
