"""Conversational mode, derived fresh on every check -- never stored.

DISCOVERY: a question is still outstanding. Normal turn-taking, no wake
word needed -- this is the main path (see `glad.agent.floor`).

AMBIENT: every question has an answer recorded. Glad stays silent unless
woken.
"""

from __future__ import annotations

from enum import Enum

from glad.agent.script import Question
from glad.agent.state import SessionState


class Mode(str, Enum):
    DISCOVERY = "discovery"
    AMBIENT = "ambient"


def outstanding_question(state: SessionState) -> Question | None:
    """The next unanswered question, or None once every question in the
    set has a recorded answer."""
    remaining = state.remaining()
    return remaining[0] if remaining else None


def derive_mode(state: SessionState) -> Mode:
    return Mode.DISCOVERY if outstanding_question(state) is not None else Mode.AMBIENT
