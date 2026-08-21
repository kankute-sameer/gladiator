"""Gemini function-calling glue: `declarations()` for connect-time config,
`dispatch()` to route an incoming tool call to its handler.

The `assert fn.__name__ == declaration.name` check in `declarations()` is
deliberate: it makes renaming a handler without updating its declaration
fail loudly at call time instead of silently drifting Gemini's idea of
what's callable out of sync with what `dispatch` actually routes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.genai import types

from glad.agent.script import QuestionSet
from glad.agent.state import SessionState
from glad.agent.tools.go_dormant import go_dormant
from glad.agent.tools.record_answer import record_answer
from glad.agent.tools.stay_engaged import stay_engaged


def _record_answer_declaration(question_set: QuestionSet) -> types.FunctionDeclaration:
    question_ids = [q.id for q in question_set.questions]
    return types.FunctionDeclaration(
        name="record_answer",
        description=(
            "Record a participant's answer to one discovery question. Call "
            "this whenever any question in the set is answered -- whether "
            "or not it was just asked, and whether or not you say anything "
            "back about it. Calling it again for the same question_id "
            "overwrites the previous value with the refined one."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "question_id": types.Schema(
                    type=types.Type.STRING,
                    format="enum",
                    enum=question_ids,
                    description="Which question this answers.",
                ),
                "value": types.Schema(
                    type=types.Type.STRING,
                    description="The answer, in the participant's own words.",
                ),
            },
            required=["question_id", "value"],
        ),
    )


def _stay_engaged_declaration(_question_set: QuestionSet) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="stay_engaged",
        description=(
            "Refresh engagement so you may keep speaking. Call this whenever "
            "the exchange with you is continuing -- a follow-up, a probe, a "
            "clarification. No arguments. Do not wait on the result."
        ),
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    )


def _go_dormant_declaration(_question_set: QuestionSet) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="go_dormant",
        description=(
            "Go silent. Any spoken audio in this turn that has not already "
            "played is dropped the moment this tool runs — there is no way "
            "to finish a sentence after it. If people should hear a short "
            "sign-off, speak that line to completion first, then call this. "
            "If they should hear nothing more, call this immediately. Call "
            "only after an explicit dismissal ('thanks Glad', 'that's all') "
            "or when every discovery question has an answer. Pass a short "
            "reason. Do NOT call this on topic drift — just stay silent "
            "and let engagement expire."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "reason": types.Schema(
                    type=types.Type.STRING,
                    description="Why you are going dormant (dismissed or script_complete).",
                ),
            },
            required=["reason"],
        ),
    )


# (handler, declaration builder) pairs. Add future tools here.
_TOOLS: list[
    tuple[Callable[..., dict[str, Any]], Callable[[QuestionSet], types.FunctionDeclaration]]
] = [
    (record_answer, _record_answer_declaration),
    (stay_engaged, _stay_engaged_declaration),
    (go_dormant, _go_dormant_declaration),
]


def declarations(question_set: QuestionSet) -> list[types.FunctionDeclaration]:
    """Gemini function declarations for `question_set`. `question_id` is a
    closed enum of that set's ids -- Gemini cannot even construct a call
    naming a question that doesn't exist in this bot's script."""
    built: list[types.FunctionDeclaration] = []
    for fn, build in _TOOLS:
        declaration = build(question_set)
        assert fn.__name__ == declaration.name, (
            f"tool handler {fn.__name__!r} does not match its own declaration "
            f"name {declaration.name!r}"
        )
        built.append(declaration)
    return built


_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {fn.__name__: fn for fn, _ in _TOOLS}


def dispatch(name: str, args: dict[str, Any], state: SessionState) -> dict[str, Any]:
    """Route one tool call by name to its handler. An unknown tool name
    returns the same `{"ok": False, ...}` shape a handler's own
    validation would."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"Unknown tool: {name!r}"}
    return handler(state, **args)
