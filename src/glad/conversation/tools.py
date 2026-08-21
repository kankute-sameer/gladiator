"""Gemini tools: record_answer, go_dormant."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.genai import types

from glad.conversation.session import QuestionSet, SessionState

_DISMISS_MARKERS = (
    "dismiss",
    "thanks",
    "that's all",
    "thats all",
    "that is all",
)
_SCRIPT_MARKERS = (
    "script_complete",
    "script done",
    "all questions",
    "questions answered",
)


def record_answer(
    state: SessionState,
    question_id: str,
    value: str,
    participant_id: int | None = None,
    participant_name: str | None = None,
) -> dict[str, Any]:
    try:
        state.record(
            question_id,
            value,
            participant_id=participant_id,
            participant_name=participant_name,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "remaining": [q.id for q in state.remaining()]}


def classify_dormant_reason(state: SessionState, reason: str) -> str:
    lowered = reason.lower()
    if not state.remaining() or any(marker in lowered for marker in _SCRIPT_MARKERS):
        return "script_complete"
    if any(marker in lowered for marker in _DISMISS_MARKERS):
        return "dismissed"
    return "not_for_me"


def go_dormant(state: SessionState, reason: str, **_ignored: Any) -> dict[str, Any]:
    close = classify_dormant_reason(state, reason)
    was = state.engagement.dismiss(close)
    return {"ok": True, "reason": close, "was_engaged": was, "detail": reason}


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


def _go_dormant_declaration(_question_set: QuestionSet) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="go_dormant",
        description=(
            "Leave the conversation. Until you call this, you stay in it. "
            "If you have not started speaking, remaining audio is discarded. "
            "If you have already started a line, that line finishes, then "
            "you go quiet. Call this FIRST (with no spoken line) when "
            "speech is not for you. Never speak after this tool. "
            "Pass a short reason."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "reason": types.Schema(
                    type=types.Type.STRING,
                    description="Why you are going dormant (not_for_me, dismissed, or script_complete).",
                ),
            },
            required=["reason"],
        ),
    )


_TOOLS: list[
    tuple[Callable[..., dict[str, Any]], Callable[[QuestionSet], types.FunctionDeclaration]]
] = [
    (record_answer, _record_answer_declaration),
    (go_dormant, _go_dormant_declaration),
]


def declarations(question_set: QuestionSet) -> list[types.FunctionDeclaration]:
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
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"Unknown tool: {name!r}"}
    return handler(state, **args)
