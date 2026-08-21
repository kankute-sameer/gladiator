"""The `record_answer` tool: the model's only write path into `SessionState`."""

from __future__ import annotations

from typing import Any

from glad.agent.state import SessionState


def record_answer(
    state: SessionState,
    question_id: str,
    value: str,
    participant_id: int | None = None,
    participant_name: str | None = None,
) -> dict[str, Any]:
    """Record one answer. Returns `{"ok": False, "error": ...}` for an
    unknown `question_id` instead of raising, so Gemini sees the rejection
    as a tool result and can correct itself."""
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
