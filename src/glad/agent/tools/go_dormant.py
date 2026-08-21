"""Explicit ENGAGED -> DORMANT. Used for dismissal and script completion."""

from __future__ import annotations

from typing import Any

from glad.agent.state import SessionState


def go_dormant(state: SessionState, reason: str, **_ignored: Any) -> dict[str, Any]:
    lowered = reason.lower()
    close = (
        "script_complete"
        if not state.remaining() or "script" in lowered or "complete" in lowered
        else "dismissed"
    )
    was = state.engagement.dismiss(close)
    return {"ok": True, "reason": close, "was_engaged": was}
