"""Refresh engagement TTL. No args; returns immediately so it cannot stall speech."""

from __future__ import annotations

from typing import Any

from glad.agent.state import SessionState


def stay_engaged(state: SessionState, **_ignored: Any) -> dict[str, Any]:
    """Extend only while already ENGAGED -- a late call after TTL expiry
    must not reopen, or decay would not be a recovery path."""
    if state.engagement.is_engaged():
        state.engagement.extend("stay_engaged")
        return {"ok": True, "extended": True}
    return {"ok": True, "extended": False}
