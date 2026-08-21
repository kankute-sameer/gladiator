"""go_dormant reason mapping: Gemini decides, TTL is the fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from glad.conversation.session import Question, QuestionSet, SessionState
from glad.conversation.tools import classify_dormant_reason, dispatch
from glad.orchestrator import Orchestrator

_QUESTION_SET = QuestionSet(
    id="test_set",
    version=1,
    questions=(
        Question(id="budget", text="What's your budget?"),
        Question(id="pain_point", text="What's the pain?"),
    ),
)


def _state() -> SessionState:
    return SessionState(session_id="s1", question_set=_QUESTION_SET)


def test_sidebar_maps_to_not_for_me() -> None:
    assert classify_dormant_reason(_state(), "they're talking to each other") == "not_for_me"
    assert classify_dormant_reason(_state(), "question is for Alice, not me") == "not_for_me"


def test_explicit_dismissal_maps_to_dismissed() -> None:
    assert classify_dormant_reason(_state(), "thanks Glad") == "dismissed"
    assert classify_dormant_reason(_state(), "that's all") == "dismissed"


def test_script_done_maps_to_script_complete() -> None:
    assert classify_dormant_reason(_state(), "script_complete") == "script_complete"
    state = _state()
    state.record("budget", "10k")
    state.record("pain_point", "latency")
    assert classify_dormant_reason(state, "wrapping up") == "script_complete"


def test_dispatch_expires_engagement_and_keeps_raw_detail() -> None:
    state = _state()
    state.engagement.extend("wake_word")
    result = dispatch("go_dormant", {"reason": "next question is for Bob"}, state)
    assert result == {
        "ok": True,
        "reason": "not_for_me",
        "was_engaged": True,
        "detail": "next question is for Bob",
    }
    assert state.engagement.is_engaged() is False


class _FakeLive:
    activity_open = False

    async def close_window(self) -> None:
        self.activity_open = False


def _orchestrator() -> Orchestrator:
    orch = Orchestrator(_QUESTION_SET, session_id="s1")
    orch.bind_live(_FakeLive())  # type: ignore[arg-type]
    orch.engagement.extend("wake_word")
    return orch


@pytest.mark.asyncio
async def test_go_dormant_while_speaking_does_not_cut_the_line(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _orchestrator()
    orch.is_speaking = True
    flush = AsyncMock()
    monkeypatch.setattr("glad.orchestrator.outbound.broadcast_flush", flush)
    monkeypatch.setattr("glad.orchestrator.outbound.has_queued_audio", lambda: True)

    await orch.handle_tool_call("go_dormant", {"reason": "dismissed"})

    assert orch._finish_current_line is True
    assert orch.is_speaking is True
    flush.assert_not_awaited()

    monkeypatch.setattr("glad.orchestrator.outbound.wait_drained", AsyncMock())
    await orch._seal_dormant_after_playback()

    assert orch._finish_current_line is False
    assert orch.is_speaking is False
    flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_go_dormant_before_speech_cuts_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _orchestrator()
    orch.is_speaking = False
    flush = AsyncMock()
    monkeypatch.setattr("glad.orchestrator.outbound.broadcast_flush", flush)
    monkeypatch.setattr("glad.orchestrator.outbound.has_queued_audio", lambda: False)

    await orch.handle_tool_call("go_dormant", {"reason": "dismissed"})

    assert orch._finish_current_line is False
    flush.assert_awaited_once()
