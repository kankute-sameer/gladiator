"""Robustness tests for tool dispatch: on gemini-3.1-flash-live, function
calling is synchronous only (no NON_BLOCKING / scheduling escape hatch),
and the model withholds `turn_complete` until a FunctionResponse arrives
for every function_call in a batch. An escaped exception or a skipped
response is therefore a permanent deadlock, not a recoverable error --
these paths must never be allowed to silently go untested again.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from google.genai import types

from glad.conversation.session import Question, QuestionSet, SessionState
from glad.conversation.tools import dispatch
from glad.live.session import LiveSession

_QUESTION_SET = QuestionSet(
    id="test_set",
    version=1,
    questions=(Question(id="budget", text="What's your budget?"),),
)


def _state() -> SessionState:
    return SessionState(session_id="s1", question_set=_QUESTION_SET)


def test_dispatch_unknown_question_id_returns_error_not_raise() -> None:
    """This branch has probably never executed for real -- assert it here
    rather than trusting it works because it hasn't been observed to fail."""
    result = dispatch("record_answer", {"question_id": "not_a_real_question", "value": "x"}, _state())
    assert result == {"ok": False, "error": "Unknown question id: 'not_a_real_question'"}


def test_dispatch_unknown_tool_name_returns_error_not_raise() -> None:
    result = dispatch("not_a_real_tool", {"foo": "bar"}, _state())
    assert result["ok"] is False
    assert "not_a_real_tool" in result["error"]


def _make_session() -> LiveSession:
    return LiveSession(
        api_key="test-key",
        model="test-model",
        instruction_provider=lambda: "system instruction",
    )


@pytest.mark.asyncio
async def test_raising_tool_handler_still_sends_a_function_response() -> None:
    """A tool handler that raises must still result in a FunctionResponse
    being sent -- an escaped exception here is a permanent deadlock, not
    just a dropped call, because the model withholds turn_complete until
    a response arrives."""

    async def raising_dispatcher(name: str, args: dict) -> dict:
        raise RuntimeError("boom")

    session = LiveSession(
        api_key="test-key",
        model="test-model",
        instruction_provider=lambda: "system instruction",
        tool_dispatcher=raising_dispatcher,
    )
    mock_ws_session = AsyncMock()
    tool_call = types.LiveServerToolCall(
        function_calls=[types.FunctionCall(id="call-1", name="record_answer", args={"question_id": "budget", "value": "x"})]
    )

    await session._handle_tool_call(mock_ws_session, tool_call)

    mock_ws_session.send_tool_response.assert_awaited_once()
    _, kwargs = mock_ws_session.send_tool_response.call_args
    responses = kwargs["function_responses"]
    assert len(responses) == 1
    assert responses[0].id == "call-1"
    assert responses[0].response.get("ok") is False


@pytest.mark.asyncio
async def test_two_function_calls_each_get_a_function_response() -> None:
    """Same deadlock class as the raising-handler case: every element of a
    multi-call `tool_call` batch must get a matching FunctionResponse, or
    the model stalls waiting on whichever id never came back."""
    seen_calls: list[tuple[str, dict]] = []

    async def recording_dispatcher(name: str, args: dict) -> dict:
        seen_calls.append((name, args))
        return {"ok": True}

    session = LiveSession(
        api_key="test-key",
        model="test-model",
        instruction_provider=lambda: "system instruction",
        tool_dispatcher=recording_dispatcher,
    )
    mock_ws_session = AsyncMock()
    tool_call = types.LiveServerToolCall(
        function_calls=[
            types.FunctionCall(id="call-1", name="record_answer", args={"question_id": "budget", "value": "a"}),
            types.FunctionCall(id="call-2", name="record_answer", args={"question_id": "budget", "value": "b"}),
        ]
    )

    await session._handle_tool_call(mock_ws_session, tool_call)

    assert len(seen_calls) == 2
    mock_ws_session.send_tool_response.assert_awaited_once()
    _, kwargs = mock_ws_session.send_tool_response.call_args
    responses = kwargs["function_responses"]
    assert {r.id for r in responses} == {"call-1", "call-2"}
    assert all(r.response.get("ok") for r in responses)


@pytest.mark.asyncio
async def test_phantom_turn_does_not_run_tools_but_still_responds() -> None:
    seen_calls: list[str] = []

    async def recording_dispatcher(name: str, args: dict) -> dict:
        seen_calls.append(name)
        return {"ok": True}

    session = LiveSession(
        api_key="test-key",
        model="test-model",
        instruction_provider=lambda: "system instruction",
        tool_dispatcher=recording_dispatcher,
    )
    session._discard_turn = True
    mock_ws_session = AsyncMock()
    tool_call = types.LiveServerToolCall(
        function_calls=[types.FunctionCall(id="call-1", name="go_dormant", args={"reason": "not_for_me"})]
    )

    await session._handle_tool_call(mock_ws_session, tool_call)

    assert seen_calls == []
    mock_ws_session.send_tool_response.assert_awaited_once()
    _, kwargs = mock_ws_session.send_tool_response.call_args
    response = kwargs["function_responses"][0]
    assert response.id == "call-1"
    assert response.response.get("ok") is False
    assert "not room speech" in response.response.get("error", "")
