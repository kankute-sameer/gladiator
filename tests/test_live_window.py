"""ActivityStart / ActivityEnd must stay paired. An unmatched ActivityEnd
makes the Live API close the socket with 1007 (Precondition check failed),
which used to take down the inbound Recall websocket with it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from glad.live.session import LiveSession


def _session() -> LiveSession:
    live = LiveSession(
        api_key="test-key",
        model="test-model",
        instruction_provider=lambda: "system instruction",
    )
    live._session = MagicMock()
    live._session.send_realtime_input = AsyncMock()
    return live


@pytest.mark.asyncio
async def test_close_without_open_does_not_send_activity_end() -> None:
    live = _session()
    await live.close_window()
    live._session.send_realtime_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_then_close_sends_start_then_end() -> None:
    live = _session()
    await live.open_window()
    await live.close_window()
    calls = live._session.send_realtime_input.await_args_list
    assert len(calls) == 2
    assert "activity_start" in calls[0].kwargs
    assert "activity_end" in calls[1].kwargs


@pytest.mark.asyncio
async def test_second_close_is_a_noop() -> None:
    live = _session()
    await live.open_window()
    await live.close_window()
    live._session.send_realtime_input.reset_mock()
    await live.close_window()
    live._session.send_realtime_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_second_open_does_not_send_another_start() -> None:
    live = _session()
    await live.open_window()
    live._session.send_realtime_input.reset_mock()
    await live.open_window()
    live._session.send_realtime_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_swallows_1007_and_clears_open_flag() -> None:
    live = _session()
    await live.open_window()
    live._session.send_realtime_input = AsyncMock(
        side_effect=ConnectionClosedError(
            rcvd=Close(code=1007, reason="Precondition check failed."),
            sent=None,
        )
    )
    await live.close_window()
    live._session.send_realtime_input = AsyncMock()
    await live.close_window()
    live._session.send_realtime_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_context_before_window_flushes_on_open() -> None:
    live = _session()
    await live.send_context("[People currently in this call: Alice]")
    live._session.send_realtime_input.assert_not_awaited()
    await live.open_window()
    calls = live._session.send_realtime_input.await_args_list
    assert "activity_start" in calls[0].kwargs
    assert calls[1].kwargs.get("text") == "[People currently in this call: Alice]"


@pytest.mark.asyncio
async def test_send_context_during_open_window_goes_out_immediately() -> None:
    live = _session()
    await live.open_window()
    live._session.send_realtime_input.reset_mock()
    await live.send_context("[People currently in this call: Alice]")
    live._session.send_realtime_input.assert_awaited_once()
    assert live._session.send_realtime_input.await_args.kwargs.get("text") == (
        "[People currently in this call: Alice]"
    )
