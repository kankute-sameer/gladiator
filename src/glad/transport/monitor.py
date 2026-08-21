"""Fan the full event bus out to `/ws/monitor`: every event `glad.obs.events`
emits, verbatim, as JSON text. Read-only -- never sends anything the
orchestrator or audio path waits on.

`events.on(...)` callbacks run inline on the emitting task (synchronously,
from inside the audio path), so a slow or absent monitor tab must never
apply backpressure: the subscriber only does a non-blocking queue push, a
per-client task does the actual socket write, and a full queue just drops
that event for that client.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from glad.logging import get_logger
from glad.obs import events

logger = get_logger(__name__)

router = APIRouter()

_QUEUE_MAXSIZE = 2000
_queues: dict[WebSocket, asyncio.Queue[str]] = {}
_subscribed = False


def _on_event(event: dict[str, Any]) -> None:
    """`events.on` callback: synchronous, called inline from `emit`. Must
    never raise and never block."""
    if not _queues:
        return
    try:
        payload = json.dumps(event, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return
    for queue in list(_queues.values()):
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            # Drop for this client rather than block the emitter -- a
            # stalled monitor tab must never slow down the audio path.
            pass


def _ensure_subscribed() -> None:
    global _subscribed
    if not _subscribed:
        events.on(_on_event)
        _subscribed = True


@router.websocket("/ws/monitor")
async def monitor_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    _ensure_subscribed()

    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    _queues[websocket] = queue
    logger.info("Monitor client connected (%d total)", len(_queues))

    async def _sender() -> None:
        while True:
            message = await queue.get()
            await websocket.send_text(message)

    sender_task = asyncio.create_task(_sender())
    try:
        while True:
            # We never expect meaningful input from the dashboard; this
            # just blocks until the browser closes the socket.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 -- any receive error just ends the socket
        pass
    finally:
        sender_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender_task
        _queues.pop(websocket, None)
        logger.info("Monitor client disconnected (%d total)", len(_queues))
