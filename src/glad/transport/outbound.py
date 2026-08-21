"""Broadcast to the meeting page: levels and streamed PCM audio.

`/ws/meeting` is shared by two kinds of client:
  - The meeting page itself (receives levels/audio, sends nothing of
    substance -- its messages are only used to detect disconnects).
  - `scripts/play_file.py` / the orchestrator (send raw PCM as binary
    frames, plus an occasional `{"t": "flush"}` control message).

Binary frames are relayed byte-for-byte -- no base64, no resampling. But
`send_bytes` (the orchestrator's path for Gemini's replies) is metered to
realtime by `_pace_loop` rather than relayed the instant bytes arrive:
Gemini generates audio faster than it plays, and forwarding that burst
straight through would push it into the browser's ring buffer, where
undoing it costs a round-tripped flush. Holding the excess in `_pending`
here means an interrupt just clears a Python buffer.

`play_file.py`'s own relay (`_handle_inbound_audio`) is untouched -- it
already paces itself to realtime before it ever reaches this socket.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from glad.audio.buffer import JitterBuffer
from glad.audio.pcm import rms
from glad.logging import get_logger
from glad.obs import events, metrics
from glad.transport.schemas import AudioFrame

logger = get_logger(__name__)

router = APIRouter()

_TICK_SECONDS = 0.1  # 10Hz
# Ignore the bot's own tile if Recall ever streams it as a participant.
_SKIP_NAMES = frozenset({"glad"})

# 24kHz mono s16le, same rate as playback.js and JitterBuffer.
_PACE_BYTES_PER_MS = 48.0
_PACE_CHUNK_BYTES = 960  # 20ms, matches play_file.py's default chunk
_PACE_REPORT_EVERY_N_CHUNKS = 10  # ~200ms, matches the client's stats cadence


_latest_levels: dict[int, dict[str, float | int | str]] = {}
_clients: set[WebSocket] = set()
_lock = asyncio.Lock()
_jitter_buffer = JitterBuffer()
_saw_audio = False
_frame_listener: Callable[[AudioFrame], Awaitable[None]] | None = None

# Gemini's not-yet-paced-out audio. `send_bytes` appends, `_pace_loop` drains.
_pending = bytearray()
_pace_task: asyncio.Task[None] | None = None
_pace_started_at: float | None = None
_pace_bytes_sent = 0


def set_frame_listener(listener: Callable[[AudioFrame], Awaitable[None]] | None) -> None:
    """Let the orchestrator (slice 2b+) observe every inbound frame, without
    `transport/inbound.py` needing to know it exists."""
    global _frame_listener
    _frame_listener = listener


async def send_bytes(data: bytes) -> None:
    """Public entry point for server-originated outbound audio (e.g. Gemini
    replies). Queues immediately; `_pace_loop` is what actually puts bytes
    on the wire, metered to realtime."""
    _pending.extend(data)
    metrics.record(metrics.SERVER_QUEUE_DEPTH_MS, len(_pending) / _PACE_BYTES_PER_MS)
    _ensure_pacer_running()


def has_queued_audio() -> bool:
    """True if Gemini audio is still waiting to play at realtime."""
    return bool(_pending)


async def wait_drained() -> None:
    """Wait until the realtime pacer has finished the queued reply."""
    task = _pace_task
    if task is None or task.done():
        return
    await task


def _ensure_pacer_running() -> None:
    global _pace_task
    if _pace_task is None or _pace_task.done():
        _pace_task = asyncio.create_task(_pace_loop())


async def _pace_loop() -> None:
    """Drain `_pending` to the client at exactly 48 bytes/ms.

    Absolute-clock pacing (target = start + bytes_sent / rate), same fix as
    `play_file.py`'s -- sleeping a fixed `chunk_ms` per iteration accumulates
    error and drifts off realtime over a long reply.
    """
    global _pace_started_at, _pace_bytes_sent
    chunks_since_report = 0
    while _pending:
        chunk_len = min(_PACE_CHUNK_BYTES, len(_pending))
        chunk = bytes(_pending[:chunk_len])
        del _pending[:chunk_len]

        if _pace_started_at is None:
            _pace_started_at = time.monotonic()
            _pace_bytes_sent = 0
        _pace_bytes_sent += len(chunk)

        _jitter_buffer.push(chunk)
        await _broadcast_bytes(chunk)

        chunks_since_report += 1
        if chunks_since_report >= _PACE_REPORT_EVERY_N_CHUNKS:
            chunks_since_report = 0
            metrics.record(metrics.BUFFER_DEPTH_MS, _jitter_buffer.depth_ms())
            metrics.record(metrics.SERVER_QUEUE_DEPTH_MS, len(_pending) / _PACE_BYTES_PER_MS)
            events.emit(
                "pace_stats",
                server_queue_ms=len(_pending) / _PACE_BYTES_PER_MS,
                sent_depth_ms=_jitter_buffer.depth_ms(),
            )

        target = _pace_started_at + (_pace_bytes_sent / _PACE_BYTES_PER_MS) / 1000.0
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    # Drained: reset the clock so the next utterance paces from its own
    # start instead of inheriting a stale target from idle time.
    _pace_started_at = None
    _pace_bytes_sent = 0


async def broadcast_control(message: dict[str, Any]) -> None:
    """Public entry point for server-originated JSON control messages other
    than `levels`/`flush` (e.g. the orchestrator's `answer.recorded`, for
    the discovery tile). Text-only -- does not touch the binary pacer."""
    await _broadcast_text(json.dumps(message))


async def broadcast_flush() -> None:
    """Server-originated flush (e.g. a Gemini barge-in). Clears the pacer's
    queue too -- otherwise the next chunk it sends is a leftover from the
    interrupted turn, not new audio. Same wire message as the client-
    initiated flush path in `_handle_inbound_control`."""
    global _pace_started_at, _pace_bytes_sent
    _pending.clear()
    _pace_started_at = None
    _pace_bytes_sent = 0
    _jitter_buffer.reset()
    events.emit("flush_sent")
    await _broadcast_text(json.dumps({"t": "flush"}))


async def ingest(frame: AudioFrame) -> None:
    """Record the latest level and fan out to any registered frame listener
    (e.g. the orchestrator), skipping the bot's own name if Recall reports it."""
    is_bot = frame.participant_name.strip().lower() in _SKIP_NAMES

    if not is_bot and _frame_listener is not None:
        await _frame_listener(frame)

    async with _lock:
        _latest_levels[frame.participant_id] = {
            "id": frame.participant_id,
            "name": frame.participant_name,
            "rms": rms(frame.pcm),
        }


async def _broadcast_text(message: str, *, exclude: WebSocket | None = None) -> None:
    dead: list[WebSocket] = []
    for client in _clients:
        if client is exclude:
            continue
        try:
            await client.send_text(message)
        except Exception:
            dead.append(client)
    for client in dead:
        _clients.discard(client)


async def _broadcast_bytes(data: bytes, *, exclude: WebSocket | None = None) -> None:
    dead: list[WebSocket] = []
    for client in _clients:
        if client is exclude:
            continue
        try:
            await client.send_bytes(data)
        except Exception:
            dead.append(client)
    for client in dead:
        _clients.discard(client)


async def _handle_inbound_audio(pcm: bytes, sender: WebSocket) -> None:
    """Relay one PCM chunk from a sender (e.g. `play_file.py`) to every other client."""
    global _saw_audio
    if not _saw_audio and pcm:
        _saw_audio = True
        events.emit("playback_start", bytes=len(pcm))
    _jitter_buffer.push(pcm)
    metrics.record(metrics.BUFFER_DEPTH_MS, _jitter_buffer.depth_ms())
    await _broadcast_bytes(pcm, exclude=sender)


async def _handle_inbound_control(text: str, sender: WebSocket) -> None:
    global _saw_audio
    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Ignoring non-JSON /ws/meeting control message")
        return

    kind = message.get("t")
    if kind == "flush":
        _jitter_buffer.reset()
        _saw_audio = False
        events.emit("flush_sent")
        logger.info("Flush requested; relaying to other clients")
        await _broadcast_text(json.dumps({"t": "flush"}), exclude=sender)
    elif kind == "stats":
        depth_ms = float(message.get("depth_ms", 0.0))
        underruns = int(message.get("underruns", 0))
        metrics.record(metrics.BUFFER_DEPTH_MS, depth_ms)
        metrics.record(metrics.PLAYBACK_UNDERRUNS, float(underruns))
        events.emit("playback_stats", depth_ms=depth_ms, underruns=underruns)
    elif kind == "flush_ack":
        events.emit(
            "flush_ack",
            depth_ms=float(message.get("depth_ms", 0.0)),
            underruns=int(message.get("underruns", 0)),
            client_mono=message.get("client_mono"),
        )


async def broadcast_loop() -> None:
    """Flush levels to all connected clients at 10Hz.

    Tolerates zero connected clients and runs forever until cancelled.
    """
    while True:
        await asyncio.sleep(_TICK_SECONDS)

        async with _lock:
            levels = list(_latest_levels.values())

        if _clients and levels:
            await _broadcast_text(json.dumps({"t": "levels", "data": levels}))


@router.websocket("/ws/meeting")
async def meeting_socket(websocket: WebSocket) -> None:
    """Shared socket: the meeting page receives levels/audio here, and
    `play_file.py` / the orchestrator send binary PCM (+ occasional flush) here."""
    await websocket.accept()
    _clients.add(websocket)
    logger.info("Meeting page connected (%d client(s))", len(_clients))
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await _handle_inbound_audio(message["bytes"], sender=websocket)
            elif message.get("text") is not None:
                await _handle_inbound_control(message["text"], sender=websocket)
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)
        logger.info("Meeting page disconnected (%d client(s))", len(_clients))
