"""Receive Recall's real-time `transcript.data` websocket events: decode
envelope -> TranscriptSegment -> fan out to a registered listener.

Only decodes here. No normalization, wake word matching, or buffering --
that's `glad.agent.wakeword` / `glad.agent.ambient`, fed by whatever
registers as the listener (the orchestrator).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from glad.logging import get_logger
from glad.transport.schemas import TranscriptSegment

logger = get_logger(__name__)

router = APIRouter()

_TRANSCRIPT_EVENT = "transcript.data"

_segment_listener: Callable[[TranscriptSegment], Awaitable[None]] | None = None


def set_segment_listener(listener: Callable[[TranscriptSegment], Awaitable[None]] | None) -> None:
    """Let the orchestrator observe every finalized transcript segment,
    without this module needing to know it exists."""
    global _segment_listener
    _segment_listener = listener


def _parse_segment(message: dict[str, Any]) -> TranscriptSegment | None:
    """Decode one `transcript.data` envelope into a TranscriptSegment, or
    None if this message is some other event type. `data.data.words` is a
    list of {text, start_timestamp, end_timestamp} in speaking order;
    joining them with spaces turns word-by-word delivery into one utterance."""
    if message.get("event") != _TRANSCRIPT_EVENT:
        return None

    payload = message["data"]["data"]
    words = payload.get("words") or []
    text = " ".join(w["text"] for w in words if w.get("text")).strip()
    if not text:
        return None

    participant = payload.get("participant") or {}
    ts = 0.0
    if words:
        start = words[0].get("start_timestamp") or {}
        ts = float(start.get("relative") or 0.0)

    return TranscriptSegment(
        participant_id=participant.get("id", -1),
        participant_name=participant.get("name") or "Unknown",
        text=text,
        ts=ts,
    )


async def _segments(websocket: WebSocket) -> AsyncIterator[TranscriptSegment]:
    while True:
        message = await websocket.receive_json()
        try:
            segment = _parse_segment(message)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed transcript message: %s", exc)
            continue
        if segment is not None:
            yield segment


@router.websocket("/ws/transcript")
async def recall_transcript(websocket: WebSocket) -> None:
    """Recall connects here and streams `transcript.data` events (FINAL
    segments only -- `transcript.partial_data` is never subscribed to in
    `recall.client.create_bot`)."""
    await websocket.accept()
    logged_first = False
    try:
        async for segment in _segments(websocket):
            if not logged_first:
                logger.info(
                    "First transcript segment: participant=%s (id=%s): %r",
                    segment.participant_name,
                    segment.participant_id,
                    segment.text,
                )
                logged_first = True
            if _segment_listener is not None:
                await _segment_listener(segment)
    except WebSocketDisconnect:
        logger.info("Recall transcript websocket disconnected")
