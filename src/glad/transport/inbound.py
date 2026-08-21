"""Receive Recall's real-time `audio_separate_raw.data` websocket events.

Only decodes: JSON envelope -> base64 -> AudioFrame. No resampling, mixing,
or analysis here — that belongs in `glad.audio`.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from glad.logging import get_logger
from glad.transport import outbound
from glad.transport.schemas import AudioFrame

logger = get_logger(__name__)

router = APIRouter()

_AUDIO_EVENT = "audio_separate_raw.data"


def _parse_frame(message: dict[str, Any]) -> AudioFrame | None:
    """Decode one realtime envelope into an AudioFrame, or None if not audio.
    Example:
    {
    "event": "audio_separate_raw.data",
    "data": {
        "data": {
        "buffer": "GPwY/CT8...",
        "timestamp": { "relative": 12.34, "absolute": "2026-08-21T09:14:02.340Z" },
        "participant": {
            "id": 42,
            "name": "Sameer",
            "is_host": false,
            "platform": "web",
            "email": null,
            "extra_data": {}
        }
        },
        "realtime_endpoint": { "id": "...", "metadata": {} },
        "audio_separate":    { "id": "...", "metadata": {} },
        "recording":         { "id": "...", "metadata": {} },
        "bot":               { "id": "...", "metadata": {} }
    }
    }
    """
    if message.get("event") != _AUDIO_EVENT:
        return None

    payload = message["data"]["data"]
    participant = payload["participant"]
    return AudioFrame(
        participant_id=participant["id"],
        participant_name=participant.get("name") or "Unknown",
        pcm=base64.b64decode(payload["buffer"]),
        ts=payload["timestamp"]["relative"],
    )


async def _frames(websocket: WebSocket) -> AsyncIterator[AudioFrame]:
    while True:
        message = await websocket.receive_json()
        try:
            frame = _parse_frame(message)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed realtime message: %s", exc)
            continue
        if frame is not None:
            yield frame


@router.websocket("/ws/recall")
async def recall_audio(websocket: WebSocket) -> None:
    """Recall connects here and streams `audio_separate_raw.data` events."""
    await websocket.accept()
    logged_first = False
    try:
        async for frame in _frames(websocket):
            if not logged_first:
                logger.info(
                    "First inbound audio frame: participant=%s (id=%s), %d bytes",
                    frame.participant_name,
                    frame.participant_id,
                    len(frame.pcm),
                )
                logged_first = True
            await outbound.ingest(frame)
    except WebSocketDisconnect:
        logger.info("Recall realtime websocket disconnected")
