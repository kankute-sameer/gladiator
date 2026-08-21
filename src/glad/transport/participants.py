"""Receive Recall `participant_events.join` / `.leave` / `.update`.

Decodes the envelope, updates the session roster, emits join/leave. No
other logic -- wake word, floor, and speech stay elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from glad.conversation.session import Roster
from glad.logging import get_logger
from glad.obs import events

logger = get_logger(__name__)

router = APIRouter()

_JOIN = "participant_events.join"
_LEAVE = "participant_events.leave"
_UPDATE = "participant_events.update"
_KINDS: dict[str, Literal["join", "leave", "update"]] = {
    _JOIN: "join",
    _LEAVE: "leave",
    _UPDATE: "update",
}

_roster: Roster | None = None


def set_roster(roster: Roster | None) -> None:
    global _roster
    _roster = roster


@dataclass(frozen=True, slots=True)
class ParticipantEvent:
    kind: Literal["join", "leave", "update"]
    participant_id: int
    name: str
    is_host: bool | None


def _parse_event(message: dict[str, Any]) -> ParticipantEvent | None:
    kind = _KINDS.get(message.get("event", ""))
    if kind is None:
        return None
    participant = message["data"]["data"]["participant"]
    is_host = participant.get("is_host")
    if is_host is not None:
        is_host = bool(is_host)
    return ParticipantEvent(
        kind=kind,
        participant_id=int(participant["id"]),
        name=(participant.get("name") or "Unknown").strip() or "Unknown",
        is_host=is_host,
    )


def _apply(event: ParticipantEvent) -> None:
    roster = _roster
    if roster is None:
        return
    if event.kind == "leave":
        person = roster.leave(event.participant_id, event.name, event.is_host)
        logger.info("%s left the call", person.name)
        events.emit(
            "participant.left",
            participant_id=person.id,
            name=person.name,
            is_host=person.is_host,
            joined_at=person.joined_at,
            left_at=person.left_at,
        )
        return
    if event.kind == "join":
        already = roster.get(event.participant_id)
        was_present = already is not None and already.left_at is None
        person = roster.join(event.participant_id, event.name, event.is_host)
        if was_present:
            return
        logger.info("%s joined the call%s", person.name, " (host)" if person.is_host else "")
        events.emit(
            "participant.joined",
            participant_id=person.id,
            name=person.name,
            is_host=person.is_host,
            joined_at=person.joined_at,
        )
        return
    already = roster.get(event.participant_id)
    first_seen = already is None
    person = roster.update(event.participant_id, event.name, event.is_host)
    if first_seen:
        logger.info("%s joined the call%s", person.name, " (host)" if person.is_host else "")
        events.emit(
            "participant.joined",
            participant_id=person.id,
            name=person.name,
            is_host=person.is_host,
            joined_at=person.joined_at,
        )


@router.websocket("/ws/participants")
async def recall_participants(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Recall participant websocket connected")
    try:
        while True:
            message = await websocket.receive_json()
            event_name = message.get("event")
            try:
                event = _parse_event(message)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed participant message (%s): %s", event_name, exc)
                continue
            if event is None:
                logger.info("Ignoring participant websocket event %s", event_name)
                continue
            _apply(event)
    except WebSocketDisconnect:
        logger.info("Recall participant websocket disconnected")
