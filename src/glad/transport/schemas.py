"""Shared data types for audio transport."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioFrame:
    participant_id: int
    participant_name: str
    pcm: bytes
    ts: float


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One FINALIZED transcript utterance from Recall's `transcript.data`
    realtime event. Never built from `transcript.partial_data` -- that
    event type is not subscribed to (see `recall.client`)."""

    participant_id: int
    participant_name: str
    text: str
    ts: float
