"""Typed request/response models for the Recall.ai bot API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class WebpageConfig(BaseModel):
    url: str


class OutputMediaWebpage(BaseModel):
    kind: str = "webpage"
    config: WebpageConfig


class OutputMedia(BaseModel):
    camera: OutputMediaWebpage


class Variant(BaseModel):
    """Bot compute tier per platform. The default `web` tier (250 millicores)
    is too weak to reliably encode/publish output_media audio, and separate
    per-participant realtime audio is explicitly documented as requiring a
    4-core bot, so `web_4_core` covers both.
    """

    google_meet: str = "web_4_core"


class AudioSeparateRaw(BaseModel):
    """Marker enabling separate raw PCM per participant. Recall's API takes
    a bare `{}` here; there are no configurable fields."""


class RecallAiStreamingConfig(BaseModel):
    """Recall's own streaming transcription provider. `prioritize_low_latency`
    over `prioritize_accuracy`: wake word detection needs FINAL segments to
    arrive quickly, not perfectly."""

    mode: str = "prioritize_low_latency"
    language_code: str = "en"


class TranscriptProvider(BaseModel):
    recallai_streaming: RecallAiStreamingConfig = RecallAiStreamingConfig()


class TranscriptConfig(BaseModel):
    provider: TranscriptProvider = TranscriptProvider()


class RealtimeEndpoint(BaseModel):
    type: Literal["websocket"] = "websocket"
    url: str
    events: list[str]


class RecordingConfig(BaseModel):
    audio_separate_raw: AudioSeparateRaw
    transcript: TranscriptConfig
    realtime_endpoints: list[RealtimeEndpoint]


class CreateBotRequest(BaseModel):
    meeting_url: str
    bot_name: str = "Glad"
    output_media: OutputMedia
    variant: Variant = Variant()
    recording_config: RecordingConfig


class CreateBotResponse(BaseModel):
    id: str
