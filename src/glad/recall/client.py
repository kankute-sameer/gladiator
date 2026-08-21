"""Async client for creating Recall.ai meeting bots."""

from __future__ import annotations

import httpx

from glad.config import settings
from glad.logging import get_logger
from glad.recall.schemas import (
    AudioSeparateRaw,
    CreateBotRequest,
    CreateBotResponse,
    OutputMedia,
    OutputMediaWebpage,
    RealtimeEndpoint,
    RecordingConfig,
    TranscriptConfig,
    WebpageConfig,
)

logger = get_logger(__name__)

_CREATE_BOT_PATH = "/api/v1/bot/"
_AUDIO_SEPARATE_EVENT = "audio_separate_raw.data"
# FINAL segments only -- never `transcript.partial_data`. Wake word stage 1
# matches on finalized utterances; partials would double-fire on every
# growing prefix of the same sentence.
_TRANSCRIPT_EVENT = "transcript.data"
_PARTICIPANT_EVENTS = (
    "participant_events.join",
    "participant_events.leave",
    "participant_events.update",
)


class RecallAPIError(Exception):
    """Raised when the Recall.ai API returns a non-2xx response."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Recall API request failed with status {status_code}: {body}")


class RecallClient:
    """Minimal async client for the Recall.ai bot API."""

    def __init__(self) -> None:
        self._base_url = settings.recall_base_url.rstrip("/")
        self._api_key = settings.recall_api_key.get_secret_value()

    async def create_bot(self, meeting_url: str) -> str:
        """Create a bot that joins `meeting_url`, streams `settings.public_url`
        as its camera, and streams per-participant audio + finalized
        transcript segments back over the same public tunnel."""
        realtime_url = f"{settings.public_ws_url}/ws/recall"
        transcript_url = f"{settings.public_ws_url}/ws/transcript"
        participants_url = f"{settings.public_ws_url}/ws/participants"
        request = CreateBotRequest(
            meeting_url=meeting_url,
            output_media=OutputMedia(
                camera=OutputMediaWebpage(config=WebpageConfig(url=settings.public_url)),
            ),
            recording_config=RecordingConfig(
                audio_separate_raw=AudioSeparateRaw(),
                transcript=TranscriptConfig(),
                realtime_endpoints=[
                    RealtimeEndpoint(url=realtime_url, events=[_AUDIO_SEPARATE_EVENT]),
                    RealtimeEndpoint(url=transcript_url, events=[_TRANSCRIPT_EVENT]),
                    RealtimeEndpoint(url=participants_url, events=list(_PARTICIPANT_EVENTS)),
                ],
            ),
        )

        async with httpx.AsyncClient(base_url=self._base_url, timeout=30.0) as client:
            response = await client.post(
                _CREATE_BOT_PATH,
                headers={"Authorization": self._api_key},
                json=request.model_dump(mode="json"),
            )

        if not response.is_success:
            logger.error("create_bot failed with status %s: %s", response.status_code, response.text)
            raise RecallAPIError(response.status_code, response.text)

        data = CreateBotResponse.model_validate(response.json())
        logger.info("Created Recall bot %s for meeting %s", data.id, meeting_url)
        return data.id

    async def get_bot(self, bot_id: str) -> dict:
        """Retrieve one bot (includes `recordings` once the call has media)."""
        async with httpx.AsyncClient(base_url=self._base_url, timeout=30.0) as client:
            response = await client.get(
                f"{_CREATE_BOT_PATH}{bot_id}/",
                headers={"Authorization": self._api_key},
            )
        if not response.is_success:
            logger.error("get_bot failed with status %s: %s", response.status_code, response.text)
            raise RecallAPIError(response.status_code, response.text)
        return response.json()

    async def list_audio_separate(self, recording_id: str) -> dict:
        """List post-call per-participant audio artifacts for a recording."""
        async with httpx.AsyncClient(base_url=self._base_url, timeout=30.0) as client:
            response = await client.get(
                "/api/v1/audio_separate",
                params={"recording_id": recording_id},
                headers={"Authorization": self._api_key},
            )
        if not response.is_success:
            logger.error(
                "list_audio_separate failed with status %s: %s",
                response.status_code,
                response.text,
            )
            raise RecallAPIError(response.status_code, response.text)
        return response.json()
