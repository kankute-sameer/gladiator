"""Wraps one google-genai Live connection: send raw 16kHz PCM, receive audio
and barge-in signals, dispatch tool calls, and reconnect with session
resumption on drop.

Activity detection is manual (`automatic_activity_detection.disabled=True`):
`glad.conversation.turn.FloorControl` decides when a window opens or closes via
`open_window` / `close_window`, not Gemini's own server-side VAD, so that
ambient background speech never auto-triggers a spoken reply.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosed

from glad.logging import LogThrottle, get_logger
from glad.obs import events, metrics

logger = get_logger(__name__)

# Routes one (tool_name, args) call to its handler; the result is sent back
# to Gemini as the FunctionResponse.response payload.
ToolDispatcher = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
InputTranscriptListener = Callable[[str], Awaitable[None]]

# Gemini Live invents this on empty/echo activity windows. It is not room audio.
_PHANTOM_INPUTS = frozenset(
    {
        "i'm going to go to the store",
        "i am going to go to the store",
        "going to the store",
        "i'm going to the store",
    }
)


def is_phantom_input(text: str) -> bool:
    """True for stock Gemini Live hallucinations, not real speech."""
    lowered = re.sub(r"[^a-z0-9'\s]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", lowered).strip()
    return normalized in _PHANTOM_INPUTS or normalized.endswith("going to go to the store")

_INPUT_MIME_TYPE = "audio/pcm;rate=16000"
_MAX_BACKOFF_S = 30.0

# A reconnect only resets the backoff once the session has stayed up this
# long, so repeated fast-fail cycles don't keep resetting to 1s.
_STABLE_CONNECTION_S = 5.0

_drop_throttle = LogThrottle(interval_s=2.0)


def _close_reason(exc: BaseException) -> str:
    """Best-effort 'code=..., reason=...' description of a websocket close."""
    if isinstance(exc, ConnectionClosed):
        close = exc.rcvd or exc.sent
        if close is not None:
            return f"code={close.code}, reason={close.reason!r}"
        return "code=unknown (no close frame seen)"
    return repr(exc)


class LiveSession:
    """One conversational connection to Gemini Live, reconnecting as needed."""

    def __init__(
        self,
        api_key: str,
        model: str,
        instruction_provider: Callable[[], str],
        tools: Sequence[types.FunctionDeclaration] | None = None,
        tool_dispatcher: ToolDispatcher | None = None,
        voice: str = "Kore",
        input_transcript_listener: InputTranscriptListener | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._voice = voice
        # Called on connect and every reconnect, since a resumed session
        # loses model-side context and needs to be re-told what's known.
        self._instruction_provider = instruction_provider
        self._tools = list(tools) if tools else None
        self._tool_dispatcher = tool_dispatcher
        self._input_transcript_listener = input_transcript_listener
        self._session: genai.live.AsyncSession | None = None
        self._resumption_handle: str | None = None
        self._closed = False
        # `send_audio` and `_handle_tool_call`'s `send_tool_response` write
        # to the same websocket from different tasks; without this lock a
        # tool response sent mid-audio-chunk corrupts the frame (1007).
        self._send_lock = asyncio.Lock()
        self._activity_open = False
        self._pending_context: str | None = None
        self._window_has_content = False
        self._turn_pending = False
        self._open_when_idle = False
        self._discard_turn = False

    @property
    def activity_open(self) -> bool:
        return self._activity_open

    async def send_audio(self, pcm: bytes) -> None:
        """Forward one chunk of 16kHz mono s16le PCM immediately. Drops (with
        a warning) if no session is connected -- never buffers for a reconnect."""
        session = self._session
        if session is None or not pcm:
            if pcm:
                if _drop_throttle.should_log():
                    n = _drop_throttle.take_count()
                    logger.warning(
                        "Dropping inbound audio: no live session connected (%d frame(s) in %.1fs)",
                        n,
                        _drop_throttle.interval_s,
                    )
            return
        try:
            async with self._send_lock:
                await session.send_realtime_input(
                    audio=types.Blob(data=pcm, mime_type=_INPUT_MIME_TYPE)
                )
            if self._activity_open:
                self._window_has_content = True
            _drop_throttle.reset()
        except Exception as exc:
            if _drop_throttle.should_log():
                n = _drop_throttle.take_count()
                logger.warning(
                    "send_audio failed; dropping frame(s) (%s, %d in %.1fs)",
                    _close_reason(exc),
                    n,
                    _drop_throttle.interval_s,
                    exc_info=not isinstance(exc, ConnectionClosed),
                )

    async def open_window(self, *, interrupt: bool = False) -> bool:
        """Mark the start of participant activity. Must be paired with
        `close_window`. Returns True if ActivityStart was sent.

        A new window right after ActivityEnd (before turn_complete) makes
        the Live API close the socket with 1007. Those opens are deferred
        unless `interrupt` is set (barge-in while Glad is talking).
        """
        session = self._session
        if session is None or self._activity_open:
            return False
        if self._turn_pending and not interrupt:
            self._open_when_idle = True
            logger.info("Holding listen window — previous turn is still closing")
            return False
        try:
            async with self._send_lock:
                await session.send_realtime_input(activity_start=types.ActivityStart())
            self._activity_open = True
            self._window_has_content = False
            self._turn_pending = False
            self._open_when_idle = False
        except Exception as exc:
            logger.warning("open_window failed (%s)", _close_reason(exc))
            return False
        await self._flush_pending_context()
        return True

    async def close_window(self) -> bool:
        """Mark the end of participant activity. Returns True if ActivityEnd
        was sent. No-ops if none is open. An empty window (Start then End
        with no audio/text) makes the Live API close the socket (1007), so
        those stays open until something is actually in them."""
        session = self._session
        if session is None or not self._activity_open:
            return False
        if not self._window_has_content:
            logger.info("Keeping listen window open — nothing in it yet")
            return False
        try:
            async with self._send_lock:
                await session.send_realtime_input(activity_end=types.ActivityEnd())
        except Exception as exc:
            logger.warning("close_window failed (%s)", _close_reason(exc))
        finally:
            self._activity_open = False
            self._turn_pending = True
        return True

    async def send_text(self, text: str) -> None:
        """Hand the model a text turn with no activity window at all. This
        is how ambient mode delivers buffered speech without opening a
        raw audio window for it."""
        session = self._session
        if session is None:
            logger.warning("send_text dropped: no live session connected")
            return
        async with self._send_lock:
            await session.send_realtime_input(text=text)

    async def send_context(self, text: str) -> None:
        """Names/facts for the current (or next) audio turn. Sent with
        `send_realtime_input` only while an activity window is open so it
        rides with the user's utterance instead of starting its own reply.
        `send_client_content(turn_complete=False)` is not used: it can
        stall later audio turns on this model."""
        if not text:
            return
        if not self._activity_open or self._session is None:
            self._pending_context = text
            return
        await self._send_realtime_text(text)

    async def _flush_pending_context(self) -> None:
        pending = self._pending_context
        if not pending or not self._activity_open:
            return
        self._pending_context = None
        await self._send_realtime_text(pending)

    async def _send_realtime_text(self, text: str) -> None:
        session = self._session
        if session is None:
            self._pending_context = text
            return
        try:
            async with self._send_lock:
                await session.send_realtime_input(text=text)
        except Exception as exc:
            logger.warning("send_context failed (%s)", _close_reason(exc))
            return
        if self._activity_open:
            self._window_has_content = True
        events.emit("context.sent", text=text, bytes=len(text.encode()))

    async def responses(self) -> AsyncIterator[tuple[bytes, bool, bool]]:
        """Yield (audio_bytes, is_interrupted, grounded) for as long as the
        session runs, reconnecting with exponential backoff on any drop and
        resuming from the last known handle.

        - `interrupted`      -> yields (b"", True, False); caller stops and flushes
        - `model_turn` parts  -> yields (part_bytes, False, grounded) per audio part
        - `turn_complete`     -> yields (b"", False, grounded), end of a reply

        `grounded` is sticky for the current turn: True once any
        `server_content` carries `grounding_metadata`, reset at the next
        `turn_complete`/`interrupted`.
        """
        backoff_s = 1.0
        attempt = 0
        while not self._closed:
            connected_at = time.monotonic()
            turn_grounded = False
            try:
                async with self._client.aio.live.connect(
                    model=self._model, config=self._build_config()
                ) as session:
                    self._session = session
                    self._activity_open = False
                    self._turn_pending = False
                    self._window_has_content = False
                    connected_at = time.monotonic()
                    if self._open_when_idle:
                        await self.open_window()
                    # session.receive() yields exactly one turn then ends;
                    # call it again on the same connection to keep the
                    # conversation (and its context) going.
                    while True:
                        async for message in session.receive():
                            update = message.session_resumption_update
                            if update is not None and update.resumable and update.new_handle:
                                if update.new_handle != self._resumption_handle:
                                    logger.debug("Session resumption handle updated")
                                self._resumption_handle = update.new_handle

                            if message.go_away is not None:
                                logger.warning(
                                    "Live server sent go_away (time_left=%s); will reconnect",
                                    message.go_away.time_left,
                                )

                            content = message.server_content
                            if content is not None and content.input_transcription and content.input_transcription.text:
                                text = content.input_transcription.text
                                phantom = is_phantom_input(text)
                                if phantom:
                                    self._discard_turn = True
                                    logger.info("Heard (Gemini phantom, ignored): %s", text)
                                else:
                                    logger.info("Heard: %s", text)
                                events.emit("input_transcript", text=text, phantom=phantom)
                                if not phantom and self._input_transcript_listener is not None:
                                    await self._input_transcript_listener(text)

                            if message.tool_call is not None:
                                await self._handle_tool_call(session, message.tool_call)

                            if message.tool_call_cancellation is not None:
                                ids = list(message.tool_call_cancellation.ids or [])
                                # We execute tool calls synchronously, so this
                                # usually arrives after the handler already ran.
                                logger.warning(
                                    "Live API cancelled in-flight tool calls on barge-in: ids=%s",
                                    ids,
                                )
                                events.emit(
                                    "tool.cancelled",
                                    ids=ids,
                                    count=len(ids),
                                    already_executed=True,
                                )

                            if content is None:
                                continue

                            if content.output_transcription and content.output_transcription.text:
                                text = content.output_transcription.text
                                logger.info("Gemini said: %s", text)
                                events.emit("output_transcript", text=text)

                            if content.grounding_metadata is not None:
                                turn_grounded = True

                            if content.interrupted:
                                turn_grounded = False
                                self._turn_pending = False
                                self._discard_turn = False
                                yield b"", True, False
                                if self._open_when_idle:
                                    await self.open_window()
                                continue
                            if content.model_turn and not self._discard_turn:
                                for part in content.model_turn.parts or []:
                                    if part.inline_data and part.inline_data.data:
                                        yield part.inline_data.data, False, turn_grounded
                            if content.turn_complete:
                                self._turn_pending = False
                                self._discard_turn = False
                                yield b"", False, turn_grounded
                                turn_grounded = False
                                if self._open_when_idle:
                                    await self.open_window()
            except Exception as exc:
                if self._closed:
                    return
                uptime_s = time.monotonic() - connected_at
                if uptime_s >= _STABLE_CONNECTION_S:
                    attempt = 0
                    backoff_s = 1.0
                attempt += 1
                logger.warning(
                    "Live session dropped after %.1fs (attempt %d, %s); reconnecting in %.1fs",
                    uptime_s,
                    attempt,
                    _close_reason(exc),
                    backoff_s,
                )
                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, _MAX_BACKOFF_S)
            finally:
                self._session = None

    async def _handle_tool_call(
        self, session: genai.live.AsyncSession, tool_call: types.LiveServerToolCall
    ) -> None:
        """Execute every function call in one `tool_call` batch and send all
        responses back together, matched by `id`. Gemini withholds
        `turn_complete` until this is sent, so a stalled dispatcher stalls
        the whole conversation."""
        calls = tool_call.function_calls or []
        if not calls:
            return

        started_at = {call.id: time.monotonic() for call in calls}
        responses: list[types.FunctionResponse] = []
        for call in calls:
            name = call.name or ""
            args = call.args or {}
            if self._discard_turn:
                logger.info("Not applying tool %s — Gemini phantom input", name)
                result = {"ok": False, "error": "ignored: that was not room speech"}
            else:
                result = await self._dispatch_tool(name, args)
            responses.append(types.FunctionResponse(id=call.id, name=name, response=result))

        async with self._send_lock:
            await session.send_tool_response(function_responses=responses)
        sent_at = time.monotonic()

        for call, response in zip(calls, responses):
            elapsed_ms = (sent_at - started_at[call.id]) * 1000
            metrics.record(metrics.TOOL_ROUNDTRIP_MS, elapsed_ms)
            payload = response.response or {}
            events.emit(
                metrics.TOOL_ROUNDTRIP_MS, name=response.name, value_ms=elapsed_ms
            )
            events.emit("tool.called", name=response.name, latency_ms=elapsed_ms)
            if not payload.get("ok"):
                events.emit("tool.rejected", name=response.name, reason=payload.get("error"))

    async def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Never raises into the receive loop -- an exception here must
        become a tool result the model can see, not a dropped connection."""
        if self._tool_dispatcher is None:
            return {"ok": False, "error": f"no tool dispatcher configured for {name!r}"}
        try:
            return await self._tool_dispatcher(name, args)
        except Exception:
            logger.exception("Tool dispatcher raised for %s", name)
            return {"ok": False, "error": "internal error handling tool call"}

    def _build_config(self) -> types.LiveConnectConfig:
        """Built fresh on every connect attempt: the system instruction
        depends on session state that changes as answers come in."""
        tools: list[types.Tool] = [types.Tool(google_search=types.GoogleSearch())]
        if self._tools:
            tools.append(types.Tool(function_declarations=self._tools))
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self._instruction_provider(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self._voice,
                    )
                ),
            ),
            tools=tools,
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(handle=self._resumption_handle),
        )

    async def close(self) -> None:
        self._closed = True
