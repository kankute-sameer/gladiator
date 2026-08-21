"""Inbound meeting audio/transcripts -> Gemini Live -> outbound page.

DORMANT: wake-word detection on Recall transcripts only. No audio or text
is sent to Gemini. ENGAGED (after wake): audio, activity windows, barge-in.
Gemini stays in the conversation until it calls go_dormant.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from glad.audio.activity import ParticipantActivity
from glad.audio.pcm import gate, rms
from glad.audio.record import inbound_recorder
from glad.config import settings
from glad.conversation.prompt import build_system_instruction, roster_context_line
from glad.conversation.session import Answer, QuestionSet, SessionState
from glad.conversation.tools import dispatch
from glad.conversation.turn import (
    AmbientBuffer,
    EngagementState,
    FloorAction,
    FloorControl,
    FloorState,
    SpeakerTracker,
)
from glad.conversation.wakeword import WakeWordResult, describe_verdict, detect as detect_wakeword
from glad.live.session import LiveSession
from glad.logging import get_logger
from glad.obs import events, metrics
from glad.transport import outbound
from glad.transport.schemas import AudioFrame, TranscriptSegment

logger = get_logger(__name__)

_STALE_AFTER_S = 0.25
_SPEECH_RMS_THRESHOLD = 0.02
_ECHO_HOLDOFF_S = 0.6
_SPEAKER_CONTEXT_INTERVAL_S = 1.0

_WAKE_ACK_FALLBACK_TEXT = (
    "[A participant just addressed you by name. Respond briefly -- do not "
    "repeat a question you already asked.]"
)


class Orchestrator:
    """Wires inbound AudioFrames -> Gemini Live -> outbound page audio."""

    def __init__(self, question_set: QuestionSet, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        self._question_set = question_set
        self._states: dict[str, SessionState] = {
            self.session_id: SessionState(
                session_id=self.session_id,
                question_set=question_set,
                engagement=EngagementState(),
            )
        }
        self._live: LiveSession | None = None
        self._activity = ParticipantActivity(stale_after_s=_STALE_AFTER_S)
        self.is_speaking = False
        self._last_input_sent_at = 0.0

        self._floor = FloorControl()
        self._ambient = AmbientBuffer()
        self._speech_active = False
        self._wakeword_stage1_total = 0
        self._wakeword_suppressed_total = 0
        self._wakeword_wakes_total = 0
        self._last_speaker_id: int | None = None
        self._last_speaker_name: str | None = None
        self._gated_participants: set[int] = set()
        self._heard_speech = False
        self._suppressing_speech = False
        self._awaiting_reply = False
        self._speaker = SpeakerTracker(threshold=_SPEECH_RMS_THRESHOLD)
        self._speaker_context_at = float("-inf")
        self._pending_speaker_line: str | None = None
        self._echo_holdoff_until = 0.0
        self._pending_dormant: dict[str, Any] | None = None
        self._finish_current_line = False

    @property
    def state(self) -> SessionState:
        return self._states[self.session_id]

    @property
    def engagement(self) -> EngagementState:
        return self.state.engagement

    def _questions_left(self) -> str:
        n = len(self.state.remaining())
        if n == 0:
            return "all questions answered"
        return "1 question left" if n == 1 else f"{n} questions left"

    def _why_engaged(self) -> str:
        return {
            "wake_word": "someone said Glad's name",
            "follow_up": "you talked during the follow-up window",
        }.get(self.engagement.engaged_by or "", self.engagement.engaged_by or "unknown")

    def _flags(self) -> str:
        engaged = "ENGAGED" if self.engagement.is_engaged() else "DORMANT"
        floor = {
            "ambient": "idle",
            "wake_pending": "WAKE_PENDING",
            "speaking": "SPEAKING",
            "yielded": "YIELDED",
        }.get(self._floor.state.value, self._floor.state.value)
        return f"[{engaged} | floor={floor} | {self._questions_left()}]"

    def bind_live(self, live: LiveSession) -> None:
        self._live = live

    def build_instruction(self) -> str:
        return build_system_instruction(self._question_set, self.state)

    async def handle_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "record_answer":
            speaker_id = self._speaker.current_speaker() or self._last_speaker_id
            speaker_name = self._last_speaker_name
            if speaker_id is not None:
                person = self.state.roster.get(speaker_id)
                if person is not None:
                    speaker_name = person.name
            args = {
                **args,
                "participant_id": speaker_id,
                "participant_name": speaker_name,
            }
        previous = self.state.answers.get(args.get("question_id")) if name == "record_answer" else None
        self._log_tool_call(name, args)
        result = dispatch(name, args, self.state)

        if name == "go_dormant":
            await self._handle_go_dormant_result(result)
        elif name == "record_answer" and result.get("ok"):
            await self._handle_record_answer_result(args, previous)
        return result

    def _log_tool_call(self, name: str, args: dict[str, Any]) -> None:
        if name == "record_answer":
            logger.info(
                "Tool record_answer — %s: %r",
                args.get("question_id"),
                args.get("value"),
            )
        elif name == "go_dormant":
            logger.info("Tool go_dormant — %s", args.get("reason"))
        else:
            logger.info("Tool %s %s", name, args)

    async def _handle_go_dormant_result(self, result: dict[str, Any]) -> None:
        if not result.get("ok"):
            return
        still_playing = self.is_speaking or outbound.has_queued_audio()
        if still_playing:
            self._pending_dormant = result
            self._finish_current_line = True
            logger.info("Letting Glad finish this line before going quiet %s", self._flags())
            if not self.is_speaking:
                asyncio.create_task(self._seal_dormant_after_playback())
            return
        await self._apply_go_dormant(result)

    async def _seal_dormant_after_playback(self) -> None:
        await outbound.wait_drained()
        pending = self._pending_dormant
        if pending is None:
            return
        self._pending_dormant = None
        self._finish_current_line = False
        if self.is_speaking:
            self._echo_holdoff_until = time.monotonic() + _ECHO_HOLDOFF_S
        await self._apply_go_dormant(pending, flush=False)

    async def _apply_go_dormant(self, result: dict[str, Any], *, flush: bool = True) -> None:
        why = result.get("reason")
        detail = result.get("detail") or why
        if why == "script_complete":
            logger.info("Glad is going quiet — discovery script is done %s", self._flags())
        elif why == "not_for_me":
            logger.info("Glad is going quiet — this isn't for Glad (%s) %s", detail, self._flags())
        else:
            logger.info("Glad is going quiet — dismissed (%s) %s", detail, self._flags())
        self.is_speaking = False
        self._awaiting_reply = False
        self._suppressing_speech = False
        self._finish_current_line = False
        self._pending_dormant = None
        if flush:
            await outbound.broadcast_flush()
        if self._live is not None and self._live.activity_open:
            await self._live.close_window()
        self._floor.on_turn_complete(listening=False)
        if result.get("was_engaged"):
            events.emit("engagement.closed", reason=why or "dismissed")

    async def _handle_record_answer_result(self, args: dict[str, Any], previous: Answer | None) -> None:
        answer = self.state.answers.get(args.get("question_id"))
        if answer is None:
            return
        events.emit(
            "answer.recorded",
            question_id=answer.question_id,
            revision=answer.revision,
            value=answer.value,
            participant_id=answer.participant_id,
            participant_name=answer.participant_name,
        )
        if (
            previous is not None
            and previous.participant_id is not None
            and answer.participant_id is not None
            and previous.participant_id != answer.participant_id
        ):
            events.emit(
                "answer.conflicted",
                question_id=answer.question_id,
                previous_participant=previous.participant_name or previous.participant_id,
                new_participant=answer.participant_name or answer.participant_id,
            )
        await outbound.broadcast_control(
            {
                "t": "answer.recorded",
                "question_id": answer.question_id,
                "value": answer.value,
                "revision": answer.revision,
            }
        )

    async def on_inbound_frame(self, frame: AudioFrame) -> None:
        """Mix speakers, drive floor/wake, and forward to Gemini only while ENGAGED."""
        assert self._live is not None, "bind_live() must be called before frames arrive"
        t_recv = time.monotonic()
        await self._note_participant(frame.participant_id, frame.participant_name)
        level = self._activity.observe(frame.participant_id, frame.pcm, t_recv)
        if level >= _SPEECH_RMS_THRESHOLD:
            self._heard_speech = True
            self._last_speaker_id = frame.participant_id
            self._last_speaker_name = frame.participant_name

        threshold = settings.audio_gate_threshold
        if not gate(frame.pcm, threshold):
            if frame.participant_id not in self._gated_participants:
                events.emit("audio.gated", participant_id=frame.participant_id, rms=level)
                self._gated_participants.add(frame.participant_id)
        else:
            self._gated_participants.discard(frame.participant_id)

        mixed = self._activity.mix(threshold)
        inbound_recorder.write(mixed)

        await self._drive_floor_control(mixed, t_recv)

        if not self.engagement.is_engaged(t_recv):
            return

        await self._track_speaker(t_recv)
        if self.is_speaking:
            return
        await self._live.send_audio(mixed)
        self._last_input_sent_at = time.monotonic()

        elapsed_ms = (self._last_input_sent_at - t_recv) * 1000
        metrics.record(metrics.INBOUND_TO_GEMINI_MS, elapsed_ms)
        events.emit(metrics.INBOUND_TO_GEMINI_MS, value_ms=elapsed_ms)

    async def _track_speaker(self, now: float) -> None:
        changed_id = self._speaker.update(self._activity.levels(now), now)
        if changed_id is not None:
            person = self.state.roster.get(changed_id)
            name = person.name if person is not None else self._last_speaker_name
            if name and name.strip().lower() != "glad":
                rms_level = self._activity.levels(now).get(changed_id, 0.0)
                self._pending_speaker_line = f"[{name} is speaking]"
                logger.info("%s is speaking %s", name, self._flags())
                events.emit(
                    "speaker.changed",
                    participant_id=changed_id,
                    name=name,
                    rms=rms_level,
                )
        await self._flush_speaker_context(now)

    async def _flush_speaker_context(self, now: float) -> None:
        if self._pending_speaker_line is None or self._live is None:
            return
        if now - self._speaker_context_at < _SPEAKER_CONTEXT_INTERVAL_S:
            return
        line = self._pending_speaker_line
        self._pending_speaker_line = None
        self._speaker_context_at = now
        await self._live.send_context(line)

    async def _drive_floor_control(self, mixed_pcm: bytes, now: float) -> None:
        is_speech = rms(mixed_pcm) >= _SPEECH_RMS_THRESHOLD
        if not self.is_speaking and now < self._echo_holdoff_until:
            is_speech = False
        engaged = self.engagement.is_engaged(now)

        action = FloorAction.NONE
        barge_in = False
        if is_speech and not self._speech_active:
            self._speech_active = True
            if self.is_speaking:
                barge_in = True
                logger.info("Barge-in — stopping Glad so the room can talk %s", self._flags())
                events.emit("interrupted")
                self.is_speaking = False
                self._suppressing_speech = False
                self._finish_current_line = False
                await outbound.broadcast_flush()
                if self._pending_dormant is not None:
                    await self._apply_go_dormant(self._pending_dormant, flush=False)
                action = self._floor.on_barge_in(now)
            else:
                action = self._floor.on_speech_started(now, discovery_mode=engaged)
                if engaged and action is FloorAction.OPEN_WINDOW:
                    logger.info("Follow-up while ENGAGED — no wake word needed %s", self._flags())
        elif not is_speech and self._speech_active:
            self._speech_active = False
            self._floor.on_speech_ended(now)

        if action is FloorAction.NONE:
            action = self._floor.tick(now)

        await self._apply_floor_action(action, interrupt=barge_in)

    def _floor_busy(self) -> bool:
        return self._activity.anyone_active(_SPEECH_RMS_THRESHOLD, time.monotonic())

    def _open_engagement(self, trigger: str, now: float | None = None) -> None:
        opened = self.engagement.extend(trigger, now)
        if opened:
            events.emit("engagement.opened", trigger=trigger)
            logger.info("DORMANT → ENGAGED (%s) %s", self._why_engaged(), self._flags())
        else:
            events.emit("engagement.extended")
            logger.info("ENGAGED held (%s) %s", self._why_engaged(), self._flags())

    async def _apply_floor_action(self, action: FloorAction, *, interrupt: bool = False) -> None:
        assert self._live is not None
        if action is FloorAction.OPEN_WINDOW:
            logger.info("Someone started talking — Glad is listening to the room %s", self._flags())
            events.emit("floor.window_opened")
            await self._live.open_window(interrupt=interrupt)
            await self._push_roster()
        elif action is FloorAction.CLOSE_WINDOW:
            logger.info("The room went quiet — done listening to that utterance %s", self._flags())
            events.emit("floor.window_closed")
            ended = await self._live.close_window()
            if ended:
                self._awaiting_reply = True
        elif action is FloorAction.WAKE:
            await self._fire_wake()

    async def _fire_wake(self) -> None:
        assert self._live is not None
        text = self._ambient.flush() or _WAKE_ACK_FALLBACK_TEXT
        self._wakeword_wakes_total += 1
        self._open_engagement("wake_word")
        logger.info("Wake word took the floor — handing Glad what was just said %s", self._flags())
        events.emit("floor.wake_fired", injected_chars=len(text), wakes_total=self._wakeword_wakes_total)
        self._awaiting_reply = True
        await self._live.send_text(self._with_roster(text))

    async def on_transcript_segment(self, segment: TranscriptSegment) -> None:
        result = detect_wakeword(segment.text)
        if result.woken:
            logger.info('%s: %r — that sounds like Glad\'s name', segment.participant_name, segment.text)
        else:
            logger.info("%s: %r", segment.participant_name, segment.text)

        if result.stage1_matches:
            self._apply_wakeword(segment.text, result, speaker=segment.participant_name)

        if not self.engagement.is_engaged():
            self._ambient.add(segment.participant_name, segment.text, segment.ts)
        self._last_speaker_id = segment.participant_id
        self._last_speaker_name = segment.participant_name
        self._heard_speech = True
        await self._note_participant(segment.participant_id, segment.participant_name)

    async def _note_participant(self, participant_id: int, name: str) -> None:
        if not name or name.strip().lower() == "glad":
            return
        if not self.state.roster.note(participant_id, name):
            return
        person = self.state.roster.get(participant_id)
        logger.info("%s is in the call %s", name, self._flags())
        events.emit(
            "participant.joined",
            participant_id=participant_id,
            name=name,
            is_host=person.is_host if person else None,
            joined_at=person.joined_at if person else None,
        )

    def _with_roster(self, text: str) -> str:
        line = roster_context_line(self.state)
        return f"{line}\n{text}" if line else text

    async def _push_roster(self) -> None:
        line = roster_context_line(self.state)
        if line is None or self._live is None:
            return
        await self._live.send_context(line)

    def _apply_wakeword(self, text: str, result: WakeWordResult, *, speaker: str) -> None:
        self._wakeword_stage1_total += 1
        reached_wake_pending = False
        if result.woken:
            reached_wake_pending = self._floor.on_wake_matched(time.monotonic())
            if reached_wake_pending:
                logger.info("Floor: idle → WAKE_PENDING (waiting for a pause) %s", self._flags())
            else:
                logger.info("Heard Glad's name while already in a turn %s", self._flags())
                self._open_engagement("wake_word")
        else:
            self._wakeword_suppressed_total += 1
            logger.info("Not waking — %s", describe_verdict(result.verdict))

        events.emit(
            "wakeword.stage1_match",
            span=text,
            speaker=speaker,
            phrase=result.matched_phrase,
            stage2_verdict=result.verdict.value if result.verdict else None,
            suppression_reason=result.suppression_reason,
            reached_wake_pending=reached_wake_pending,
            stage1_total=self._wakeword_stage1_total,
            suppressed_total=self._wakeword_suppressed_total,
            wakes_total=self._wakeword_wakes_total,
        )

    async def run(self) -> None:
        assert self._live is not None, "bind_live() must be called before run()"
        async for audio_bytes, is_interrupted, grounded in self._live.responses():
            if is_interrupted:
                if self.is_speaking:
                    logger.info("Someone talked over Glad — stopping playback")
                self.is_speaking = False
                self._suppressing_speech = False
                self._awaiting_reply = False
                self._finish_current_line = False
                listening = self._speech_active or bool(
                    getattr(self._live, "activity_open", False)
                )
                self._floor.on_interrupted(listening=listening)
                events.emit("interrupted")
                await outbound.broadcast_flush()
                if self._pending_dormant is not None:
                    await self._apply_go_dormant(self._pending_dormant, flush=False)
                continue

            if not audio_bytes:
                if self._pending_dormant is not None:
                    await self._seal_dormant_after_playback()
                    continue
                if self.is_speaking:
                    self._echo_holdoff_until = time.monotonic() + _ECHO_HOLDOFF_S
                self.is_speaking = False
                self._suppressing_speech = False
                self._awaiting_reply = False
                listening = self._speech_active or bool(
                    getattr(self._live, "activity_open", False)
                )
                self._floor.on_turn_complete(listening=listening)
                continue

            t_recv = time.monotonic()
            if not self.engagement.is_engaged(t_recv) and not self._finish_current_line:
                await self._suppress_outbound("dormant")
                continue
            if self._floor_busy() and not self.is_speaking:
                await self._suppress_outbound("floor_busy")
                continue
            self._suppressing_speech = False

            if not self.is_speaking:
                self.is_speaking = True
                first_byte_ms = (t_recv - self._last_input_sent_at) * 1000
                metrics.record(metrics.GEMINI_FIRST_BYTE_MS, first_byte_ms)
                metrics.record(
                    metrics.GEMINI_FIRST_BYTE_MS_GROUNDED if grounded else metrics.GEMINI_FIRST_BYTE_MS_UNGROUNDED,
                    first_byte_ms,
                )
                events.emit(metrics.GEMINI_FIRST_BYTE_MS, value_ms=first_byte_ms, grounded=grounded)

            await outbound.send_bytes(audio_bytes)
            elapsed_ms = (time.monotonic() - t_recv) * 1000
            metrics.record(metrics.GEMINI_TO_OUTBOUND_MS, elapsed_ms)
            events.emit(metrics.GEMINI_TO_OUTBOUND_MS, value_ms=elapsed_ms)

    async def _suppress_outbound(self, reason: str) -> None:
        if not self._suppressing_speech:
            events.emit("speech.suppressed", reason=reason)
            self._suppressing_speech = True
            logger.info(
                "Not playing Glad's audio — %s %s",
                "DORMANT" if reason == "dormant" else "floor busy (someone else is talking)",
                self._flags(),
            )
        if self.is_speaking:
            self.is_speaking = False
            await outbound.broadcast_flush()
