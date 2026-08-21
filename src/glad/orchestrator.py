"""Fans out inbound meeting audio to Gemini Live and Gemini's replies to the
outbound page. Owns the session-scoped discovery-question state (one
`SessionState` per bot) and the wake-word / floor-control state machine
that decides when a listening window opens or closes on `LiveSession`.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from glad.agent.ambient import AmbientBuffer
from glad.agent.engagement import EngagementState
from glad.agent.floor import FloorAction, FloorControl, FloorState
from glad.agent.mode import Mode, derive_mode
from glad.agent.prompt import build_system_instruction, roster_context_line
from glad.agent.script import QuestionSet
from glad.agent.self_initiate import SelfInitiate
from glad.agent.speaker import SpeakerTracker
from glad.agent.state import Answer, SessionState
from glad.agent.tools import dispatch
from glad.agent.wakeword import WakeWordResult, describe_verdict, detect as detect_wakeword
from glad.audio.activity import ParticipantActivity
from glad.audio.pcm import gate, rms
from glad.audio.record import inbound_recorder
from glad.config import settings
from glad.live.session import LiveSession
from glad.logging import get_logger
from glad.obs import events, metrics
from glad.transport import outbound
from glad.transport.schemas import AudioFrame, TranscriptSegment

logger = get_logger(__name__)

# Drop a participant's audio from the mix if we haven't heard from them in
# this long (slightly more than one Recall realtime chunk).
_STALE_AFTER_S = 0.25

# RMS threshold above which audio counts as speech, driving FloorControl.
_SPEECH_RMS_THRESHOLD = 0.02

# At most one "[name is speaking]" context message per second.
_SPEAKER_CONTEXT_INTERVAL_S = 1.0

_WAKE_ACK_FALLBACK_TEXT = (
    "[Ambient context: a participant just addressed you by name to get "
    "your attention, but did not say anything else you haven't already "
    "heard. Respond briefly -- do not repeat a question you already asked.]"
)

_SELF_INITIATE_NUDGE = (
    "[The room has been quiet and there are still outstanding discovery "
    "questions. Ask the next one naturally now ({question_id}): {question_text}. "
    "Do not mention this instruction.]"
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
                engagement=EngagementState(
                    ttl_s=settings.engagement_ttl_s,
                    hard_cap_s=settings.engagement_hard_cap_s,
                ),
            )
        }
        # Attached via bind_live() once LiveSession exists (it needs
        # build_instruction/handle_tool_call as callbacks on this instance).
        self._live: LiveSession | None = None
        self._activity = ParticipantActivity(stale_after_s=_STALE_AFTER_S)
        self.is_speaking = False
        self._last_input_sent_at = 0.0

        self._floor = FloorControl()
        self._ambient = AmbientBuffer()
        self._self_initiate = SelfInitiate(
            gap_s=settings.self_initiate_gap_s,
            cooldown_s=settings.self_initiate_cooldown_s,
        )
        self._speech_active = False
        # Running totals reported alongside wake-word events.
        self._wakeword_stage1_total = 0
        self._wakeword_suppressed_total = 0
        self._wakeword_wakes_total = 0
        self._last_speaker_id: int | None = None
        self._last_speaker_name: str | None = None
        self._gated_participants: set[int] = set()
        self._heard_speech = False
        self._suppressing_speech = False
        self._closing_turn = False
        self._awaiting_reply = False
        self._speaker = SpeakerTracker(threshold=_SPEECH_RMS_THRESHOLD)
        self._speaker_context_at = float("-inf")
        self._pending_speaker_line: str | None = None

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
            "self_initiated": "Glad asked a question on its own",
            "stay_engaged": "the exchange is still going",
            "speech_complete": "Glad just finished talking",
            "follow_up": "you talked during the follow-up window",
        }.get(self.engagement.engaged_by or "", self.engagement.engaged_by or "unknown")

    def _flags(self) -> str:
        """Compact state for logs so ENGAGED / floor flips are visible."""
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
        """LiveSession's instruction provider, called on every (re)connect."""
        return build_system_instruction(self._question_set, self.state)

    async def handle_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """LiveSession's tool dispatcher. Runs regardless of engagement
        state -- recording an overheard answer while dormant is intended."""
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

        if name == "stay_engaged":
            self._log_stay_engaged_result(result)
        elif name == "go_dormant":
            self._handle_go_dormant_result(result)
        elif name == "record_answer" and result.get("ok"):
            await self._handle_record_answer_result(args, previous)
        return result

    def _log_tool_call(self, name: str, args: dict[str, Any]) -> None:
        if name == "record_answer":
            logger.info("Recording answer to %s: %r", args.get("question_id"), args.get("value"))
        elif name not in ("stay_engaged", "go_dormant"):
            logger.info("Tool %s %s", name, args)

    def _log_stay_engaged_result(self, result: dict[str, Any]) -> None:
        if result.get("extended"):
            events.emit("engagement.extended")
            logger.info("Glad will keep listening for a follow-up %s", self._flags())
        else:
            logger.info("stay_engaged ignored — Glad is already silent %s", self._flags())

    def _handle_go_dormant_result(self, result: dict[str, Any]) -> None:
        if not result.get("ok"):
            return
        why = result.get("reason")
        if why == "script_complete":
            logger.info("Glad is going quiet — discovery script is done %s", self._flags())
        else:
            logger.info("Glad is going quiet — dismissed (%s) %s", why, self._flags())
        self._closing_turn = False
        if result.get("was_engaged"):
            events.emit("engagement.closed", reason=result.get("reason", "dismissed"))
            if result.get("reason") == "dismissed":
                self._self_initiate.enabled = False
                logger.info("Won't ask another question until someone says Glad")

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
        if not self.state.remaining():
            self._closing_turn = True
            self._suppressing_speech = False
            logger.info("Last question recorded — letting Glad say a short sign-off %s", self._flags())

    async def on_inbound_frame(self, frame: AudioFrame) -> None:
        """Mix concurrent speakers, forward to Gemini, and drive floor
        control. Audio is always forwarded regardless of window state --
        Gemini only retains it as context once a window is open."""
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

        self._sync_engagement(t_recv)
        await self._drive_floor_control(mixed, t_recv)
        await self._track_speaker(t_recv)
        await self._maybe_self_initiate(t_recv)

        await self._live.send_audio(mixed)
        self._last_input_sent_at = time.monotonic()

        elapsed_ms = (self._last_input_sent_at - t_recv) * 1000
        metrics.record(metrics.INBOUND_TO_GEMINI_MS, elapsed_ms)
        events.emit(metrics.INBOUND_TO_GEMINI_MS, value_ms=elapsed_ms)

    async def _track_speaker(self, now: float) -> None:
        """Debounced RMS winner → '[name is speaking]' on the open window."""
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
        """Speech-energy edge detection feeds `FloorControl`; its resulting
        action is applied every frame (`tick()` notices a silence gap)."""
        is_speech = rms(mixed_pcm) >= _SPEECH_RMS_THRESHOLD
        discovery_mode = derive_mode(self.state) is Mode.DISCOVERY
        # ENGAGED speech is a follow-up: no wake word needed.
        follow_up = self.engagement.is_engaged(now)
        take_turn = discovery_mode or follow_up

        action = FloorAction.NONE
        if is_speech and not self._speech_active:
            self._speech_active = True
            if self.is_speaking:
                logger.info("Barge-in — stopping Glad so the room can talk %s", self._flags())
                events.emit("interrupted")
                self.is_speaking = False
                self._suppressing_speech = False
                self._closing_turn = False
                await outbound.broadcast_flush()
                action = self._floor.on_barge_in(now)
                if follow_up:
                    self.engagement.extend("follow_up", now)
            else:
                action = self._floor.on_speech_started(now, discovery_mode=take_turn)
                if follow_up and action is FloorAction.OPEN_WINDOW:
                    self.engagement.extend("follow_up", now)
                    logger.info("Follow-up while ENGAGED — no wake word needed %s", self._flags())
        elif not is_speech and self._speech_active:
            self._speech_active = False
            self._floor.on_speech_ended(now)

        if action is FloorAction.NONE:
            action = self._floor.tick(now)

        await self._apply_floor_action(action)

    def _ttl_paused(self) -> bool:
        """User talking, listen window open, or Glad still owes a reply —
        the follow-up timer must not run, or self-initiate steals the turn."""
        return (
            self.is_speaking
            or self._speech_active
            or self._awaiting_reply
            or self._floor.state in (FloorState.SPEAKING, FloorState.WAKE_PENDING)
        )

    def _sync_engagement(self, now: float | None = None) -> bool:
        """Expire TTL/hard-cap spans and emit engagement.closed once.

        TTL does not run while Glad is speaking, the user is talking, a
        listen window is open, or we are waiting for Gemini to answer.
        """
        now = time.monotonic() if now is None else now
        if self._ttl_paused() and self.engagement.last_engaged_at is not None:
            self.engagement.hold(now)
            if self.engagement.is_engaged(now):
                return True
        reason = self.engagement.poll_expiry(now)
        if reason is not None:
            events.emit("engagement.closed", reason=reason)
            logger.info(
                "ENGAGED → DORMANT (%s) %s",
                "no follow-up after Glad finished talking" if reason == "ttl_expired" else reason.replace("_", " "),
                self._flags(),
            )
        return self.engagement.is_engaged(now)

    def _restart_ttl_after_speech(self) -> None:
        """Start the engagement TTL from the moment Glad finishes talking."""
        if self.engagement.last_engaged_at is None:
            return
        self.engagement.extend("speech_complete")
        events.emit("engagement.extended")
        logger.info("Glad finished talking — 10s follow-up window started %s", self._flags())

    def _floor_busy(self) -> bool:
        """True if any participant is currently mid-utterance."""
        return self._activity.anyone_active(_SPEECH_RMS_THRESHOLD, time.monotonic())

    def _open_engagement(self, trigger: str, now: float | None = None) -> None:
        opened = self.engagement.extend(trigger, now)
        if opened:
            events.emit("engagement.opened", trigger=trigger)
            if trigger == "wake_word":
                logger.info("DORMANT → ENGAGED (wake word) %s", self._flags())
            elif trigger == "self_initiated":
                logger.info("DORMANT → ENGAGED (Glad asked on its own) %s", self._flags())
            else:
                logger.info("DORMANT → ENGAGED (%s) %s", self._why_engaged(), self._flags())
        else:
            events.emit("engagement.extended")
            logger.info("ENGAGED held (%s) %s", self._why_engaged(), self._flags())

    async def _maybe_self_initiate(self, now: float) -> None:
        """Ask the next script question in a natural gap. Rate-limited so an
        ignored question does not retrigger immediately."""
        assert self._live is not None
        if (
            not self._self_initiate.enabled
            or self.is_speaking
            or self.engagement.is_engaged(now)
            or self._awaiting_reply
            or self._speech_active
            or self._floor.state in (FloorState.SPEAKING, FloorState.WAKE_PENDING)
        ):
            self._self_initiate.ready(now, floor_busy=True)
            return
        if not self.state.remaining() or not self._heard_speech:
            return
        if not self._self_initiate.ready(now, floor_busy=self._floor_busy()):
            return

        question = self.state.remaining()[0]
        self._open_engagement("self_initiated", now)
        self._self_initiate.mark_fired(now)
        logger.info("Asking the next question unprompted: %s — %s", question.id, question.text)
        self._awaiting_reply = True
        await self._live.send_text(
            self._with_roster(
                _SELF_INITIATE_NUDGE.format(question_id=question.id, question_text=question.text)
            )
        )

    async def _apply_floor_action(self, action: FloorAction) -> None:
        assert self._live is not None
        if action is FloorAction.OPEN_WINDOW:
            logger.info("Someone started talking — Glad is listening to the room %s", self._flags())
            events.emit("floor.window_opened")
            await self._live.open_window()
            await self._push_roster()
        elif action is FloorAction.CLOSE_WINDOW:
            logger.info("The room went quiet — done listening to that utterance %s", self._flags())
            events.emit("floor.window_closed")
            await self._live.close_window()
            self._awaiting_reply = True
        elif action is FloorAction.WAKE:
            await self._fire_wake()

    async def _fire_wake(self) -> None:
        """The post-wake-word silence gap elapsed: hand the model a text
        turn built from whatever ambient speech was buffered."""
        assert self._live is not None
        text = self._ambient.flush() or _WAKE_ACK_FALLBACK_TEXT
        self._wakeword_wakes_total += 1
        self._self_initiate.enabled = True
        self._open_engagement("wake_word")
        logger.info("Wake word took the floor — handing Glad what was just said %s", self._flags())
        events.emit("floor.wake_fired", injected_chars=len(text), wakes_total=self._wakeword_wakes_total)
        self._awaiting_reply = True
        await self._live.send_text(self._with_roster(text))

    async def on_transcript_segment(self, segment: TranscriptSegment) -> None:
        """Feed one FINAL Recall transcript segment through wake word
        detection (runs regardless of mode), and buffer it for the next
        discovery turn if mode is currently ambient."""
        result = detect_wakeword(segment.text)
        if result.woken:
            logger.info('%s: %r — that sounds like Glad\'s name', segment.participant_name, segment.text)
        else:
            logger.info("%s: %r", segment.participant_name, segment.text)

        if result.stage1_matches:
            self._log_wakeword_match(segment, result)

        if derive_mode(self.state) is Mode.AMBIENT:
            self._ambient.add(segment.participant_name, segment.text, segment.ts)
        self._last_speaker_id = segment.participant_id
        self._last_speaker_name = segment.participant_name
        self._heard_speech = True
        await self._note_participant(segment.participant_id, segment.participant_name)

    async def _note_participant(self, participant_id: int, name: str) -> None:
        """First sighting of a person (audio or transcript) fills the roster
        even when Recall never sent a join — people already in the call
        when the bot arrives. Names reach Gemini on the next listening
        window / text turn, because system_instruction cannot change mid-session."""
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

    def _log_wakeword_match(self, segment: TranscriptSegment, result: WakeWordResult) -> None:
        self._wakeword_stage1_total += 1
        reached_wake_pending = False
        if result.woken:
            reached_wake_pending = self._floor.on_wake_matched(time.monotonic())
            # Engage now, not at _fire_wake. Discovery already sent the
            # utterance in an audio window; Gemini may answer before the
            # 1.2s pause elapses. Waiting left those replies DORMANT.
            self._self_initiate.enabled = True
            self._open_engagement("wake_word")
            if reached_wake_pending:
                logger.info("Floor: idle → WAKE_PENDING (waiting for a pause) %s", self._flags())
            else:
                logger.info(
                    "Heard Glad's name while already listening — reply will play %s",
                    self._flags(),
                )
        else:
            self._wakeword_suppressed_total += 1
            logger.info("Not waking — %s", describe_verdict(result.verdict))

        events.emit(
            "wakeword.stage1_match",
            span=segment.text,
            speaker=segment.participant_name,
            phrase=result.matched_phrase,
            stage2_verdict=result.verdict.value if result.verdict else None,
            suppression_reason=result.suppression_reason,
            reached_wake_pending=reached_wake_pending,
            stage1_total=self._wakeword_stage1_total,
            suppressed_total=self._wakeword_suppressed_total,
            wakes_total=self._wakeword_wakes_total,
        )

    async def run(self) -> None:
        """Drain Gemini's responses forever: forward audio, handle barge-in."""
        assert self._live is not None, "bind_live() must be called before run()"
        async for audio_bytes, is_interrupted, grounded in self._live.responses():
            if is_interrupted:
                if self.is_speaking:
                    logger.info("Someone talked over Glad — stopping playback")
                    self._restart_ttl_after_speech()
                self.is_speaking = False
                self._suppressing_speech = False
                self._closing_turn = False
                self._awaiting_reply = False
                listening = self._speech_active or bool(
                    getattr(self._live, "activity_open", False)
                )
                self._floor.on_interrupted(listening=listening)
                events.emit("interrupted")
                await outbound.broadcast_flush()
                continue

            if not audio_bytes:
                # Turn-complete sentinel: reply ended normally.
                if self.is_speaking:
                    self._restart_ttl_after_speech()
                self.is_speaking = False
                self._suppressing_speech = False
                self._closing_turn = False
                self._awaiting_reply = False
                listening = self._speech_active or bool(
                    getattr(self._live, "activity_open", False)
                )
                self._floor.on_turn_complete(listening=listening)
                continue

            t_recv = time.monotonic()
            self._sync_engagement(t_recv)
            # Finish a turn even if TTL elapsed mid-sentence; barge-in is
            # handled on the inbound speech edge, not by dropping chunks here.
            if self.is_speaking:
                pass
            elif not self.engagement.is_engaged(t_recv) and not self._closing_turn:
                await self._suppress_outbound("dormant")
                continue
            elif self._floor_busy():
                await self._suppress_outbound("floor_busy")
                continue
            else:
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
        """Drop model audio before the outbound path. One event per suppressed span."""
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
