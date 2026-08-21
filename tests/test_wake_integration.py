"""Integration tests: wake word detection (glad.agent.wakeword) + floor
control (glad.agent.floor) wired together through the real Orchestrator,
driven via its public `on_transcript_segment` / `on_inbound_frame` entry
points -- the same ones `transport/transcript_inbound.py` and
`transport/outbound.py` call in production.

`LiveSession` itself is replaced with a recording fake: these tests are
about whether floor control reaches the right state and calls the right
primitive at the right time, not about the real Gemini wire.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from glad.agent.floor import FloorAction, FloorState
from glad.agent.script import Question, QuestionSet
from glad.orchestrator import Orchestrator
from glad.transport.schemas import AudioFrame, TranscriptSegment

_QUESTION_SET = QuestionSet(
    id="test_set",
    version=1,
    questions=(
        Question(id="budget", text="What's your budget?"),
        Question(id="timeline", text="What's your timeline?"),
    ),
)

_LOUD_PCM = (b"\x00\x40\x00\xc0" * 200)  # alternating +/- 16384, well above the RMS threshold
_SILENT_PCM = b"\x00\x00" * 400


@dataclass
class FakeLiveSession:
    """Records every call instead of touching a real websocket."""

    audio_sent: list[bytes] = field(default_factory=list)
    windows_opened: int = 0
    windows_closed: int = 0
    texts_sent: list[str] = field(default_factory=list)
    activity_open: bool = False

    async def send_audio(self, pcm: bytes) -> None:
        self.audio_sent.append(pcm)

    async def open_window(self) -> None:
        self.windows_opened += 1
        self.activity_open = True

    async def close_window(self) -> None:
        self.windows_closed += 1
        self.activity_open = False

    async def send_text(self, text: str) -> None:
        self.texts_sent.append(text)

    async def send_context(self, text: str) -> None:
        self.texts_sent.append(text)


def _orchestrator() -> tuple[Orchestrator, FakeLiveSession]:
    orch = Orchestrator(_QUESTION_SET, session_id="itest")
    fake = FakeLiveSession()
    orch.bind_live(fake)  # type: ignore[arg-type]
    return orch, fake


async def _speech_frame(orch: Orchestrator, now: float, participant_id: int = 1) -> None:
    frame = AudioFrame(participant_id=participant_id, participant_name="Alice", pcm=_LOUD_PCM, ts=now)
    await orch._drive_floor_control(frame.pcm, now)


async def _silence_frame(orch: Orchestrator, now: float) -> None:
    await orch._drive_floor_control(_SILENT_PCM, now)


@pytest.mark.asyncio
async def test_first_audio_frame_seeds_roster_and_sends_context() -> None:
    """Recall join events often never fire for people already in the call.
    First inbound frame is enough to name them to Gemini."""
    orch, fake = _orchestrator()
    frame = AudioFrame(participant_id=7, participant_name="Sameer Kankute", pcm=_LOUD_PCM, ts=1.0)
    await orch.on_inbound_frame(frame)

    present = orch.state.roster.present()
    assert len(present) == 1
    assert present[0].name == "Sameer Kankute"
    assert any("Sameer Kankute" in text and "Do not reply" in text for text in fake.texts_sent)

    await orch.on_inbound_frame(frame)
    assert len([t for t in fake.texts_sent if "Sameer Kankute" in t]) == 1


@pytest.mark.asyncio
async def test_wake_word_while_participant_keeps_talking_stays_wake_pending() -> None:
    """AMBIENT mode (every question answered). A wake word arrives, then the
    same speaker keeps talking for several seconds -- floor control must
    stay WAKE_PENDING throughout and never open a window / interrupt."""
    orch, fake = _orchestrator()
    orch.state.record("budget", "$50k")
    orch.state.record("timeline", "next quarter")  # -> AMBIENT mode

    now = 0.0
    await orch.on_transcript_segment(
        TranscriptSegment(participant_id=1, participant_name="Alice", text="Glad, what do you think?", ts=now)
    )
    assert orch._floor.state is FloorState.WAKE_PENDING
    assert orch.engagement.is_engaged() is True
    assert fake.texts_sent == []  # not fired yet

    # The same speaker keeps talking for 8 more seconds (on/off speech,
    # never a sustained silence gap).
    for _ in range(16):
        now += 0.1
        await _speech_frame(orch, now)
        assert orch._floor.state is FloorState.WAKE_PENDING
        now += 0.4
        await _silence_frame(orch, now)
        assert orch._floor.state is FloorState.WAKE_PENDING  # gap kept resetting

    assert fake.texts_sent == []
    assert fake.windows_opened == 0

    # Now real silence, long enough to clear the gap threshold.
    now += orch._floor.gap_threshold_s + 0.2
    await _silence_frame(orch, now)

    assert orch._floor.state is FloorState.SPEAKING
    assert len(fake.texts_sent) == 1
    assert "what do you think" in fake.texts_sent[0].lower()
    assert fake.windows_opened == 0  # ambient wake path never opens a raw-audio window


@pytest.mark.asyncio
async def test_wake_word_during_discovery_mode_still_fires() -> None:
    """DISCOVERY mode (a question is still outstanding). A wake word
    arriving while the floor is otherwise idle must still reach
    WAKE_PENDING and eventually fire -- discovery mode does not disable
    wake word detection, it just also allows ordinary speech to open a
    window directly."""
    orch, fake = _orchestrator()
    # No answers recorded -- mode is DISCOVERY.

    now = 0.0
    await orch.on_transcript_segment(
        TranscriptSegment(participant_id=2, participant_name="Bob", text="Hey Glad, can you help with this?", ts=now)
    )
    assert orch._floor.state is FloorState.WAKE_PENDING
    assert orch.engagement.is_engaged() is True

    now += orch._floor.gap_threshold_s + 0.2
    await _silence_frame(orch, now)

    assert orch._floor.state is FloorState.SPEAKING
    assert len(fake.texts_sent) == 1


@pytest.mark.asyncio
async def test_adjectival_glad_during_ambient_never_reaches_wake_pending() -> None:
    orch, fake = _orchestrator()
    orch.state.record("budget", "$50k")
    orch.state.record("timeline", "next quarter")  # -> AMBIENT mode

    await orch.on_transcript_segment(
        TranscriptSegment(participant_id=1, participant_name="Alice", text="so glad you called", ts=0.0)
    )
    assert orch._floor.state is FloorState.AMBIENT
    assert orch._wakeword_suppressed_total == 1
    assert orch._wakeword_wakes_total == 0

    # Suppressed ambient chatter is still buffered as context for the next
    # discovery turn.
    assert len(orch._ambient) == 1


@pytest.mark.asyncio
async def test_discovery_speech_opens_and_closes_a_real_window() -> None:
    """Normal discovery turn-taking: no wake word needed at all."""
    orch, fake = _orchestrator()
    # An unanswered question exists -> DISCOVERY mode.

    now = 0.0
    await _speech_frame(orch, now)
    assert orch._floor.state is FloorState.SPEAKING
    assert fake.windows_opened == 1

    now += 1.0
    await _silence_frame(orch, now)
    assert fake.windows_closed == 0  # endpoint gap not elapsed yet

    now += orch._floor.endpoint_gap_s + 0.2
    await _silence_frame(orch, now)
    assert fake.windows_closed == 1
    assert orch._floor.state is FloorState.AMBIENT


@pytest.mark.asyncio
async def test_speech_while_engaged_opens_window_without_wake_word() -> None:
    """After the script is done, ambient mode normally needs a wake word.
    During the ENGAGED follow-up window, ordinary speech must still open
    a window so Glad can answer 'no I can't' without being addressed."""
    orch, fake = _orchestrator()
    orch.state.record("budget", "$50k")
    orch.state.record("timeline", "next quarter")

    now = time.monotonic()
    orch.engagement.extend("wake_word", now)
    await _speech_frame(orch, now)

    assert fake.windows_opened == 1
    assert orch._floor.state is FloorState.SPEAKING
    assert orch.engagement.is_engaged(now) is True


@pytest.mark.asyncio
async def test_ambient_speech_while_dormant_still_needs_a_wake_word() -> None:
    orch, fake = _orchestrator()
    orch.state.record("budget", "$50k")
    orch.state.record("timeline", "next quarter")

    await _speech_frame(orch, 0.0)
    assert fake.windows_opened == 0
    assert orch._floor.state is FloorState.AMBIENT


@pytest.mark.asyncio
async def test_wake_word_during_open_window_engages_so_the_reply_can_play() -> None:
    """Discovery already opened a listen window. The wake transcript must
    still ENGAGE — otherwise Gemini answers and we drop it as DORMANT."""
    orch, fake = _orchestrator()
    now = 0.0
    await _speech_frame(orch, now)
    assert orch._floor.state is FloorState.SPEAKING
    assert orch.engagement.is_engaged(now) is False

    await orch.on_transcript_segment(
        TranscriptSegment(
            participant_id=1,
            participant_name="Alice",
            text="Glad, can you tell me my name?",
            ts=now,
        )
    )
    assert orch.engagement.is_engaged() is True
    assert fake.texts_sent == []


@pytest.mark.asyncio
async def test_user_speech_holds_engagement_past_ttl() -> None:
    """Asking 'what is my name' mid-follow-up used to expire TTL while the
    window was still open, then self-initiate stole the turn."""
    orch, fake = _orchestrator()
    orch.engagement.extend("wake_word", 100.0)
    orch._speech_active = True
    orch._floor.state = FloorState.SPEAKING

    assert orch._sync_engagement(115.0) is True
    assert orch.engagement.is_engaged(115.0) is True
    assert fake.texts_sent == []


@pytest.mark.asyncio
async def test_close_window_waits_for_reply_instead_of_self_initiating() -> None:
    orch, fake = _orchestrator()
    orch._heard_speech = True
    orch.engagement.dismiss("ttl_expired")
    orch._self_initiate._floor_free_since = 0.0
    orch._self_initiate._last_fired_at = float("-inf")

    await orch._apply_floor_action(FloorAction.CLOSE_WINDOW)
    await orch._maybe_self_initiate(time.monotonic())

    assert fake.windows_closed == 1
    assert orch._awaiting_reply is True
    assert fake.texts_sent == []
