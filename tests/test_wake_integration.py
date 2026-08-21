"""Wake word + floor control through Orchestrator, with a fake LiveSession."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from glad.conversation.session import Question, QuestionSet
from glad.conversation.turn import FloorAction, FloorState
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

_LOUD_PCM = (b"\x00\x40\x00\xc0" * 200)
_SILENT_PCM = b"\x00\x00" * 400


@dataclass
class FakeLiveSession:
    audio_sent: list[bytes] = field(default_factory=list)
    windows_opened: int = 0
    windows_closed: int = 0
    texts_sent: list[str] = field(default_factory=list)
    activity_open: bool = False

    async def send_audio(self, pcm: bytes) -> None:
        self.audio_sent.append(pcm)

    async def open_window(self, *, interrupt: bool = False) -> bool:
        self.windows_opened += 1
        self.activity_open = True
        return True

    async def close_window(self) -> bool:
        self.windows_closed += 1
        self.activity_open = False
        return True

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
async def test_dormant_inbound_audio_is_not_sent_to_gemini() -> None:
    orch, fake = _orchestrator()
    frame = AudioFrame(participant_id=7, participant_name="Sameer Kankute", pcm=_LOUD_PCM, ts=1.0)
    await orch.on_inbound_frame(frame)

    present = orch.state.roster.present()
    assert len(present) == 1
    assert present[0].name == "Sameer Kankute"
    assert fake.audio_sent == []
    assert fake.texts_sent == []
    assert fake.windows_opened == 0


@pytest.mark.asyncio
async def test_engaged_inbound_audio_is_sent_to_gemini() -> None:
    orch, fake = _orchestrator()
    orch.engagement.extend("wake_word", time.monotonic())
    frame = AudioFrame(participant_id=7, participant_name="Sameer Kankute", pcm=_LOUD_PCM, ts=1.0)
    await orch.on_inbound_frame(frame)
    assert fake.audio_sent
    assert fake.windows_opened == 1


@pytest.mark.asyncio
async def test_wake_word_while_participant_keeps_talking_stays_wake_pending() -> None:
    orch, fake = _orchestrator()

    now = 0.0
    await orch.on_transcript_segment(
        TranscriptSegment(participant_id=1, participant_name="Alice", text="Glad, what do you think?", ts=now)
    )
    assert orch._floor.state is FloorState.WAKE_PENDING
    assert orch.engagement.is_engaged() is False
    assert fake.texts_sent == []

    for _ in range(16):
        now += 0.1
        await _speech_frame(orch, now)
        assert orch._floor.state is FloorState.WAKE_PENDING
        now += 0.4
        await _silence_frame(orch, now)
        assert orch._floor.state is FloorState.WAKE_PENDING

    assert fake.texts_sent == []
    assert fake.windows_opened == 0

    now += orch._floor.gap_threshold_s + 0.2
    await _silence_frame(orch, now)

    assert orch._floor.state is FloorState.SPEAKING
    assert orch.engagement.is_engaged() is True
    assert len(fake.texts_sent) == 1
    assert "what do you think" in fake.texts_sent[0].lower()
    assert fake.windows_opened == 0


@pytest.mark.asyncio
async def test_wake_word_fires_text_turn_after_silence_gap() -> None:
    orch, fake = _orchestrator()

    now = 0.0
    await orch.on_transcript_segment(
        TranscriptSegment(participant_id=2, participant_name="Bob", text="Hey Glad, can you help with this?", ts=now)
    )
    assert orch._floor.state is FloorState.WAKE_PENDING
    assert orch.engagement.is_engaged() is False

    now += orch._floor.gap_threshold_s + 0.2
    await _silence_frame(orch, now)

    assert orch._floor.state is FloorState.SPEAKING
    assert orch.engagement.is_engaged() is True
    assert len(fake.texts_sent) == 1
    assert "can you help" in fake.texts_sent[0].lower()


@pytest.mark.asyncio
async def test_adjectival_glad_never_reaches_wake_pending() -> None:
    orch, fake = _orchestrator()
    await orch.on_transcript_segment(
        TranscriptSegment(participant_id=1, participant_name="Alice", text="so glad you called", ts=0.0)
    )
    assert orch._floor.state is FloorState.AMBIENT
    assert orch._wakeword_suppressed_total == 1
    assert orch._wakeword_wakes_total == 0
    assert len(orch._ambient) == 1


@pytest.mark.asyncio
async def test_dormant_speech_does_not_open_a_window() -> None:
    orch, fake = _orchestrator()
    now = 0.0
    await _speech_frame(orch, now)
    assert orch._floor.state is FloorState.AMBIENT
    assert fake.windows_opened == 0

    now += orch._floor.endpoint_gap_s + 1.0
    await _silence_frame(orch, now)
    assert fake.windows_closed == 0


@pytest.mark.asyncio
async def test_speech_while_engaged_opens_window_without_wake_word() -> None:
    orch, fake = _orchestrator()
    now = time.monotonic()
    orch.engagement.extend("wake_word", now)
    await _speech_frame(orch, now)

    assert fake.windows_opened == 1
    assert orch._floor.state is FloorState.SPEAKING
    assert orch.engagement.is_engaged(now) is True


@pytest.mark.asyncio
async def test_stays_engaged_until_go_dormant() -> None:
    orch, fake = _orchestrator()
    orch.engagement.extend("wake_word", 100.0)
    orch._speech_active = True
    orch._floor.state = FloorState.SPEAKING

    assert orch.engagement.is_engaged(115.0) is True
    assert fake.texts_sent == []


@pytest.mark.asyncio
async def test_close_window_does_not_send_text() -> None:
    orch, fake = _orchestrator()
    await orch._apply_floor_action(FloorAction.CLOSE_WINDOW)
    assert fake.windows_closed == 1
    assert orch._awaiting_reply is True
    assert fake.texts_sent == []


@pytest.mark.asyncio
async def test_echo_after_glad_finishes_does_not_open_a_window() -> None:
    orch, fake = _orchestrator()
    orch.engagement.extend("wake_word", 10.0)
    orch._echo_holdoff_until = 10.6

    await _speech_frame(orch, 10.1)
    assert fake.windows_opened == 0
    assert orch._floor.state is FloorState.AMBIENT

    await _silence_frame(orch, 10.2)
    await _speech_frame(orch, 10.7)
    assert fake.windows_opened == 1
