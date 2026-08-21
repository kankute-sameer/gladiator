"""SpeakerTracker: debounce RMS, emit only on change."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from glad.agent.script import Question, QuestionSet
from glad.agent.speaker import SpeakerTracker
from glad.audio.activity import ParticipantActivity
from glad.live.session import LiveSession
from glad.orchestrator import Orchestrator, _SPEECH_RMS_THRESHOLD

_LOUD = b"\x00\x40\x00\xc0" * 200


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _tracker(threshold: float = 0.02, debounce_s: float = 0.3) -> tuple[SpeakerTracker, FakeClock]:
    clock = FakeClock()
    return SpeakerTracker(threshold=threshold, debounce_s=debounce_s, clock=clock), clock


def test_quiet_levels_do_not_declare_a_speaker() -> None:
    tracker, _ = _tracker()
    assert tracker.update({1: 0.001}, 0.0) is None
    assert tracker.current_speaker() is None


def test_leader_must_hold_debounce_before_being_declared() -> None:
    tracker, clock = _tracker()
    assert tracker.update({1: 0.5}, 0.0) is None
    clock.t = 0.29
    assert tracker.update({1: 0.5}, 0.29) is None
    assert tracker.current_speaker() is None
    clock.t = 0.30
    assert tracker.update({1: 0.5}, 0.30) == 1
    assert tracker.current_speaker() == 1


def test_same_speaker_does_not_re_emit() -> None:
    tracker, _ = _tracker()
    tracker.update({1: 0.5}, 0.0)
    assert tracker.update({1: 0.5}, 0.30) == 1
    assert tracker.update({1: 0.6}, 0.50) is None


def test_overlapping_speech_does_not_flap_without_holding_debounce() -> None:
    tracker, _ = _tracker()
    tracker.update({1: 0.5, 2: 0.1}, 0.0)
    tracker.update({1: 0.1, 2: 0.5}, 0.1)
    tracker.update({1: 0.5, 2: 0.1}, 0.2)
    assert tracker.update({1: 0.1, 2: 0.5}, 0.3) is None
    assert tracker.current_speaker() is None


def test_new_speaker_must_hold_debounce_after_a_change() -> None:
    tracker, _ = _tracker()
    tracker.update({1: 0.5}, 0.0)
    assert tracker.update({1: 0.5}, 0.30) == 1
    tracker.update({2: 0.8, 1: 0.1}, 0.40)
    assert tracker.update({2: 0.8, 1: 0.1}, 0.69) is None
    assert tracker.update({2: 0.8, 1: 0.1}, 0.71) == 2
    assert tracker.current_speaker() == 2


def test_silence_does_not_clear_the_declared_speaker() -> None:
    tracker, _ = _tracker()
    tracker.update({1: 0.5}, 0.0)
    tracker.update({1: 0.5}, 0.30)
    assert tracker.update({}, 1.0) is None
    assert tracker.current_speaker() == 1


def test_activity_levels_are_stored_rms_not_recomputed() -> None:
    activity = ParticipantActivity(stale_after_s=0.25)
    activity.observe(7, _LOUD, 1.0)
    levels = activity.levels(1.0)
    assert 7 in levels
    assert levels[7] >= _SPEECH_RMS_THRESHOLD


def _orchestrator_with_live() -> tuple[Orchestrator, MagicMock]:
    questions = QuestionSet(id="t", version=1, questions=(Question(id="q", text="Q?"),))
    orch = Orchestrator(questions, session_id="spk")
    live = MagicMock(spec=LiveSession)
    live.send_context = AsyncMock()
    live.activity_open = True
    orch.bind_live(live)
    return orch, live


@pytest.mark.asyncio
async def test_orchestrator_sends_speaker_context_on_change() -> None:
    orch, live = _orchestrator_with_live()
    orch.state.roster.note(1, "Alice")
    orch._activity.observe(1, _LOUD, 0.0)
    await orch._track_speaker(0.0)
    live.send_context.assert_not_awaited()
    orch._activity.observe(1, _LOUD, 0.30)
    await orch._track_speaker(0.30)
    live.send_context.assert_awaited_once_with("[Alice is speaking]")


@pytest.mark.asyncio
async def test_orchestrator_rate_limits_speaker_context_to_once_per_second() -> None:
    orch, live = _orchestrator_with_live()
    orch.state.roster.note(1, "Alice")
    orch.state.roster.note(2, "Bob")
    orch._activity.observe(1, _LOUD, 0.0)
    await orch._track_speaker(0.0)
    orch._activity.observe(1, _LOUD, 0.30)
    await orch._track_speaker(0.30)
    live.send_context.reset_mock()
    orch._activity.observe(2, _LOUD, 0.40)
    await orch._track_speaker(0.40)
    orch._activity.observe(2, _LOUD, 0.70)
    await orch._track_speaker(0.70)
    live.send_context.assert_not_awaited()
    orch._activity.observe(2, _LOUD, 1.31)
    await orch._track_speaker(1.31)
    live.send_context.assert_awaited_once_with("[Bob is speaking]")
