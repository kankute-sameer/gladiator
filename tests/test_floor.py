"""Tests for glad.agent.floor: the 4-state floor control machine.

Detection (wakeword) is entirely out of scope here -- these tests drive
`FloorControl` directly with `on_wake_matched` standing in for "stage 2
already accepted this".
"""

from __future__ import annotations

from glad.agent.floor import FloorAction, FloorControl, FloorState


def test_discovery_speech_opens_window_directly_no_wake_needed() -> None:
    floor = FloorControl()
    action = floor.on_speech_started(0.0, discovery_mode=True)
    assert action is FloorAction.OPEN_WINDOW
    assert floor.state is FloorState.SPEAKING


def test_ambient_speech_without_wake_word_does_nothing() -> None:
    floor = FloorControl()
    action = floor.on_speech_started(0.0, discovery_mode=False)
    assert action is FloorAction.NONE
    assert floor.state is FloorState.AMBIENT


def test_wake_word_moves_ambient_to_wake_pending() -> None:
    floor = FloorControl()
    reached = floor.on_wake_matched(0.0)
    assert reached
    assert floor.state is FloorState.WAKE_PENDING


def test_wake_word_while_already_pending_is_not_a_new_transition() -> None:
    floor = FloorControl()
    assert floor.on_wake_matched(0.0)
    assert not floor.on_wake_matched(0.5)
    assert floor.state is FloorState.WAKE_PENDING


def test_wake_pending_stays_pending_while_participant_keeps_talking() -> None:
    """'Glad, what do you think' followed by 8 more seconds of talking must
    not fire early -- every tick before the gap elapses is a no-op, and
    continued speech keeps resetting the silence clock."""
    floor = FloorControl(gap_threshold_s=1.2)
    assert floor.on_wake_matched(0.0)

    # Wake word utterance ends, but the same speaker keeps going.
    floor.on_speech_ended(0.1)
    assert floor.tick(0.5) is FloorAction.NONE
    assert floor.state is FloorState.WAKE_PENDING

    # 8 more seconds of on/off talking, silence never sustained past the gap.
    t = 0.5
    for _ in range(16):
        floor.on_speech_started(t, discovery_mode=False)
        assert floor.state is FloorState.WAKE_PENDING  # never interrupted
        t += 0.3
        floor.on_speech_ended(t)
        assert floor.tick(t + 0.4) is FloorAction.NONE  # gap not reached yet
        t += 0.4

    assert floor.state is FloorState.WAKE_PENDING


def test_wake_pending_fires_once_gap_elapses() -> None:
    floor = FloorControl(gap_threshold_s=1.2)
    assert floor.on_wake_matched(0.0)
    floor.on_speech_ended(1.0)

    assert floor.tick(1.5) is FloorAction.NONE  # only 0.5s of silence so far
    assert floor.state is FloorState.WAKE_PENDING

    action = floor.tick(2.3)  # 1.3s of silence -- past the 1.2s gap
    assert action is FloorAction.WAKE
    assert floor.state is FloorState.SPEAKING


def test_discovery_window_closes_after_endpoint_gap() -> None:
    floor = FloorControl(endpoint_gap_s=0.5)
    floor.on_speech_started(0.0, discovery_mode=True)
    floor.on_speech_ended(1.0)

    assert floor.tick(1.2) is FloorAction.NONE  # only 0.2s silence
    action = floor.tick(1.6)  # 0.6s silence -- past the 0.5s endpoint gap
    assert action is FloorAction.CLOSE_WINDOW
    assert floor.state is FloorState.AMBIENT


def test_wake_word_during_discovery_mode_still_reaches_wake_pending() -> None:
    """A wake word during discovery mode must still work -- discovery mode
    is a flag on top of the state machine, not a bypass of it."""
    floor = FloorControl()
    # Not currently speaking (floor idle even though mode is discovery).
    reached = floor.on_wake_matched(0.0)
    assert reached
    assert floor.state is FloorState.WAKE_PENDING

    floor.on_speech_ended(0.1)
    action = floor.tick(0.1 + floor.gap_threshold_s + 0.1)
    assert action is FloorAction.WAKE
    assert floor.state is FloorState.SPEAKING


def test_interrupted_yields_and_returns_to_ambient() -> None:
    floor = FloorControl()
    floor.on_speech_started(0.0, discovery_mode=True)
    assert floor.state is FloorState.SPEAKING

    floor.on_interrupted()
    assert floor.state is FloorState.AMBIENT


def test_interrupted_keeps_speaking_while_listen_window_is_open() -> None:
    floor = FloorControl()
    floor.on_speech_started(0.0, discovery_mode=True)
    floor.on_interrupted(listening=True)
    assert floor.state is FloorState.SPEAKING
    floor.on_speech_ended(1.0)
    assert floor.tick(1.0 + floor.endpoint_gap_s + 0.1) is FloorAction.CLOSE_WINDOW


def test_turn_complete_returns_ambient_wake_turn_with_no_window_to_close() -> None:
    """An ambient WAKE turn is text-only -- there is no live window for
    `tick()` to ever close, so `turn_complete` must be what returns the
    floor to AMBIENT."""
    floor = FloorControl(gap_threshold_s=0.1)
    floor.on_wake_matched(0.0)
    floor.on_speech_ended(0.0)
    assert floor.tick(0.2) is FloorAction.WAKE
    assert floor.state is FloorState.SPEAKING

    floor.on_turn_complete()
    assert floor.state is FloorState.AMBIENT


def test_turn_complete_does_not_drop_an_open_listen_window() -> None:
    floor = FloorControl()
    floor.on_speech_started(0.0, discovery_mode=True)
    floor.on_turn_complete(listening=True)
    assert floor.state is FloorState.SPEAKING


def test_speech_started_while_speaking_does_not_reopen_window() -> None:
    floor = FloorControl()
    floor.on_speech_started(0.0, discovery_mode=True)
    assert floor.state is FloorState.SPEAKING

    action = floor.on_speech_started(0.2, discovery_mode=True)
    assert action is FloorAction.NONE
    assert floor.state is FloorState.SPEAKING


def test_barge_in_takes_floor_even_when_already_speaking() -> None:
    """Talking over a wake/text turn: floor is already SPEAKING, so
    on_speech_started is a no-op. Barge-in must still return OPEN_WINDOW
    so Gemini gets ActivityStart and stops generating."""
    floor = FloorControl()
    floor.on_wake_matched(0.0)
    floor.on_speech_ended(0.0)
    assert floor.tick(floor.gap_threshold_s + 0.1) is FloorAction.WAKE
    assert floor.state is FloorState.SPEAKING

    action = floor.on_barge_in(2.0)
    assert action is FloorAction.OPEN_WINDOW
    assert floor.state is FloorState.SPEAKING
