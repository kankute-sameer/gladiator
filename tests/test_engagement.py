"""EngagementState: derived from timestamps, clock injected, no sleeps."""

from __future__ import annotations

from glad.agent.engagement import EngagementState


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _state(ttl_s: float = 10.0, hard_cap_s: float = 120.0) -> tuple[EngagementState, FakeClock]:
    clock = FakeClock()
    return EngagementState(ttl_s=ttl_s, hard_cap_s=hard_cap_s, clock=clock), clock


def test_joins_dormant() -> None:
    engagement, _ = _state()
    assert engagement.is_engaged() is False


def test_extend_then_immediately_engaged() -> None:
    engagement, _ = _state()
    opened = engagement.extend("wake_word")
    assert opened is True
    assert engagement.is_engaged() is True
    assert engagement.engaged_by == "wake_word"


def test_extend_then_clock_past_ttl_not_engaged() -> None:
    engagement, clock = _state(ttl_s=10.0)
    engagement.extend("wake_word")
    clock.t = 9.99
    assert engagement.is_engaged() is True
    clock.t = 10.0
    assert engagement.is_engaged() is False


def test_repeated_extends_past_hard_cap_not_engaged() -> None:
    engagement, clock = _state(ttl_s=10.0, hard_cap_s=120.0)
    engagement.extend("self_initiated")
    for t in range(5, 125, 5):
        clock.t = float(t)
        if engagement.is_engaged():
            engagement.extend("stay_engaged")
    clock.t = 121.0
    assert engagement.is_engaged() is False


def test_dismiss_expires_immediately_regardless_of_ttl() -> None:
    engagement, clock = _state(ttl_s=10.0)
    engagement.extend("wake_word")
    clock.t = 1.0
    assert engagement.is_engaged() is True
    assert engagement.dismiss("dismissed") is True
    assert engagement.is_engaged() is False
    clock.t = 1.5
    assert engagement.is_engaged() is False


def test_hold_refreshes_ttl_without_changing_trigger() -> None:
    engagement, clock = _state(ttl_s=10.0)
    engagement.extend("wake_word")
    clock.t = 9.0
    engagement.hold()
    clock.t = 18.5
    assert engagement.is_engaged() is True
    assert engagement.engaged_by == "wake_word"
    clock.t = 19.0
    assert engagement.is_engaged() is False
