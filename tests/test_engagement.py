"""EngagementState: wake opens, go_dormant closes. No timer."""

from __future__ import annotations

from glad.conversation.turn import EngagementState


def test_joins_dormant() -> None:
    engagement = EngagementState()
    assert engagement.is_engaged() is False


def test_extend_then_engaged() -> None:
    engagement = EngagementState()
    opened = engagement.extend("wake_word")
    assert opened is True
    assert engagement.is_engaged() is True
    assert engagement.engaged_by == "wake_word"
    assert engagement.extend("follow_up") is False
    assert engagement.engaged_by == "wake_word"


def test_stays_engaged_until_dismissed() -> None:
    engagement = EngagementState()
    engagement.extend("wake_word")
    assert engagement.is_engaged() is True
    assert engagement.dismiss("not_for_me") is True
    assert engagement.is_engaged() is False
    assert engagement.last_close_reason == "not_for_me"
    assert engagement.dismiss("dismissed") is False
