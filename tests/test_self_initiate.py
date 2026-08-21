"""Tests for glad.agent.self_initiate.SelfInitiate."""

from __future__ import annotations

from glad.agent.self_initiate import SelfInitiate


def test_not_ready_until_gap_elapses_while_floor_free() -> None:
    si = SelfInitiate(gap_s=3.0, cooldown_s=20.0)
    assert si.ready(0.0, floor_busy=False) is False  # starts the floor-free clock
    assert si.ready(2.9, floor_busy=False) is False
    assert si.ready(3.1, floor_busy=False) is True


def test_floor_busy_resets_the_floor_free_clock() -> None:
    si = SelfInitiate(gap_s=3.0, cooldown_s=20.0)
    si.ready(0.0, floor_busy=False)
    si.ready(2.0, floor_busy=True)  # someone started talking -> reset
    assert si.ready(2.1, floor_busy=False) is False  # clock restarted here
    assert si.ready(5.2, floor_busy=False) is True


def test_cooldown_blocks_refiring_immediately() -> None:
    si = SelfInitiate(gap_s=1.0, cooldown_s=20.0)
    si.ready(0.0, floor_busy=False)
    assert si.ready(1.1, floor_busy=False) is True
    si.mark_fired(1.1)

    si.ready(1.2, floor_busy=False)
    assert si.ready(2.3, floor_busy=False) is False  # cooldown still running

    assert si.ready(21.2, floor_busy=False) is True  # cooldown elapsed


def test_enabled_defaults_true_and_is_a_plain_flag() -> None:
    si = SelfInitiate(gap_s=1.0, cooldown_s=1.0)
    assert si.enabled is True
    si.enabled = False
    assert si.enabled is False
