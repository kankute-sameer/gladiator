"""Tests for glad.audio.activity.ParticipantActivity."""

from __future__ import annotations

from glad.audio.activity import ParticipantActivity

_LOUD_PCM = b"\x00\x40\x00\xc0" * 200  # well above any reasonable RMS threshold
_SILENT_PCM = b"\x00\x00" * 400


def test_observe_returns_this_frames_rms() -> None:
    activity = ParticipantActivity(stale_after_s=1.0)
    level = activity.observe(1, _LOUD_PCM, now=0.0)
    assert level > 0.0
    assert activity.observe(2, _SILENT_PCM, now=0.0) == 0.0


def test_mix_combines_all_live_participants() -> None:
    activity = ParticipantActivity(stale_after_s=1.0)
    activity.observe(1, _LOUD_PCM, now=0.0)
    activity.observe(2, _LOUD_PCM, now=0.0)
    mixed = activity.mix(gate_threshold=None)
    assert len(mixed) == len(_LOUD_PCM)


def test_stale_participant_is_evicted_from_the_mix() -> None:
    activity = ParticipantActivity(stale_after_s=0.25)
    activity.observe(1, _LOUD_PCM, now=0.0)
    activity.observe(2, _LOUD_PCM, now=1.0)  # far apart -> evicts participant 1

    mixed = activity.mix(gate_threshold=None)
    assert mixed == _LOUD_PCM


def test_anyone_active_true_only_while_loud_and_fresh() -> None:
    activity = ParticipantActivity(stale_after_s=0.25)
    activity.observe(1, _LOUD_PCM, now=0.0)

    assert activity.anyone_active(threshold=0.02, now=0.1) is True
    assert activity.anyone_active(threshold=0.02, now=1.0) is False  # gone stale


def test_anyone_active_false_when_below_threshold() -> None:
    activity = ParticipantActivity(stale_after_s=1.0)
    activity.observe(1, _SILENT_PCM, now=0.0)
    assert activity.anyone_active(threshold=0.02, now=0.0) is False


def test_gate_threshold_drops_quiet_frames_from_the_mix() -> None:
    activity = ParticipantActivity(stale_after_s=1.0)
    activity.observe(1, _SILENT_PCM, now=0.0)
    activity.observe(2, _LOUD_PCM, now=0.0)

    mixed = activity.mix(gate_threshold=0.02)
    assert mixed == _LOUD_PCM
