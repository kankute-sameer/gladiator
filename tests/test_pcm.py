"""Tests for glad.audio.pcm: rms, gate, and mix."""

from __future__ import annotations

import struct

import pytest

from glad.audio.pcm import gate, mix, rms


def _pcm(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def test_rms_of_silence_is_zero() -> None:
    assert rms(_pcm([0] * 100)) == 0.0


def test_rms_of_full_scale_square_wave_is_near_one() -> None:
    square_wave = _pcm([32767, -32768] * 50)
    assert 0.99 <= rms(square_wave) <= 1.0


def test_rms_of_odd_length_input_raises() -> None:
    with pytest.raises(ValueError):
        rms(b"\x00\x01\x02")


def test_silence_fails_gate_speech_passes() -> None:
    silence = _pcm([0] * 160)
    speech = _pcm([8000] * 160)
    assert gate(silence, 0.02) is False
    assert gate(speech, 0.02) is True


def test_mix_drops_gated_frames() -> None:
    silence = _pcm([0, 0, 0, 0])
    speech = _pcm([20000, 20000, 20000, 20000])
    assert mix([silence, speech], gate_threshold=0.02) == speech


def test_mix_ten_full_amplitude_inputs_clip_without_wrapping() -> None:
    """int32 sum of 10 full-scale frames must clip to int16 max, not wrap."""
    frames = [_pcm([32767, -32768])] * 10
    mixed = mix(frames, gate_threshold=0.0)
    assert mixed == _pcm([32767, -32768])


def test_mix_of_no_frames_is_empty() -> None:
    assert mix([]) == b""


def test_mix_of_one_frame_is_unchanged() -> None:
    frame = _pcm([100, -200, 300])
    assert mix([frame]) == frame


def test_mix_sums_not_averages() -> None:
    a = _pcm([10000, -10000])
    b = _pcm([10000, -10000])
    assert mix([a, b]) == _pcm([20000, -20000])


def test_mix_clips_to_int16_range() -> None:
    a = _pcm([30000, -30000])
    b = _pcm([30000, -30000])
    assert mix([a, b]) == _pcm([32767, -32768])


def test_mix_does_not_wrap_on_clip() -> None:
    # True sum is 98301: must clip to int16 max, not wrap to a negative value.
    frames = [_pcm([32767])] * 3
    assert mix(frames) == _pcm([32767])


def test_mix_zero_pads_unequal_lengths() -> None:
    long = _pcm([100, 200, 300])
    short = _pcm([1, 2])
    assert mix([long, short]) == _pcm([101, 202, 300])
