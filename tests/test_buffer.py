"""Tests for glad.audio.buffer.JitterBuffer."""

from __future__ import annotations

import time

from glad.audio.buffer import JitterBuffer

# 24kHz mono s16le: 48 bytes/ms, per the spec this buffer is sized for.
_BYTES_PER_MS = 48


def test_depth_ms_correct_for_known_byte_count() -> None:
    buf = JitterBuffer()
    buf.push(b"\x00\x00" * (100 * _BYTES_PER_MS // 2))  # exactly 100ms of silence

    # A little real time necessarily elapses between push() and depth_ms();
    # allow a small margin instead of asserting an exact 100.0.
    depth = buf.depth_ms()
    assert 95.0 <= depth <= 100.0


def test_reset_zeroes_depth() -> None:
    buf = JitterBuffer()
    buf.push(b"\x00\x00" * (500 * _BYTES_PER_MS // 2))
    assert buf.depth_ms() > 0.0

    buf.reset()
    assert buf.depth_ms() == 0.0


def test_depth_clamps_at_zero_when_drained_faster_than_filled() -> None:
    buf = JitterBuffer()
    buf.push(b"\x00\x00" * 10)  # ~0.4ms of audio: far less than the sleep below
    time.sleep(0.05)  # 50ms of real time elapses, well past what was pushed

    assert buf.depth_ms() == 0.0


def test_depth_ms_zero_before_any_push() -> None:
    buf = JitterBuffer()
    assert buf.depth_ms() == 0.0
