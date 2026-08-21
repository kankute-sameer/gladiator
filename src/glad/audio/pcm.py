"""Pure PCM math: level, noise gate, and mix.

No I/O, no per-participant state — those live in `activity.py` / `record.py`.
"""

from __future__ import annotations

import array
import math
import sys
from itertools import zip_longest

_FULL_SCALE = 32768.0
_INT16_MIN = -32768
_INT16_MAX = 32767


def rms(pcm: bytes) -> float:
    """Root-mean-square level of signed 16-bit little-endian PCM, normalized to [0, 1].

    Raises ValueError on odd-length input rather than silently truncating a
    trailing partial sample.
    """
    if len(pcm) % 2 != 0:
        raise ValueError(f"pcm length must be a multiple of 2 (16-bit samples), got {len(pcm)}")
    if not pcm:
        return 0.0

    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()

    sum_squares = sum(sample * sample for sample in samples)
    mean_square = sum_squares / len(samples)
    return min(math.sqrt(mean_square) / _FULL_SCALE, 1.0)


def gate(pcm: bytes, threshold: float) -> bool:
    """True if this s16le frame's RMS is at or above `threshold`."""
    return rms(pcm) >= threshold


def mix(frames: list[bytes], gate_threshold: float | None = None) -> bytes:
    """Sum s16le PCM frames as int32 and clip once to int16 range.

    Never averages -- averaging quietens every speaker as headcount rises.
    Unequal-length frames are zero-padded to the longest. Frames that fail
    the noise gate are dropped before the sum so N channels of keyboard
    noise cannot clip the mix.
    """
    if gate_threshold is not None:
        frames = [frame for frame in frames if gate(frame, gate_threshold)]
    if not frames:
        return b""

    channels: list[array.array] = []
    for frame in frames:
        trimmed = frame[: len(frame) - (len(frame) % 2)]
        channel = array.array("h")
        channel.frombytes(trimmed)
        channels.append(channel)

    mixed = array.array("h")
    for samples in zip_longest(*channels, fillvalue=0):
        total = sum(samples)  # full-precision int sum, clip only at the end
        mixed.append(max(_INT16_MIN, min(_INT16_MAX, total)))
    return mixed.tobytes()
