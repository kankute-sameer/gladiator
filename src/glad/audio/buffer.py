"""Server-side jitter buffer depth tracking for outbound audio streaming.

This does not hold or replay audio itself -- outbound PCM is relayed to
clients immediately as it arrives. `JitterBuffer` only answers "how far
ahead of realtime is the stream I'm relaying", so pacing bugs (e.g. a
sender pushing faster than realtime) show up as depth that grows without
bound instead of staying flat.
"""

from __future__ import annotations

import time

# 24_000 Hz * 2 bytes/sample / 1000 ms/s
_BYTES_PER_MS_AT_24KHZ = 48.0


class JitterBuffer:
    """Tracks bytes pushed vs. wall-clock time elapsed to estimate buffered depth."""

    def __init__(self, bytes_per_ms: float = _BYTES_PER_MS_AT_24KHZ) -> None:
        self._bytes_per_ms = bytes_per_ms
        self._pushed_bytes = 0
        self._started_at: float | None = None

    def push(self, pcm: bytes) -> None:
        """Record `pcm` as having arrived now."""
        if self._started_at is None:
            self._started_at = time.monotonic()
        self._pushed_bytes += len(pcm)

    def depth_ms(self) -> float:
        """Milliseconds of audio pushed ahead of what real time has consumed.

        Zero before the first push, and clamped at zero if the pacer falls
        behind realtime (elapsed time exceeds the duration of audio pushed).
        """
        if self._started_at is None:
            return 0.0
        pushed_ms = self._pushed_bytes / self._bytes_per_ms
        elapsed_ms = (time.monotonic() - self._started_at) * 1000.0
        return max(pushed_ms - elapsed_ms, 0.0)

    def reset(self) -> None:
        """Drop all tracked state, as if no audio had ever been pushed."""
        self._pushed_bytes = 0
        self._started_at = None
