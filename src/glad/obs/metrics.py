"""Minimal in-process metrics: fixed-size rolling samples with percentiles.

No external deps (no prometheus_client, no statsd). Good enough to answer
"is the buffer depth stable" and "are we underrunning" from a REPL or a log
line, which is all this slice needs.
"""

from __future__ import annotations

from collections import defaultdict, deque

BUFFER_DEPTH_MS = "glad_buffer_depth_ms"
PLAYBACK_UNDERRUNS = "glad_playback_underruns"
INBOUND_TO_GEMINI_MS = "glad_inbound_to_gemini_ms"
GEMINI_FIRST_BYTE_MS = "glad_gemini_first_byte_ms"
GEMINI_TO_OUTBOUND_MS = "glad_gemini_to_outbound_ms"
# Bytes Gemini has produced but the realtime pacer hasn't sent yet -- this is
# where "Gemini generates faster than speech plays" piles up now, instead of
# in the browser's ring buffer. Also the number that bounds interrupt latency.
SERVER_QUEUE_DEPTH_MS = "glad_server_queue_depth_ms"
# Tool call received -> function response sent back to the model. Bounds how
# long Gemini stalls waiting on `record_answer` (gemini-3.1-flash-live
# withholds turn_complete until the response is sent).
TOOL_ROUNDTRIP_MS = "glad_tool_roundtrip_ms"
# Same measurement as GEMINI_FIRST_BYTE_MS, split by whether the turn used
# Google Search grounding. Grounding never calls a tool (so it's invisible
# to TOOL_ROUNDTRIP_MS) but does add latency before the first audio byte --
# split at the source instead of after the aggregate is already polluted.
GEMINI_FIRST_BYTE_MS_GROUNDED = "glad_gemini_first_byte_ms_grounded"
GEMINI_FIRST_BYTE_MS_UNGROUNDED = "glad_gemini_first_byte_ms_ungrounded"

_MAX_SAMPLES = 1000
_samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_MAX_SAMPLES))


def record(name: str, value_ms: float) -> None:
    """Append one sample for `name`. Oldest samples drop once the deque is full."""
    _samples[name].append(value_ms)


def percentiles(name: str) -> dict[str, float]:
    """p50/p95/p99 of the samples currently held for `name`. Zeros if empty."""
    values = sorted(_samples[name])
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def _at(pct: float) -> float:
        index = min(int(pct * len(values)), len(values) - 1)
        return values[index]

    return {"p50": _at(0.50), "p95": _at(0.95), "p99": _at(0.99)}
