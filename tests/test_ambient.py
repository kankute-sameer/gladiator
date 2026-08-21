"""Tests for glad.agent.ambient: the buffer that carries volunteered
ambient speech to the next turn Glad takes."""

from __future__ import annotations

from glad.agent.ambient import AmbientBuffer


def test_empty_buffer_flushes_to_none() -> None:
    buffer = AmbientBuffer()
    assert buffer.flush() is None


def test_flush_returns_joined_text_and_clears() -> None:
    buffer = AmbientBuffer()
    buffer.add("Alice", "our budget is around fifty thousand", ts=1.0)
    buffer.add("Bob", "yeah and we need it by next quarter", ts=2.0)
    assert len(buffer) == 2

    flushed = buffer.flush()
    assert flushed is not None
    assert "Alice: our budget is around fifty thousand" in flushed
    assert "Bob: yeah and we need it by next quarter" in flushed

    assert len(buffer) == 0
    assert buffer.flush() is None


def test_blank_text_is_not_buffered() -> None:
    buffer = AmbientBuffer()
    buffer.add("Alice", "   ", ts=1.0)
    assert len(buffer) == 0
    assert buffer.flush() is None
