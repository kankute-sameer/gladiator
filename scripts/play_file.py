"""Pace a 24kHz mono s16le WAV file into /ws/meeting for playback testing.

This proves the streaming playback path (worklet ring buffer, prebuffer,
flush) in isolation, with a known-good file, before any model produces
audio. Any clicks or gaps heard while running this script are the buffer's
fault, not a model's.

Usage:
    python -m scripts.play_file assets/test_speech.wav
    python -m scripts.play_file assets/test_speech.wav --jitter 50
    python -m scripts.play_file assets/test_speech.wav --flush-after 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import wave
from pathlib import Path

import websockets

from glad.config import settings
from glad.logging import configure_logging, get_logger

configure_logging(settings.log_level)
logger = get_logger(__name__)

_EXPECTED_SAMPLE_RATE = 24_000
_EXPECTED_SAMPLE_WIDTH = 2  # bytes per sample (s16le)
_EXPECTED_CHANNELS = 1


def _read_pcm(path: Path) -> bytes:
    """Read raw PCM frames from `path`, or exit with a clear error if the
    file isn't 24kHz mono s16le."""
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getframerate() != _EXPECTED_SAMPLE_RATE:
            raise SystemExit(
                f"{path} is {wav_file.getframerate()}Hz, expected {_EXPECTED_SAMPLE_RATE}Hz"
            )
        if wav_file.getsampwidth() != _EXPECTED_SAMPLE_WIDTH:
            raise SystemExit(
                f"{path} is {wav_file.getsampwidth() * 8}-bit, expected 16-bit"
            )
        if wav_file.getnchannels() != _EXPECTED_CHANNELS:
            raise SystemExit(
                f"{path} has {wav_file.getnchannels()} channel(s), expected mono"
            )
        return wav_file.readframes(wav_file.getnframes())


def _chunks(pcm: bytes, chunk_bytes: int) -> list[bytes]:
    return [pcm[offset : offset + chunk_bytes] for offset in range(0, len(pcm), chunk_bytes)]


async def play_file(
    path: Path,
    url: str,
    chunk_ms: int,
    jitter_ms: int,
    flush_after_s: float | None,
    duration_s: float | None,
) -> None:
    pcm = _read_pcm(path)
    chunk_bytes = int(_EXPECTED_SAMPLE_RATE * _EXPECTED_SAMPLE_WIDTH * chunk_ms / 1000)
    chunk_seconds = chunk_ms / 1000.0
    file_seconds = len(pcm) / (_EXPECTED_SAMPLE_RATE * _EXPECTED_SAMPLE_WIDTH)
    chunks = _chunks(pcm, chunk_bytes)

    logger.info(
        "Connecting to %s (file=%.2fs, duration=%s, %dms chunks, jitter=+/-%dms)",
        url,
        file_seconds,
        f"{duration_s:.2f}s" if duration_s is not None else "one pass",
        chunk_ms,
        jitter_ms,
    )

    async with websockets.connect(url) as ws:
        elapsed_s = 0.0
        index = 0
        # Absolute-clock pacing: accumulate sleep error makes us run slow,
        # which drains the client buffer and looks like underruns. Target
        # wall-clock deadlines from t0 instead.
        t0 = time.monotonic()

        while True:
            if duration_s is not None and elapsed_s >= duration_s:
                break
            if flush_after_s is not None and elapsed_s >= flush_after_s:
                # Barge-in semantics: cut the model off and do not resume.
                await ws.send(json.dumps({"t": "flush"}))
                logger.info("Sent flush at t=%.2fs; stopping (no refill)", elapsed_s)
                break
            if duration_s is None and index >= len(chunks):
                break

            chunk = chunks[index % len(chunks)]
            index += 1
            await ws.send(chunk)

            elapsed_s += chunk_seconds
            target = t0 + elapsed_s
            if jitter_ms:
                target += random.uniform(-jitter_ms, jitter_ms) / 1000.0
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    logger.info("Finished sending %s (%.2fs of audio)", path, elapsed_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", type=Path, help="24kHz mono s16le WAV file to play")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=settings.port, help="Server port")
    parser.add_argument(
        "--chunk-ms", type=int, default=20, help="Chunk size in milliseconds (default: 20)"
    )
    parser.add_argument(
        "--jitter", type=int, default=0, help="Randomly vary pacing by +/- N ms (default: 0)"
    )
    parser.add_argument(
        "--flush-after",
        type=float,
        default=None,
        help="Send flush at N seconds, then stop sending (barge-in; no refill)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Loop the file and keep sending for N seconds (for depth stability runs)",
    )
    args = parser.parse_args()

    url = f"ws://{args.host}:{args.port}/ws/meeting"
    asyncio.run(
        play_file(
            args.wav_path,
            url,
            args.chunk_ms,
            args.jitter,
            args.flush_after,
            args.duration,
        )
    )


if __name__ == "__main__":
    main()
