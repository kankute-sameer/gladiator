"""Optional debug recorder: persist the PCM Glad sends to Gemini as a WAV.

Records the *mixed* inbound stream (same bytes as `LiveSession.send_audio`),
16 kHz mono s16le — so you can listen to exactly what the model was given
and compare it to `Heard:` transcripts.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path

from glad.logging import get_logger

logger = get_logger(__name__)

_SAMPLE_RATE = 16_000
_SAMPLE_WIDTH = 2
_CHANNELS = 1


class InboundRecorder:
    """Append-only WAV writer for one run. Thread-safe; no-ops when disabled."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wav: wave.Wave_write | None = None
        self._path: Path | None = None
        self._bytes_written = 0
        self._enabled = False

    @property
    def path(self) -> Path | None:
        return self._path

    def configure(self, enabled: bool) -> None:
        self._enabled = enabled

    def start(self, run_jsonl: Path) -> Path | None:
        """Open `<run_stem>-inbound.wav` next to the JSONL run file."""
        self.close()
        if not self._enabled:
            return None
        path = run_jsonl.with_name(f"{run_jsonl.stem}-inbound.wav")
        wav = wave.open(str(path), "wb")
        wav.setnchannels(_CHANNELS)
        wav.setsampwidth(_SAMPLE_WIDTH)
        wav.setframerate(_SAMPLE_RATE)
        with self._lock:
            self._wav = wav
            self._path = path
            self._bytes_written = 0
        logger.info("Inbound audio recording: %s", path)
        return path

    def write(self, pcm: bytes) -> None:
        """Append one chunk of 16 kHz mono s16le PCM. No-op if not recording."""
        if not pcm:
            return
        with self._lock:
            wav = self._wav
            if wav is None:
                return
            wav.writeframes(pcm)
            self._bytes_written += len(pcm)

    def close(self) -> None:
        with self._lock:
            wav = self._wav
            path = self._path
            nbytes = self._bytes_written
            self._wav = None
            self._path = None
            self._bytes_written = 0
        if wav is None:
            return
        wav.close()
        duration_s = nbytes / (_SAMPLE_RATE * _SAMPLE_WIDTH * _CHANNELS)
        logger.info(
            "Inbound audio recording closed: %s (%.1fs, %d bytes)",
            path,
            duration_s,
            nbytes,
        )


inbound_recorder = InboundRecorder()
