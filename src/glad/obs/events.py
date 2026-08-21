"""Append-only event bus that persists every event to a JSONL run file.

One process, one active run file. Start a run with `start_run(label=...)`;
every subsequent `emit(...)` is written as a single JSON line. Callers that
only care about live observation can subscribe via `on(...)`.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from glad.logging import get_logger

logger = get_logger(__name__)

_RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"
_lock = threading.Lock()
_run_path: Path | None = None
_subscribers: list[Callable[[dict[str, Any]], None]] = []


def start_run(label: str = "slice2a") -> Path:
    """Open a new JSONL file under `runs/` and make it the active sink."""
    global _run_path
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = _RUNS_DIR / f"{label}-{stamp}.jsonl"
    with _lock:
        _run_path = path
        path.write_text("")  # truncate / create
        (_RUNS_DIR / "CURRENT").write_text(str(path) + "\n", encoding="utf-8")
    # Reset outbound playback-window bookkeeping for the new run.
    try:
        from glad.transport import outbound as _outbound

        _outbound._saw_audio = False
        _outbound._jitter_buffer.reset()
    except Exception:  # noqa: BLE001
        pass
    logger.info("Event run file: %s", path)
    emit("run_start", label=label)
    try:
        from glad.audio.record import inbound_recorder
        from glad.config import settings

        inbound_recorder.configure(settings.record_inbound_audio)
        wav_path = inbound_recorder.start(path)
        if wav_path is not None:
            emit("inbound_recording_start", path=str(wav_path))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to start inbound audio recorder")
    return path


def stop_run() -> None:
    """Finalize side channels tied to the active run (inbound WAV, etc.)."""
    try:
        from glad.audio.record import inbound_recorder

        path = inbound_recorder.path
        inbound_recorder.close()
        if path is not None:
            emit("inbound_recording_stop", path=str(path))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to stop inbound audio recorder")


def current_run() -> Path | None:
    with _lock:
        if _run_path is not None:
            return _run_path
    pointer = _RUNS_DIR / "CURRENT"
    if pointer.exists():
        text = pointer.read_text(encoding="utf-8").strip()
        return Path(text) if text else None
    return None


def on(callback: Callable[[dict[str, Any]], None]) -> None:
    """Subscribe to live events (in addition to the JSONL write)."""
    _subscribers.append(callback)


def emit(kind: str, **fields: Any) -> None:
    """Record one event. Safe to call before `start_run` (logged, not written)."""
    event: dict[str, Any] = {
        "ts": time.time(),
        "mono": time.monotonic(),
        "t": kind,
        **fields,
    }
    with _lock:
        path = _run_path
        if path is not None:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")
    for callback in list(_subscribers):
        try:
            callback(event)
        except Exception:  # noqa: BLE001 — subscribers must not break emit
            logger.exception("Event subscriber failed for %s", kind)
