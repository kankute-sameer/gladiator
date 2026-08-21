"""Read-only access to past runs for the monitor dashboard.

Parses `runs/*.jsonl` -- the same files `glad.obs.events` writes -- and
never touches the live event bus itself (that's `events.on`, consumed by
`glad.transport.monitor` for the live side of the dashboard).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"


def _parse_lines(path: Path) -> list[dict[str, Any]]:
    """Parse one JSONL run file, tolerating:
      - a truncated final line (a killed process is the normal case here,
        not an error -- every line up to that point is still good data)
      - a concurrently in-progress file (read whatever is flushed to disk
        right now; a partial trailing write just gets skipped)
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _summarize(run_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    session_id = None
    question_set_id = None
    for event in events:
        if event.get("t") == "run_meta":
            session_id = event.get("session_id")
            question_set_id = event.get("question_set_id")
            break
    return {
        "id": run_id,
        "started_at": events[0].get("ts") if events else None,
        "ended_at": events[-1].get("ts") if events else None,
        # No `run_stop` means the process never reached a clean shutdown
        # (killed, crashed, or -- for the most recent run -- still live).
        "completed": any(event.get("t") == "run_stop" for event in events),
        "session_id": session_id,
        "question_set_id": question_set_id,
        "event_count": len(events),
    }


def list_runs() -> list[dict[str, Any]]:
    """Metadata for every run under `runs/`, most recent first. Empty (not
    an error) files, and directories with no runs yet, both come back as
    an empty list."""
    if not _RUNS_DIR.exists():
        return []

    runs = []
    for path in _RUNS_DIR.glob("*.jsonl"):
        events = _parse_lines(path)
        if not events:
            continue
        runs.append(_summarize(path.stem, events))
    # Sort on the run's own first-event timestamp, not the filename --
    # labels vary ("slice3-...", "wakeword_report-...", "...-reconnect-
    # test-..."), so lexicographic filename order is not chronological.
    runs.sort(key=lambda r: r["started_at"] or 0.0, reverse=True)
    return runs


def load_run(run_id: str) -> dict[str, Any]:
    """Full metadata plus every event for one run, chronological.

    Raises FileNotFoundError for an unknown id -- `main.py` turns that
    into a 404 rather than a 500.
    """
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise FileNotFoundError(run_id)
    path = _RUNS_DIR / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(run_id)

    events = _parse_lines(path)
    # File order is already emit order (single writer under a lock in
    # `events.emit`), but sort defensively on the monotonic clock -- the
    # one timestamp guaranteed not to jump around within a single run --
    # rather than trust that no future writer ever appends out of order.
    events.sort(key=lambda e: e.get("mono", e.get("ts", 0.0)))

    summary = _summarize(run_id, events)
    summary["events"] = events
    return summary
