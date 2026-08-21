"""Download Recall's post-call per-participant audio for a bot.

After a Meet call ends, Recall finishes processing `audio_separate_raw`.
This pulls those files (closer to "what the bot heard in the room" than the
realtime websocket) into `runs/recall-<bot_id>/`.

Raw 16 kHz mono s16le parts are also wrapped as `.wav` for easy playback.

Usage:
    python -m scripts.fetch_recall_audio <bot-id>
    python -m scripts.fetch_recall_audio <bot-id> --wait 120

`send_bot` prints the bot id when it creates the bot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import wave
from pathlib import Path
from typing import Any

import httpx

from glad.config import settings
from glad.logging import configure_logging, get_logger
from glad.recall.client import RecallAPIError, RecallClient

configure_logging(settings.log_level)
logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
_RUNS = ROOT / "runs"
_RAW_RATE = 16_000


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()) or "participant"
    return cleaned[:80]


def _wrap_raw_as_wav(raw_path: Path) -> Path:
    """Write a sibling .wav so the raw PCM is playable without extra tools."""
    data = raw_path.read_bytes()
    if len(data) % 2:
        data = data[:-1]
    wav_path = raw_path.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_RAW_RATE)
        handle.writeframes(data)
    return wav_path


async def _wait_for_done(
    client: RecallClient,
    recording_id: str,
    *,
    timeout_s: float,
    poll_s: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = await client.list_audio_separate(recording_id)
        results = last.get("results") or []
        if results and all((r.get("status") or {}).get("code") == "done" for r in results):
            return last
        statuses = [(r.get("status") or {}).get("code") for r in results]
        logger.info(
            "Waiting for audio_separate (recording=%s, status=%s)",
            recording_id,
            statuses or ["<none yet>"],
        )
        await asyncio.sleep(poll_s)
    raise TimeoutError(
        f"audio_separate not done within {timeout_s:.0f}s for recording {recording_id}"
    )


async def fetch_bot_audio(bot_id: str, *, wait_s: float, out_dir: Path) -> list[Path]:
    client = RecallClient()
    bot = await client.get_bot(bot_id)
    recordings = bot.get("recordings") or []
    if not recordings:
        raise RuntimeError(
            f"Bot {bot_id} has no recordings yet — leave the call and wait a bit, then retry"
        )

    recording_id = recordings[0]["id"]
    logger.info("Bot %s recording %s", bot_id, recording_id)

    if wait_s > 0:
        listing = await _wait_for_done(client, recording_id, timeout_s=wait_s)
    else:
        listing = await client.list_audio_separate(recording_id)

    results = listing.get("results") or []
    if not results:
        raise RuntimeError(
            f"No audio_separate artifacts for recording {recording_id}. "
            "Confirm the bot used audio_separate_raw and the call has ended."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")

    saved: list[Path] = []
    async with httpx.AsyncClient(timeout=120.0) as http:
        for artifact in results:
            status = (artifact.get("status") or {}).get("code")
            fmt = artifact.get("format") or "raw"
            download_url = (artifact.get("data") or {}).get("download_url")
            if status != "done" or not download_url:
                logger.warning(
                    "Skipping artifact %s (status=%s, has_url=%s)",
                    artifact.get("id"),
                    status,
                    bool(download_url),
                )
                continue

            parts_resp = await http.get(download_url)
            parts_resp.raise_for_status()
            parts = parts_resp.json()
            if not isinstance(parts, list):
                parts = parts.get("parts") or parts.get("results") or [parts]

            for index, part in enumerate(parts):
                participant = part.get("participant") or {}
                name = _safe_name(
                    str(participant.get("name") or participant.get("id") or f"part{index}")
                )
                part_url = part.get("download_url")
                if not part_url:
                    logger.warning("Part %s for %s has no download_url", index, name)
                    continue

                ext = "raw" if fmt == "raw" else ("mp3" if fmt in {"mp3", "ogg"} else fmt)
                for candidate in (".mp3", ".ogg", ".wav", ".webm", ".raw"):
                    if candidate in part_url.split("?")[0].lower():
                        ext = candidate.lstrip(".")
                        break

                dest = out_dir / f"{name}-{index}.{ext}"
                logger.info(
                    "Downloading %s (%.1fs) -> %s",
                    name,
                    float(part.get("duration") or 0.0),
                    dest.name,
                )
                media = await http.get(part_url)
                media.raise_for_status()
                dest.write_bytes(media.content)
                saved.append(dest)
                logger.info("Saved %s (%d bytes)", dest, dest.stat().st_size)
                if ext == "raw":
                    wav_path = _wrap_raw_as_wav(dest)
                    saved.append(wav_path)
                    logger.info("Wrote playable %s", wav_path.name)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bot_id", help="Recall bot id from send_bot")
    parser.add_argument(
        "--wait",
        type=float,
        default=180.0,
        help="Seconds to wait for audio_separate status=done (0 = don't wait)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: runs/recall-<bot_id>)",
    )
    args = parser.parse_args()
    out_dir = args.out or (_RUNS / f"recall-{args.bot_id}")

    try:
        paths = asyncio.run(fetch_bot_audio(args.bot_id, wait_s=args.wait, out_dir=out_dir))
    except (RecallAPIError, RuntimeError, TimeoutError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    if not paths:
        logger.error("No participant audio files downloaded")
        raise SystemExit(1)
    logger.info("Done. %d file(s) in %s", len(paths), out_dir)


if __name__ == "__main__":
    main()
