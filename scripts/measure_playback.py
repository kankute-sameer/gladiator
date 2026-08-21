"""Drive the meeting page in Chromium and run play_file scenarios.

The server event bus persists `playback_stats` / `flush_sent` / `flush_ack`
to `runs/*.jsonl`. This script rotates the server run file, opens the page
in Chromium (same AudioWorklet path the Meet tile uses), drives play_file,
then greps the JSONL for the latency table.

Usage (server must already be running on --port):
    python -m scripts.measure_playback --scenario depth60
    python -m scripts.measure_playback --scenario all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

from glad.config import settings
from glad.logging import configure_logging, get_logger

configure_logging(settings.log_level)
logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
WAV = ROOT / "assets" / "test_speech.wav"


async def _rotate_run(port: int, label: str) -> Path:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{port}/obs/run",
            params={"label": label},
            timeout=10.0,
        )
        response.raise_for_status()
        return Path(response.json()["path"])


async def _open_page(port: int):
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,
        args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
        ],
    )
    context = await browser.new_context()
    await context.grant_permissions(["microphone"])
    page = await context.new_page()
    await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
    await page.wait_for_function(
        "() => document.getElementById('status')?.textContent?.includes('Listening')",
        timeout=15000,
    )
    return playwright, browser, page


async def _play(
    port: int,
    *,
    chunk_ms: int = 20,
    jitter: int = 0,
    flush_after: float | None = None,
    duration: float | None = None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.play_file",
        str(WAV),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--chunk-ms",
        str(chunk_ms),
        "--jitter",
        str(jitter),
    ]
    if flush_after is not None:
        cmd.extend(["--flush-after", str(flush_after)])
    if duration is not None:
        cmd.extend(["--duration", str(duration)])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    logger.info("play_file exit=%s\n%s", proc.returncode, stdout.decode())
    if proc.returncode != 0:
        raise RuntimeError(f"play_file failed: {stdout.decode()}")


def _summarize(run_path: Path) -> dict:
    stats: list[dict] = []
    flush_sent_mono: float | None = None
    flush_ack_mono: float | None = None
    playback_start_mono: float | None = None
    for line in run_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["t"] == "playback_stats":
            stats.append(event)
        elif event["t"] == "flush_sent":
            flush_sent_mono = event["mono"]
        elif event["t"] == "flush_ack":
            flush_ack_mono = event["mono"]
        elif event["t"] == "playback_start" and playback_start_mono is None:
            playback_start_mono = event["mono"]

    # Only score depth during active playback, not the pre-roll settle window.
    if playback_start_mono is not None:
        stats = [s for s in stats if s["mono"] >= playback_start_mono]
        t0 = playback_start_mono
    elif stats:
        t0 = stats[0]["mono"]
    else:
        t0 = 0.0

    def depth_at(target_s: float) -> float | None:
        for event in stats:
            if event["mono"] - t0 >= target_s:
                return round(event["depth_ms"], 1)
        return None

    summary: dict = {"n_stats": len(stats), "run": str(run_path)}
    if stats:
        t_end = stats[-1]["mono"]
        summary["span_s"] = round(t_end - t0, 2)
        summary["depth_10s"] = depth_at(10.0)
        summary["depth_30s"] = depth_at(30.0)
        summary["depth_60s"] = depth_at(60.0)
        summary["depth_final"] = round(stats[-1]["depth_ms"], 1)
        summary["underruns_final"] = stats[-1]["underruns"]
        summary["underruns_start"] = stats[0]["underruns"]
        summary["underruns_delta"] = stats[-1]["underruns"] - stats[0]["underruns"]
        summary["depth_max"] = round(max(s["depth_ms"] for s in stats), 1)
        summary["depth_min"] = round(min(s["depth_ms"] for s in stats), 1)
        # Steady-state: ignore the first second of prebuffer fill.
        steady = [s for s in stats if s["mono"] - t0 >= 1.0]
        if steady:
            summary["depth_steady_mean"] = round(
                sum(s["depth_ms"] for s in steady) / len(steady), 1
            )
    if flush_sent_mono is not None and flush_ack_mono is not None:
        summary["flush_latency_ms"] = round((flush_ack_mono - flush_sent_mono) * 1000, 1)
    return summary


SCENARIOS = {
    "depth60": {"chunk_ms": 20, "jitter": 0, "duration": 60.0, "settle": 2.0},
    "chunk100": {"chunk_ms": 100, "jitter": 0, "duration": 60.0, "settle": 2.0},
    "jitter50": {"chunk_ms": 20, "jitter": 50, "duration": 60.0, "settle": 2.0},
    "flush": {"chunk_ms": 20, "jitter": 0, "flush_after": 3.0, "settle": 1.0},
}


async def run_scenario(name: str, port: int) -> dict:
    cfg = SCENARIOS[name]
    run_path = await _rotate_run(port, f"slice2a-{name}")

    playwright, browser, page = await _open_page(port)
    try:
        await asyncio.sleep(cfg.get("settle", 1.0))
        await _play(
            port,
            chunk_ms=cfg["chunk_ms"],
            jitter=cfg["jitter"],
            flush_after=cfg.get("flush_after"),
            duration=cfg.get("duration"),
        )
        await asyncio.sleep(1.5)
    finally:
        await browser.close()
        await playwright.stop()

    summary = _summarize(run_path)
    summary["scenario"] = name
    print(json.dumps(summary, indent=2))
    return summary


async def main_async(args: argparse.Namespace) -> None:
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = []
    for name in names:
        logger.info("=== scenario %s ===", name)
        results.append(await run_scenario(name, args.port))
        await asyncio.sleep(1.0)
    out = ROOT / "runs" / f"slice2a-summary-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote summary %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS.keys(), "all"],
        default="all",
    )
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
