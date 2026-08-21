"""FastAPI application factory for the Glad meeting bot."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from glad.agent.script import load_question_set
from glad.agent.tools import declarations as tool_declarations
from glad.config import settings
from glad.live.session import LiveSession
from glad.logging import configure_logging, get_logger
from glad.obs import events
from glad.obs import runs as obs_runs
from glad.orchestrator import Orchestrator
from glad.transport import inbound, monitor, outbound, participants, transcript_inbound

configure_logging(settings.log_level)
logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "web" / "meeting"
MONITOR_STATIC_DIR = Path(__file__).resolve().parent / "web" / "monitor"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    run_path = events.start_run("slice3")
    app.state.run_path = str(run_path)

    question_set = load_question_set(settings.question_set)
    logger.info(
        "Glad starting (model=%s, voice=%s, question_set=%s, run=%s)",
        settings.gemini_model,
        settings.gemini_voice,
        question_set.id,
        run_path,
    )

    orchestrator = Orchestrator(question_set)
    live = LiveSession(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
        instruction_provider=orchestrator.build_instruction,
        tools=tool_declarations(question_set),
        tool_dispatcher=orchestrator.handle_tool_call,
        voice=settings.gemini_voice,
    )
    orchestrator.bind_live(live)
    outbound.set_frame_listener(orchestrator.on_inbound_frame)
    transcript_inbound.set_segment_listener(orchestrator.on_transcript_segment)
    participants.set_roster(orchestrator.state.roster)
    events.emit("run_meta", session_id=orchestrator.session_id, question_set_id=question_set.id)
    app.state.orchestrator = orchestrator
    app.state.question_set = question_set

    broadcast_task = asyncio.create_task(outbound.broadcast_loop())
    orchestrator_task = asyncio.create_task(orchestrator.run())
    try:
        yield
    finally:
        logger.info("Glad shutting down")
        events.emit("run_stop")
        events.stop_run()
        outbound.set_frame_listener(None)
        transcript_inbound.set_segment_listener(None)
        participants.set_roster(None)
        await live.close()
        for task in (broadcast_task, orchestrator_task):
            task.cancel()
        for task in (broadcast_task, orchestrator_task):
            with contextlib.suppress(asyncio.CancelledError):
                await task


def create_app() -> FastAPI:
    """Build the FastAPI app: health check, realtime websockets, and the static meeting page."""
    app = FastAPI(title="Glad", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/obs/run")
    async def get_run() -> dict[str, str | None]:
        path = events.current_run()
        return {"path": str(path) if path else None}

    @app.post("/obs/run")
    async def rotate_run(label: str = "slice2a") -> dict[str, str]:
        events.stop_run()
        path = events.start_run(label)
        return {"path": str(path)}

    @app.get("/state/questions")
    async def get_questions(request: Request) -> dict[str, Any]:
        """The full question set plus any answers already recorded, so the
        meeting page can render every row (pending or filled) on first
        load, before it has seen a single `answer.recorded` push."""
        question_set = request.app.state.question_set
        orchestrator: Orchestrator = request.app.state.orchestrator
        return {
            "question_set_id": question_set.id,
            "questions": [{"id": q.id, "text": q.text} for q in question_set.questions],
            "answers": {
                question_id: {"value": answer.value, "revision": answer.revision}
                for question_id, answer in orchestrator.state.answers.items()
            },
        }

    @app.get("/api/runs")
    async def api_list_runs() -> list[dict[str, Any]]:
        return obs_runs.list_runs()

    @app.get("/api/runs/{run_id}")
    async def api_load_run(run_id: str) -> dict[str, Any]:
        try:
            return obs_runs.load_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}") from exc

    @app.get("/monitor", include_in_schema=False)
    async def monitor_index() -> FileResponse:
        # The StaticFiles mount below 404s bare "/monitor" (no trailing slash).
        return FileResponse(MONITOR_STATIC_DIR / "index.html")

    app.include_router(inbound.router)
    app.include_router(outbound.router)
    app.include_router(transcript_inbound.router)
    app.include_router(participants.router)
    app.include_router(monitor.router)

    # Mounted before the catch-all "/" below so /monitor/* resolves here
    # first -- Starlette matches routes in registration order.
    app.mount("/monitor", StaticFiles(directory=MONITOR_STATIC_DIR, html=True), name="monitor")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="meeting")

    return app
