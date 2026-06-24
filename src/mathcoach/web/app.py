"""FastAPI application for the local MathCoach web UI."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mathcoach.config import Settings
from mathcoach.pipeline import PipelineExecutionError, run_mathcoach_pipeline
from mathcoach.schemas.inputs import UserQuery

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Sentinel pushed onto the SSE queue when the worker thread is finished.
_STREAM_DONE = object()


class SolveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    student_level: str | None = None
    explanation_style: str | None = None
    model: str | None = None
    include_trace: bool = True


def create_app(
    *,
    pipeline_runner: Callable = run_mathcoach_pipeline,
    stream_pipeline_runner: Callable | None = None,
    require_api_key: bool = True,
) -> FastAPI:
    """Create the local MathCoach web app.

    ``pipeline_runner``, ``stream_pipeline_runner`` and ``require_api_key`` are
    injectable so endpoint tests can exercise route behavior without calling
    external LLM services. The streaming runner must accept an ``on_event``
    keyword; it defaults to the real pipeline.
    """
    app = FastAPI(title="MathCoach", version="0.1.0")
    app.state.pipeline_runner = pipeline_runner
    app.state.stream_pipeline_runner = stream_pipeline_runner or run_mathcoach_pipeline
    app.state.require_api_key = require_api_key

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        settings = Settings.from_env()
        return {
            "ok": True,
            "default_model": settings.default_model,
            "openrouter_base_url": settings.openrouter_base_url,
            "openrouter_api_key_present": bool(settings.openrouter_api_key),
        }

    @app.post("/api/solve")
    def solve(request: SolveRequest):
        query, agent_kwargs = _prepare_request(request, require_api_key)

        try:
            return pipeline_runner(query, agent_kwargs=agent_kwargs)
        except PipelineExecutionError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "pipeline_failed",
                    "stage_key": exc.stage_key,
                    "message": exc.message,
                    "stages": [stage.model_dump() for stage in exc.stages],
                },
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "solve_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                },
            ) from exc

    @app.post("/api/solve/stream")
    def solve_stream(request: SolveRequest):
        query, agent_kwargs = _prepare_request(request, require_api_key)
        runner = app.state.stream_pipeline_runner
        return StreamingResponse(
            _run_event_stream(runner, query, agent_kwargs),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _prepare_request(
    request: SolveRequest, require_api_key: bool
) -> tuple[UserQuery, dict[str, Any] | None]:
    """Validate a solve request and build the pipeline inputs."""
    settings = Settings.from_env()
    if require_api_key and not settings.openrouter_api_key:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "missing_api_key",
                "message": "OPENROUTER_API_KEY is not configured.",
            },
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_question",
                "message": "Question must not be empty.",
            },
        )

    model = request.model.strip() if request.model else None
    agent_kwargs = {"model": model} if model else None
    query = UserQuery(
        question=question,
        student_level=request.student_level,
        explanation_style=request.explanation_style,
    )
    return query, agent_kwargs


def _run_event_stream(
    runner: Callable,
    query: UserQuery,
    agent_kwargs: dict[str, Any] | None,
):
    """Run the pipeline in a worker thread, yielding SSE frames per event."""
    event_queue: queue.Queue[Any] = queue.Queue()

    def worker() -> None:
        try:
            runner(
                query,
                agent_kwargs=agent_kwargs,
                on_event=event_queue.put,
            )
        except PipelineExecutionError as exc:
            event_queue.put(
                {
                    "type": "error",
                    "code": "pipeline_failed",
                    "stage_key": exc.stage_key,
                    "message": exc.message,
                    "stages": [stage.model_dump() for stage in exc.stages],
                }
            )
        except Exception as exc:  # noqa: BLE001 - normalize for stream clients
            event_queue.put(
                {
                    "type": "error",
                    "code": "solve_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            event_queue.put(_STREAM_DONE)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        event = event_queue.get()
        if event is _STREAM_DONE:
            break
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    thread.join()
