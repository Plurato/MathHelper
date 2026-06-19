"""FastAPI application for the local MathCoach web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mathcoach.config import Settings
from mathcoach.pipeline import PipelineExecutionError, run_mathcoach_pipeline
from mathcoach.schemas.inputs import UserQuery

STATIC_DIR = Path(__file__).resolve().parent / "static"


class SolveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    student_level: str | None = None
    explanation_style: str | None = None
    model: str | None = None
    include_trace: bool = True


def create_app(
    *,
    pipeline_runner: Callable = run_mathcoach_pipeline,
    require_api_key: bool = True,
) -> FastAPI:
    """Create the local MathCoach web app.

    ``pipeline_runner`` and ``require_api_key`` are injectable so endpoint
    tests can exercise route behavior without calling external LLM services.
    """
    app = FastAPI(title="MathCoach", version="0.1.0")
    app.state.pipeline_runner = pipeline_runner
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

    return app
