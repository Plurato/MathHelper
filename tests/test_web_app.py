from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mathcoach.pipeline import (
    PipelineExecutionError,
    PipelineResult,
    PipelineStage,
    PipelineUsage,
)
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.schemas.teaching import TeachingExplanation
from mathcoach.schemas.verification import (
    AnswerItem,
    SolvingVerification,
    VerificationResult,
)
from mathcoach.web.app import create_app


def test_health_reports_model_and_key_status(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = TestClient(create_app(require_api_key=False))

    res = client.get("/api/health")

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["openrouter_api_key_present"] is True
    assert body["default_model"]


def test_index_is_served():
    client = TestClient(create_app(require_api_key=False))

    res = client.get("/")

    assert res.status_code == 200
    assert "MathCoach" in res.text


def test_index_loads_katex_auto_render_assets():
    client = TestClient(create_app(require_api_key=False))

    res = client.get("/")

    assert res.status_code == 200
    assert "katex.min.css" in res.text
    assert "katex.min.js" in res.text
    assert "auto-render.min.js" in res.text
    assert "/static/vendor/katex/" in res.text
    assert "cdn.jsdelivr.net" not in res.text


def test_local_katex_assets_are_served():
    client = TestClient(create_app(require_api_key=False))

    css = client.get("/static/vendor/katex/katex.min.css")
    js = client.get("/static/vendor/katex/katex.min.js")
    auto_render = client.get("/static/vendor/katex/contrib/auto-render.min.js")

    assert css.status_code == 200
    assert "@font-face" in css.text
    assert js.status_code == 200
    assert "katex" in js.text.lower()
    assert auto_render.status_code == 200
    assert "renderMathInElement" in auto_render.text


def test_frontend_assets_are_served():
    client = TestClient(create_app(require_api_key=False))

    css = client.get("/static/app.css")
    js = client.get("/static/app.js")

    assert css.status_code == 200
    assert "mathcoach-shell" in css.text
    assert js.status_code == 200
    assert "renderResult" in js.text


def test_frontend_renders_latex_after_dynamic_results():
    js = Path("src/mathcoach/web/static/app.js").read_text(encoding="utf-8")

    assert "function renderMath" in js
    assert "renderMathInElement" in js
    assert 'left: "$"' in js
    assert 'left: "$$"' in js
    assert "renderMath(els.resultWorkspace)" in js


def test_frontend_css_allows_health_pill_to_wrap_on_mobile():
    css = Path("src/mathcoach/web/static/app.css").read_text(encoding="utf-8")

    assert ".health-pill" in css
    assert "overflow-wrap: anywhere" in css
    assert "white-space: normal" in css


def test_serve_script_imports():
    script = Path("scripts/serve_web.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "uvicorn.run" in text
    assert "create_app" in text


def test_solve_returns_pipeline_result():
    client = TestClient(
        create_app(pipeline_runner=_fake_runner, require_api_key=False)
    )

    res = client.post(
        "/api/solve",
        json={
            "question": "解方程 x^2 - 5x + 6 = 0。",
            "student_level": "高中",
            "explanation_style": "详细版",
        },
    )

    assert res.status_code == 200
    body = res.json()
    assert body["analysis"]["problem_type"] == "一元二次方程"
    assert body["plan"]["method"] == "因式分解"
    assert body["verification"]["verification"]["status"] == "passed"
    assert body["teaching"]["explanation"] == "先因式分解，再代入验证。"


def test_solve_missing_key_returns_structured_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    client = TestClient(create_app(require_api_key=True))

    res = client.post("/api/solve", json={"question": "1+1"})

    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "missing_api_key"


def test_solve_blank_question_returns_validation_error():
    client = TestClient(create_app(require_api_key=False))

    res = client.post("/api/solve", json={"question": "   "})

    assert res.status_code == 422
    assert res.json()["detail"]["code"] == "invalid_question"


def test_solve_pipeline_failure_returns_stage_error():
    client = TestClient(
        create_app(pipeline_runner=_failing_runner, require_api_key=False)
    )

    res = client.post("/api/solve", json={"question": "1+1"})

    assert res.status_code == 500
    detail = res.json()["detail"]
    assert detail["code"] == "pipeline_failed"
    assert detail["stage_key"] == "planning"
    assert "planner exploded" in detail["message"]


def _fake_runner(query, *, agent_kwargs=None):
    return PipelineResult(
        question=query.question,
        student_level=query.student_level,
        explanation_style=query.explanation_style,
        analysis=ProblemUnderstanding(
            problem_type="一元二次方程",
            knowledge_points=["因式分解"],
            conditions={"方程": "x^2 - 5x + 6 = 0"},
            goal="求 x",
            difficulty="简单",
        ),
        plan=SolvingPlan(
            method="因式分解",
            steps=["化为 (x-2)(x-3)=0", "得到 x=2 或 x=3"],
            key_steps=["因式分解"],
            warnings=["不要漏根"],
        ),
        verification=SolvingVerification(
            solution_steps=["(x-2)(x-3)=0", "x=2 或 x=3"],
            answer=[
                AnswerItem(label="x_1", latex="$2$", sympy="2", numeric=2),
                AnswerItem(label="x_2", latex="$3$", sympy="3", numeric=3),
            ],
            verification=VerificationResult(
                method="SymPy", status="passed", confidence=0.95
            ),
            assertions=[],
        ),
        teaching=TeachingExplanation(
            explanation="先因式分解，再代入验证。",
            key_points=["零因子性质"],
            common_mistakes=["漏写一个根"],
            practice_questions=["解方程 x^2-4x+3=0。"],
            learning_advice="多练习因式分解。",
        ),
        stages=[
            PipelineStage(
                key="understanding",
                label="题目理解",
                agent_name="ProblemUnderstandingAgent",
                status="succeeded",
                duration_s=0.01,
            )
        ],
        trace=[],
        usage=PipelineUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        duration_s=0.02,
    )


def _failing_runner(query, *, agent_kwargs=None):
    raise PipelineExecutionError(
        stage_key="planning",
        message="RuntimeError: planner exploded",
        stages=[
            PipelineStage(
                key="planning",
                label="解题规划",
                agent_name="SolvingPlanningAgent",
                status="failed",
                duration_s=0.01,
                error="RuntimeError: planner exploded",
            )
        ],
    )
