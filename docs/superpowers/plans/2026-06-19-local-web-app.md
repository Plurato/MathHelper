# Local Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete local MathCoach web application served by one Python command, with a tested pipeline wrapper, API routes, and polished static UI.

**Architecture:** Add `mathcoach.pipeline` as the orchestration boundary above the existing four agents, then expose it through `mathcoach.web.app`. The same Python ASGI service serves `/api/*` and static frontend files from `src/mathcoach/web/static/`.

**Tech Stack:** Python 3.10+, Pydantic v2, FastAPI, Uvicorn, pytest, vanilla HTML/CSS/JavaScript.

---

## File Structure

- Modify `pyproject.toml`: add FastAPI/Uvicorn runtime dependencies and HTTPX for API tests.
- Modify `requirements.txt`: mirror runtime dependencies used outside editable installs.
- Create `src/mathcoach/pipeline.py`: pipeline request/response models, stage metadata, trace summaries, orchestration, error normalization, token aggregation.
- Create `src/mathcoach/web/__init__.py`: web package marker.
- Create `src/mathcoach/web/app.py`: FastAPI factory, health route, solve route, static file serving.
- Create `src/mathcoach/web/static/index.html`: one-screen tool shell and result workspace.
- Create `src/mathcoach/web/static/app.css`: premium minimal responsive styling.
- Create `src/mathcoach/web/static/app.js`: client-side solve flow, rendering, health check, error states.
- Create `scripts/serve_web.py`: local startup script for the integrated app.
- Create `tests/test_pipeline.py`: unit tests for orchestration, usage, stages, and failure shape.
- Create `tests/test_web_app.py`: API/static tests using FastAPI TestClient.
- Modify `README.md`: add local web app startup instructions.

## Task 1: Add Web Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies in metadata**

Update `[project].dependencies` to include:

```toml
"fastapi>=0.115",
"uvicorn>=0.30",
```

Update `[project.optional-dependencies].dev` to include:

```toml
dev = ["pytest", "ruff", "httpx"]
```

- [ ] **Step 2: Mirror runtime dependencies**

Add to `requirements.txt`:

```text
fastapi>=0.115
uvicorn>=0.30
```

- [ ] **Step 3: Install dependencies into the existing venv**

Run:

```bash
.venv/bin/python -m pip install -e ".[dev,solve]"
```

Expected: installation succeeds and keeps `sympy`/`numpy` available.

- [ ] **Step 4: Verify baseline**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: existing tests pass before behavior work continues.

## Task 2: Build Tested Pipeline Orchestration

**Files:**
- Create: `tests/test_pipeline.py`
- Create: `src/mathcoach/pipeline.py`

- [ ] **Step 1: Write failing success-path test**

Create fake agents whose `run_with_trace()` methods return `AgentRunResult` objects for `ProblemUnderstanding`, `SolvingPlan`, `SolvingVerification`, and `TeachingExplanation`. Test:

```python
def test_pipeline_runs_four_agents_and_shapes_response():
    query = UserQuery(question="解方程 x^2 - 5x + 6 = 0。", student_level="高中")
    result = run_mathcoach_pipeline(query, agents=_fake_agents())

    assert result.analysis.problem_type == "一元二次方程"
    assert result.plan.method == "因式分解"
    assert result.verification.verification.status == "passed"
    assert result.teaching.explanation
    assert [stage.key for stage in result.stages] == [
        "understanding",
        "planning",
        "verification",
        "teaching",
    ]
    assert all(stage.status == "succeeded" for stage in result.stages)
    assert result.usage.prompt_tokens == 40
    assert result.usage.completion_tokens == 20
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_pipeline.py::test_pipeline_runs_four_agents_and_shapes_response -q
```

Expected: failure because `mathcoach.pipeline` does not exist.

- [ ] **Step 3: Implement minimal pipeline models and success path**

Implement:

```python
class PipelineStage(BaseModel):
    key: str
    label: str
    agent_name: str
    status: Literal["succeeded", "failed"]
    duration_s: float
    error: str | None = None

class PipelineUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class PipelineResult(BaseModel):
    question: str
    student_level: str | None
    explanation_style: str | None
    analysis: ProblemUnderstanding
    plan: SolvingPlan
    verification: SolvingVerification
    teaching: TeachingExplanation
    stages: list[PipelineStage]
    trace: list[AgentTraceSummary]
    usage: PipelineUsage
    duration_s: float
```

Add `PipelineAgents` and `run_mathcoach_pipeline(query, agents=None, agent_kwargs=None)`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -q
```

Expected: current pipeline tests pass.

- [ ] **Step 5: Write failing failure-path test**

Add a fake planning agent that raises `RuntimeError("planner exploded")`. Test:

```python
def test_pipeline_failure_includes_stage_and_partial_stages():
    with pytest.raises(PipelineExecutionError) as exc_info:
        run_mathcoach_pipeline(UserQuery(question="1+1"), agents=_fake_agents(fail_at="planning"))

    err = exc_info.value
    assert err.stage_key == "planning"
    assert "planner exploded" in err.message
    assert err.stages[0].status == "succeeded"
    assert err.stages[1].status == "failed"
```

- [ ] **Step 6: Verify RED, implement failure normalization, verify GREEN**

Run the new test and confirm it fails because `PipelineExecutionError` is missing. Implement the exception and failed stage capture. Re-run:

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -q
```

Expected: all pipeline tests pass.

## Task 3: Build Tested Web API

**Files:**
- Create: `tests/test_web_app.py`
- Create: `src/mathcoach/web/__init__.py`
- Create: `src/mathcoach/web/app.py`

- [ ] **Step 1: Write failing health/static tests**

Use `fastapi.testclient.TestClient`:

```python
def test_health_reports_model_and_key_status(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = TestClient(create_app(require_api_key=False))

    res = client.get("/api/health")

    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert res.json()["openrouter_api_key_present"] is True

def test_index_is_served():
    client = TestClient(create_app(require_api_key=False))
    res = client.get("/")
    assert res.status_code == 200
    assert "MathCoach" in res.text
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_web_app.py::test_health_reports_model_and_key_status tests/test_web_app.py::test_index_is_served -q
```

Expected: failure because `mathcoach.web.app` and static files do not exist.

- [ ] **Step 3: Implement app factory and placeholder index**

Create `create_app(pipeline_runner=run_mathcoach_pipeline, require_api_key=True)`, `GET /api/health`, and `GET /` using `FileResponse`.

- [ ] **Step 4: Verify health/static GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_web_app.py -q
```

Expected: currently written web tests pass.

- [ ] **Step 5: Write failing solve endpoint tests**

Add tests:

```python
def test_solve_returns_pipeline_result():
    client = TestClient(create_app(pipeline_runner=_fake_runner, require_api_key=False))
    res = client.post("/api/solve", json={"question": "解方程 x^2 - 5x + 6 = 0。"})
    assert res.status_code == 200
    assert res.json()["analysis"]["problem_type"] == "一元二次方程"

def test_solve_missing_key_returns_structured_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    client = TestClient(create_app(require_api_key=True))
    res = client.post("/api/solve", json={"question": "1+1"})
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "missing_api_key"

def test_solve_pipeline_failure_returns_stage_error():
    client = TestClient(create_app(pipeline_runner=_failing_runner, require_api_key=False))
    res = client.post("/api/solve", json={"question": "1+1"})
    assert res.status_code == 500
    assert res.json()["detail"]["stage_key"] == "planning"
```

- [ ] **Step 6: Verify RED, implement solve route, verify GREEN**

Run the tests, confirm failures, implement request/response/error handling, then run:

```bash
.venv/bin/python -m pytest tests/test_web_app.py -q
```

Expected: all web API tests pass.

## Task 4: Create the Static Frontend

**Files:**
- Modify: `src/mathcoach/web/static/index.html`
- Create: `src/mathcoach/web/static/app.css`
- Create: `src/mathcoach/web/static/app.js`
- Modify: `tests/test_web_app.py`

- [ ] **Step 1: Write failing static asset tests**

Add:

```python
def test_frontend_assets_are_served():
    client = TestClient(create_app(require_api_key=False))
    css = client.get("/static/app.css")
    js = client.get("/static/app.js")
    assert css.status_code == 200
    assert "mathcoach-shell" in css.text
    assert js.status_code == 200
    assert "renderResult" in js.text
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_web_app.py::test_frontend_assets_are_served -q
```

Expected: failure because CSS/JS do not exist.

- [ ] **Step 3: Implement HTML/CSS/JS**

The UI must include:

```html
<textarea id="questionInput"></textarea>
<select id="studentLevel"></select>
<select id="explanationStyle"></select>
<button id="solveButton" type="submit">开始解题</button>
<section id="agentFlow"></section>
<section id="resultWorkspace"></section>
<section id="errorPanel"></section>
```

The JavaScript must:

```js
async function solveProblem(payload) {
  const response = await fetch("/api/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw data;
  return data;
}

function renderResult(result) {
  // Render answer, verification, analysis, plan, solution, teaching, practice, trace.
}
```

CSS must provide responsive A+B hybrid layout and stable non-overlapping controls.

- [ ] **Step 4: Verify frontend asset GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_web_app.py -q
```

Expected: all web app tests pass.

## Task 5: Add Startup Script and Documentation

**Files:**
- Create: `scripts/serve_web.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing startup import test**

Add to `tests/test_web_app.py`:

```python
def test_serve_script_imports():
    script = Path("scripts/serve_web.py")
    assert script.exists()
    assert "uvicorn.run" in script.read_text(encoding="utf-8")
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_web_app.py::test_serve_script_imports -q
```

Expected: failure because script does not exist.

- [ ] **Step 3: Implement script**

Create CLI with `--host`, `--port`, and `--reload` flags. It should insert `src/` on `sys.path`, create the app, print `MathCoach Web: http://HOST:PORT`, and run Uvicorn.

- [ ] **Step 4: Update README**

Add a local web app section:

```bash
pip install -e ".[dev,solve]"
.venv/bin/python scripts/serve_web.py
```

Mention `.env` must contain `OPENROUTER_API_KEY`.

- [ ] **Step 5: Verify script/docs tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_web_app.py -q
```

Expected: all web tests pass.

## Task 6: Full Verification and Browser Acceptance

**Files:**
- No new files unless fixes are required.

- [ ] **Step 1: Run full automated tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Start local app**

Run:

```bash
.venv/bin/python scripts/serve_web.py --port 8000
```

Expected: service starts and prints `http://127.0.0.1:8000`.

- [ ] **Step 3: Browser smoke test**

Open `http://127.0.0.1:8000` in the in-app browser. Submit:

```text
解方程 x^2 - 5x + 6 = 0。
```

Expected:

- Page accepts input without console errors.
- Agent flow shows completed stages after response.
- Final answer section renders labeled answers.
- Verification status/confidence appears.
- Analysis, plan, solution steps, teaching explanation, mistakes, practice questions, and trace sections are visible.

- [ ] **Step 4: Check browser console**

Read console logs.

Expected: no JavaScript errors.

- [ ] **Step 5: Final git/status check**

Run:

```bash
git status --short
```

Expected: only intentional project files are modified; unrelated `AGENTS.md` remains untracked unless the user asks to include it.
