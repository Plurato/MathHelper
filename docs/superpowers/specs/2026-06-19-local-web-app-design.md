# MathCoach Local Web App Design

## Goal

Deliver MathCoach-Agent as a complete local web application: one command starts a Python service that serves both the API and a polished browser UI for solving math problems with the existing four-agent pipeline.

## Confirmed Product Direction

The user selected a local integrated web app. The UI direction is an A+B hybrid:

- A-style first screen: focused question input, student controls, clear four-agent flow status, and a prominent solve action.
- B-style result reading: a workspace layout optimized for longer solution content, with stable navigation across answer summary, analysis, plan, verification, teaching, mistakes, practice questions, and trace details.

The visual tone should be minimal, premium, spacious, and instructional. The first screen should be the usable tool, not a marketing landing page.

## Architecture

Add a thin application layer above the existing agents:

```text
Browser UI
  ↓ POST /api/solve
Local Web Server
  ↓ calls
mathcoach.pipeline
  ↓ orchestrates
ProblemUnderstanding → SolvingPlanning → SolvingVerification → TeachingExplanation
```

The pipeline layer owns sequencing, timing, token aggregation, error normalization, and response shaping. Web routes do not assemble agent internals directly.

## Backend Design

Use a Python ASGI app served locally. The preferred implementation is FastAPI plus Uvicorn because it gives typed request/response handling, simple static serving, and testable endpoints without adding a JavaScript build chain.

Required routes:

- `GET /`: serve the main web UI.
- `GET /api/health`: return readiness, configured model, and whether an OpenRouter API key is present.
- `POST /api/solve`: accept `question`, optional `student_level`, optional `explanation_style`, optional `model`, and optional trace/debug flag; return the full structured solution.

The service should be started by:

```bash
.venv/bin/python scripts/serve_web.py
```

The script should default to `127.0.0.1:8000` and print the local URL.

## Pipeline Response

The solve response should include:

- `analysis`: `ProblemUnderstanding`
- `plan`: `SolvingPlan`
- `verification`: `SolvingVerification`
- `teaching`: `TeachingExplanation`
- `stages`: ordered stage metadata with status and duration
- `trace`: compact per-agent trace data suitable for expandable UI panels
- `usage`: prompt tokens, completion tokens, total tokens when available
- `duration_s`: total elapsed time

On failure, the API should return a structured error with a clear message and the stage that failed when known.

## Frontend Design

Use static HTML, CSS, and vanilla JavaScript served by the Python app. Avoid a Node/Vite build for this version.

The interface should include:

- Large math question textarea.
- Segmented or select controls for student level and explanation style.
- Optional model input collapsed behind advanced settings.
- Four-stage agent progress strip.
- Answer summary with verification status and confidence.
- Result workspace with sections for analysis, plan, detailed solution, teaching explanation, common mistakes, practice questions, and trace/debug.
- Loading, empty, success, and error states.

The layout must work on desktop and mobile without overlapping text or controls.

## Error Handling

Handle these cases explicitly:

- Missing `OPENROUTER_API_KEY`
- Empty question
- OpenRouter/API failure
- Agent JSON validation failure
- SymPy verifier failure or non-verifiable assertions

Errors should be readable in the UI and machine-readable in API tests.

## Testing

Maintain the existing baseline:

```bash
.venv/bin/python -m pytest -q
```

Add tests for:

- Pipeline orchestration with fake agents or fake LLM clients.
- API health and solve endpoint behavior.
- Static frontend assets being served.
- Error response shape for validation and pipeline failures.

Manual/browser verification should start the local app, submit at least one real math problem using the available OpenRouter key, and confirm the result sections render correctly.

## Out of Scope

This version will not add accounts, saved history, image OCR, RAG, multi-turn chat, deployment hosting, or a frontend build system. Those can remain future extensions after the local app is stable.
