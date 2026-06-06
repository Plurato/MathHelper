# MathCoach-Agent

Multi-agent math coaching system that breaks problem solving into structured stages:
problem understanding, solving planning, verification, and teaching explanation.

Phase 1 implements the first two agents:
- **Problem Understanding Agent** — parses a math question into structured metadata.
- **Solving Planning Agent** — produces a step-by-step solution plan.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy the environment template and fill in your OpenRouter credentials:

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
```

## Run Demo

```bash
python scripts/demo_agents.py
python scripts/demo_agents.py --question "Solve x^2 - 5x + 6 = 0."
```

By default the demo prints full execution traces for each agent:
- prompts sent to the model
- reasoning content (when supported by the model provider)
- raw LLM response
- parsed JSON and validated output
- token usage

Optional flags:

```bash
python scripts/demo_agents.py --quiet          # final outputs only
python scripts/demo_agents.py --hide-prompts   # hide system/user prompts
python scripts/demo_agents.py --no-reasoning   # skip reasoning request
```

## Run Tests

```bash
pytest -q
```

## Project Layout

```
src/mathcoach/
  agents/       # Agent implementations
  llm/          # OpenRouter client
  prompts/      # Prompt templates
  schemas/      # Pydantic data models
  utils/        # Shared utilities
scripts/        # CLI demos
tests/          # Unit tests
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key | *(required for live runs)* |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `MATHCOACH_DEFAULT_MODEL` | Default LLM model | `openai/gpt-4o-mini` |
| `MATHCOACH_DEFAULT_TEMPERATURE` | Default sampling temperature | `0.2` |

## Extending

Future agents (solving/verification, teaching explanation) should inherit from
`BaseAgent` in `src/mathcoach/agents/base.py` and define their own schemas
under `src/mathcoach/schemas/`.
