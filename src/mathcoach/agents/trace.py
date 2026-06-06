"""Execution trace models for agent observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

TOutput = TypeVar("TOutput")


@dataclass(frozen=True)
class LLMStepTrace:
    """Trace for a single LLM call within an agent run."""

    attempt: int
    raw_response: str
    reasoning: str | None
    parsed_payload: dict[str, Any] | None
    validation_error: str | None
    status: Literal["success", "retry", "failed"]
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class AgentRunTrace:
    """Full execution trace for one agent invocation."""

    agent_name: str
    system_prompt: str
    user_prompt: str
    steps: list[LLMStepTrace] = field(default_factory=list)


@dataclass(frozen=True)
class AgentRunResult(Generic[TOutput]):
    """Validated agent output bundled with its execution trace."""

    output: TOutput
    trace: AgentRunTrace
