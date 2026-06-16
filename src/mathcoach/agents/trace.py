"""Execution trace models for agent observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

TOutput = TypeVar("TOutput")


@dataclass(frozen=True)
class LLMStepTrace:
    """One step in an agent run — either an LLM call (`kind="llm"`) or a
    tool invocation such as the SymPy verifier (`kind="tool"`)."""

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
    kind: Literal["llm", "tool"] = "llm"


@dataclass(frozen=True)
class AgentRunTrace:
    agent_name: str
    system_prompt: str
    user_prompt: str
    steps: list[LLMStepTrace] = field(default_factory=list)


@dataclass(frozen=True)
class AgentRunResult(Generic[TOutput]):
    output: TOutput
    trace: AgentRunTrace
