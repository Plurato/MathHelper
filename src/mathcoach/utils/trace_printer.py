"""Console helpers for printing agent execution traces."""

from __future__ import annotations

import json
from typing import Any

from mathcoach.agents.trace import AgentRunResult, AgentRunTrace, LLMStepTrace


def print_agent_trace(result: AgentRunResult[Any], *, show_prompts: bool = True) -> None:
    """Print a formatted execution trace for one agent run."""
    trace = result.trace
    _print_header(trace.agent_name)
    if show_prompts:
        _print_prompts(trace)
    for step in trace.steps:
        _print_step(step)
    _print_final_output(result.output.model_dump())


def _print_header(agent_name: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f" Agent: {agent_name}")
    print(line)


def _print_prompts(trace: AgentRunTrace) -> None:
    print("\n--- Prompts ---")
    print("\n[System Prompt]")
    print(trace.system_prompt.strip())
    print("\n[User Prompt]")
    print(trace.user_prompt.strip())


def _print_step(step: LLMStepTrace) -> None:
    print(f"\n--- LLM Call (attempt {step.attempt}, status: {step.status}) ---")
    if step.model:
        print(f"Model: {step.model}")
    if step.total_tokens is not None:
        print(
            "Tokens: "
            f"prompt={step.prompt_tokens}, "
            f"completion={step.completion_tokens}, "
            f"total={step.total_tokens}"
        )
    if step.reasoning:
        print("\n[Reasoning]")
        print(step.reasoning.strip())
    print("\n[Raw Response]")
    print(step.raw_response.strip())
    if step.parsed_payload is not None:
        print("\n[Parsed JSON]")
        print(json.dumps(step.parsed_payload, ensure_ascii=False, indent=2))
    if step.validation_error:
        print("\n[Validation Error]")
        print(step.validation_error.strip())


def _print_final_output(payload: dict[str, Any]) -> None:
    print("\n--- Validated Output ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
