"""Console helpers for printing agent execution traces."""

from __future__ import annotations

import json
from typing import Any

from mathcoach.agents.trace import AgentRunResult, AgentRunTrace, LLMStepTrace


def print_agent_trace(result: AgentRunResult[Any], *, show_prompts: bool = True) -> None:
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
    if step.kind == "tool":
        _print_tool_step(step)
    else:
        _print_llm_step(step)


def _print_llm_step(step: LLMStepTrace) -> None:
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


def _print_tool_step(step: LLMStepTrace) -> None:
    tool_name = (
        step.parsed_payload.get("tool")
        if isinstance(step.parsed_payload, dict)
        else None
    ) or step.model or "tool"
    print(f"\n--- Tool Call ({tool_name}, status: {step.status}) ---")

    payload = step.parsed_payload if isinstance(step.parsed_payload, dict) else None
    if payload:
        n_total = payload.get("n_total")
        if n_total is not None:
            print(
                f"Items: total={n_total}, "
                f"passed={payload.get('n_passed', 0)}, "
                f"failed={payload.get('n_failed', 0)}, "
                f"error={payload.get('n_error', 0)}, "
                f"skipped={payload.get('n_skipped', 0)}"
            )
        agg = payload.get("aggregated")
        if isinstance(agg, dict):
            print(
                f"Aggregated: status={agg.get('status')}, "
                f"confidence={agg.get('confidence')}, "
                f"method={agg.get('method')}"
            )
            if agg.get("detail"):
                print(f"Detail: {agg.get('detail')}")
        items = payload.get("items")
        if isinstance(items, list) and items:
            print("\n[Per-item Results]")
            for i, it in enumerate(items, start=1):
                desc = it.get("description") or _truncate(it.get("expr", ""), 60)
                res = it.get("result", {})
                line = (
                    f"  {i}. [{res.get('status')}] {desc}"
                    f"  → expected={it.get('expected')!r}"
                )
                print(line)
                if res.get("detail") and res.get("status") in ("failed", "error"):
                    print(f"      detail: {res['detail']}")

    print("\n[Submitted Assertions]")
    print(step.raw_response.strip())


def _print_final_output(payload: dict[str, Any]) -> None:
    print("\n--- Validated Output ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"
