"""Console helpers for printing agent execution traces."""

from __future__ import annotations

import json
import sys
from typing import IO, Any

from mathcoach.agents.trace import AgentRunResult, AgentRunTrace, LLMStepTrace


def print_agent_trace(
    result: AgentRunResult[Any],
    *,
    show_prompts: bool = True,
    out: IO[str] | None = None,
) -> None:
    """Render an agent's execution trace.

    Args:
        result: Agent run output and trace produced by ``BaseAgent.run_with_trace``.
        show_prompts: When True, include the system and user prompts.
        out: Stream to write to. Defaults to ``sys.stdout``. Pass an explicit
            stream (e.g. ``io.StringIO``) to capture output without mutating
            the process-global stdout.
    """
    out = out if out is not None else sys.stdout
    trace = result.trace
    _print_header(trace.agent_name, out)
    if show_prompts:
        _print_prompts(trace, out)
    for step in trace.steps:
        _print_step(step, out)
    _print_final_output(result.output.model_dump(), out)


def _print_header(agent_name: str, out: IO[str]) -> None:
    line = "=" * 60
    print(f"\n{line}", file=out)
    print(f" Agent: {agent_name}", file=out)
    print(line, file=out)


def _print_prompts(trace: AgentRunTrace, out: IO[str]) -> None:
    print("\n--- Prompts ---", file=out)
    print("\n[System Prompt]", file=out)
    print(trace.system_prompt.strip(), file=out)
    print("\n[User Prompt]", file=out)
    print(trace.user_prompt.strip(), file=out)


def _print_step(step: LLMStepTrace, out: IO[str]) -> None:
    if step.kind == "tool":
        _print_tool_step(step, out)
    else:
        _print_llm_step(step, out)


def _print_llm_step(step: LLMStepTrace, out: IO[str]) -> None:
    print(f"\n--- LLM Call (attempt {step.attempt}, status: {step.status}) ---", file=out)
    if step.model:
        print(f"Model: {step.model}", file=out)
    if step.total_tokens is not None:
        print(
            "Tokens: "
            f"prompt={step.prompt_tokens}, "
            f"completion={step.completion_tokens}, "
            f"total={step.total_tokens}",
            file=out,
        )
    if step.reasoning:
        print("\n[Reasoning]", file=out)
        print(step.reasoning.strip(), file=out)
    print("\n[Raw Response]", file=out)
    print(step.raw_response.strip(), file=out)
    if step.parsed_payload is not None:
        print("\n[Parsed JSON]", file=out)
        print(json.dumps(step.parsed_payload, ensure_ascii=False, indent=2), file=out)
    if step.validation_error:
        print("\n[Validation Error]", file=out)
        print(step.validation_error.strip(), file=out)


def _print_tool_step(step: LLMStepTrace, out: IO[str]) -> None:
    tool_name = (
        step.parsed_payload.get("tool")
        if isinstance(step.parsed_payload, dict)
        else None
    ) or step.model or "tool"
    print(f"\n--- Tool Call ({tool_name}, status: {step.status}) ---", file=out)

    payload = step.parsed_payload if isinstance(step.parsed_payload, dict) else None
    if payload:
        n_total = payload.get("n_total")
        if n_total is not None:
            print(
                f"Items: total={n_total}, "
                f"passed={payload.get('n_passed', 0)}, "
                f"failed={payload.get('n_failed', 0)}, "
                f"error={payload.get('n_error', 0)}, "
                f"skipped={payload.get('n_skipped', 0)}",
                file=out,
            )
        agg = payload.get("aggregated")
        if isinstance(agg, dict):
            print(
                f"Aggregated: status={agg.get('status')}, "
                f"confidence={agg.get('confidence')}, "
                f"method={agg.get('method')}",
                file=out,
            )
            if agg.get("detail"):
                print(f"Detail: {agg.get('detail')}", file=out)
        items = payload.get("items")
        if isinstance(items, list) and items:
            print("\n[Per-item Results]", file=out)
            for i, it in enumerate(items, start=1):
                desc = it.get("description") or _truncate(it.get("expr", ""), 60)
                res = it.get("result", {})
                line = (
                    f"  {i}. [{res.get('status')}] {desc}"
                    f"  → expected={it.get('expected')!r}"
                )
                print(line, file=out)
                if res.get("detail") and res.get("status") in ("failed", "error"):
                    print(f"      detail: {res['detail']}", file=out)

    print("\n[Submitted Assertions]", file=out)
    print(step.raw_response.strip(), file=out)


def _print_final_output(payload: dict[str, Any], out: IO[str]) -> None:
    print("\n--- Validated Output ---", file=out)
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=out)


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"
