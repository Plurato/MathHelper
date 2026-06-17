"""Run the agent pipeline over a list of EvalProblems and grade outputs."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import IO, Any

from mathcoach.agents.problem_understanding import ProblemUnderstandingAgent
from mathcoach.agents.solving_planning import (
    SolvingPlanningAgent,
    SolvingPlanningInput,
)
from mathcoach.agents.solving_verification import (
    SolvingVerificationAgent,
    SolvingVerificationInput,
)
from mathcoach.agents.teaching_explanation import (
    TeachingExplanationAgent,
    TeachingExplanationInput,
)
from mathcoach.agents.trace import AgentRunResult
from mathcoach.eval import grader
from mathcoach.eval.types import EvalProblem, EvalRow
from mathcoach.schemas.inputs import UserQuery
from mathcoach.schemas.verification import AnswerItem, SolvingVerification
from mathcoach.utils.trace_printer import print_agent_trace


def run(
    problems: list[EvalProblem],
    *,
    full_pipeline: bool = False,
    traces_dir: Path | None = None,
    agent_kwargs: dict[str, Any] | None = None,
) -> list[EvalRow]:
    if traces_dir is not None:
        traces_dir.mkdir(parents=True, exist_ok=True)

    agent_kwargs = agent_kwargs or {}
    return [
        _run_one(
            p,
            full_pipeline=full_pipeline,
            traces_dir=traces_dir,
            agent_kwargs=agent_kwargs,
        )
        for p in problems
    ]


def _run_one(
    problem: EvalProblem,
    *,
    full_pipeline: bool,
    traces_dir: Path | None,
    agent_kwargs: dict[str, Any],
) -> EvalRow:
    started = time.monotonic()
    trace_buf = io.StringIO()

    pipeline_status: str = "ok"
    pipeline_error = ""
    v_result: AgentRunResult[SolvingVerification] | None = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        v_result, tokens = _run_pipeline(
            problem.question,
            full_pipeline=full_pipeline,
            agent_kwargs=agent_kwargs,
            out=trace_buf,
        )
        prompt_tokens, completion_tokens = tokens
    except Exception as exc:  # noqa: BLE001
        pipeline_status = "failed"
        pipeline_error = f"{type(exc).__name__}: {exc}"

    duration = time.monotonic() - started

    if traces_dir is not None:
        (traces_dir / f"{problem.id}.log").write_text(trace_buf.getvalue(), encoding="utf-8")

    if v_result is None:
        return EvalRow(
            id=problem.id,
            group=problem.group,
            type=problem.type,
            expected_verifier=problem.expected_verifier,
            correct=None,
            grader_status="error",
            grader_layer="error",
            grader_reason=pipeline_error or "pipeline did not produce verification",
            verifier_status=None,
            verifier_confidence=None,
            n_assertions=0,
            n_passed=0,
            n_failed=0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_s=duration,
            pipeline_status="failed",
            pipeline_error=pipeline_error,
            pipeline_answer_repr="",
            truth_answer_repr=_answer_repr(problem.truth.answer),
        )

    verification = v_result.output
    grader_result = grader.compare(verification.answer, problem.truth.answer)

    grader_status: str
    if grader_result.correct is None and grader_result.layer == "error":
        grader_status = "error"
    elif grader_result.correct is None and grader_result.layer == "no_sympy":
        grader_status = "no_sympy"
    else:
        grader_status = "ok"

    n_total, n_passed, n_failed, _n_error = _extract_assertion_counts(v_result)

    return EvalRow(
        id=problem.id,
        group=problem.group,
        type=problem.type,
        expected_verifier=problem.expected_verifier,
        correct=grader_result.correct,
        grader_status=grader_status,  # type: ignore[arg-type]
        grader_layer=grader_result.layer,
        grader_reason=grader_result.reason,
        verifier_status=verification.verification.status,
        verifier_confidence=verification.verification.confidence,
        n_assertions=n_total,
        n_passed=n_passed,
        n_failed=n_failed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_s=duration,
        pipeline_status=pipeline_status,  # type: ignore[arg-type]
        pipeline_error=pipeline_error,
        pipeline_answer_repr=_answer_repr(verification.answer),
        truth_answer_repr=_answer_repr(problem.truth.answer),
    )


def _run_pipeline(
    question: str,
    *,
    full_pipeline: bool,
    agent_kwargs: dict[str, Any],
    out: IO[str] | None = None,
) -> tuple[AgentRunResult[SolvingVerification], tuple[int, int]]:
    user_query = UserQuery(question=question)

    understanding_agent = ProblemUnderstandingAgent(**agent_kwargs)
    planning_agent = SolvingPlanningAgent(**agent_kwargs)
    verification_agent = SolvingVerificationAgent(**agent_kwargs)

    u_result = understanding_agent.run_with_trace(user_query)
    print_agent_trace(u_result, show_prompts=False, out=out)

    p_input = SolvingPlanningInput(
        analysis=u_result.output, original_question=question
    )
    p_result = planning_agent.run_with_trace(p_input)
    print_agent_trace(p_result, show_prompts=False, out=out)

    v_input = SolvingVerificationInput(
        analysis=u_result.output,
        plan=p_result.output,
        original_question=question,
    )
    v_result = verification_agent.run_with_trace(v_input)
    print_agent_trace(v_result, show_prompts=False, out=out)

    if full_pipeline:
        teaching_agent = TeachingExplanationAgent(**agent_kwargs)
        t_input = TeachingExplanationInput(
            analysis=u_result.output,
            plan=p_result.output,
            verification=v_result.output,
            original_question=question,
        )
        t_result = teaching_agent.run_with_trace(t_input)
        print_agent_trace(t_result, show_prompts=False, out=out)
        results: list[AgentRunResult[Any]] = [u_result, p_result, v_result, t_result]
    else:
        results = [u_result, p_result, v_result]

    prompt_tokens = sum(_sum_tokens(r, "prompt_tokens") for r in results)
    completion_tokens = sum(_sum_tokens(r, "completion_tokens") for r in results)

    return v_result, (prompt_tokens, completion_tokens)


def _sum_tokens(result: AgentRunResult[Any], field: str) -> int:
    total = 0
    for step in result.trace.steps:
        if step.kind == "tool":
            continue
        v = getattr(step, field, None)
        if v:
            total += int(v)
    return total


def _extract_assertion_counts(
    v_result: AgentRunResult[SolvingVerification],
) -> tuple[int, int, int, int]:
    """Return ``(n_total, n_passed, n_failed, n_error)`` for the verifier run.

    Reads the per-status counts from the ``kind="tool"`` step appended by
    :class:`SolvingVerificationAgent`. When ``assertions`` is empty the agent
    skips the tool step; in that case ``n_total`` falls back to the length of
    the agent's self-reported ``assertions`` list and the rest are zero.
    """
    for step in reversed(v_result.trace.steps):
        if step.kind == "tool" and isinstance(step.parsed_payload, dict):
            p = step.parsed_payload
            return (
                int(p.get("n_total", 0)),
                int(p.get("n_passed", 0)),
                int(p.get("n_failed", 0)),
                int(p.get("n_error", 0)),
            )
    n = len(v_result.output.assertions or [])
    return n, 0, 0, 0


def _answer_repr(items: list[AnswerItem]) -> str:
    parts = []
    for it in items:
        sy = it.sympy if it.sympy is not None else "<no sympy>"
        parts.append(f"{it.label}={sy}")
    return "; ".join(parts)
