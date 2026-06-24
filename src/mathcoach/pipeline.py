"""Application-level orchestration for the full MathCoach pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from mathcoach.agents.problem_understanding import ProblemUnderstandingAgent
from mathcoach.agents.solving_planning import (
    FailedAssertion,
    PlanningFeedback,
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
from mathcoach.agents.trace import AgentRunResult, LLMStepTrace
from mathcoach.schemas.inputs import UserQuery
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.schemas.teaching import TeachingExplanation
from mathcoach.schemas.verification import SolvingVerification


@dataclass
class PipelineAgents:
    understanding: Any
    planning: Any
    verification: Any
    teaching: Any


class TraceStepSummary(BaseModel):
    kind: str
    attempt: int
    status: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    parsed_payload: dict[str, Any] | None = None
    validation_error: str | None = None
    raw_response_preview: str | None = None


class AgentTraceSummary(BaseModel):
    agent_name: str
    steps: list[TraceStepSummary] = Field(default_factory=list)


class PipelineStage(BaseModel):
    key: str
    label: str
    agent_name: str
    status: Literal["succeeded", "failed"]
    duration_s: float
    attempt: int = 1
    error: str | None = None


class PipelineUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class PipelineResult(BaseModel):
    question: str
    student_level: str | None = None
    explanation_style: str | None = None
    analysis: ProblemUnderstanding
    plan: SolvingPlan
    verification: SolvingVerification
    teaching: TeachingExplanation
    stages: list[PipelineStage]
    trace: list[AgentTraceSummary]
    usage: PipelineUsage
    duration_s: float
    planning_attempts: int = 1


class PipelineExecutionError(RuntimeError):
    """Raised when one stage of the MathCoach pipeline fails."""

    def __init__(
        self,
        *,
        stage_key: str,
        message: str,
        stages: list[PipelineStage],
        trace: list[AgentTraceSummary] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage_key = stage_key
        self.message = message
        self.stages = stages
        self.trace = trace or []


_STAGE_LABELS = {
    "understanding": "题目理解",
    "planning": "解题规划",
    "verification": "求解验证",
    "teaching": "教学讲解",
}

# Verification verdicts that should trigger a re-plan + re-verify loop.
_PLANNING_RETRY_STATES = {"failed", "error"}
# Maximum number of extra planning attempts after the first one fails.
_MAX_PLANNING_RETRIES = 2

# A progress callback receives JSON-serializable event dicts as the pipeline
# advances. See `_emit` below for the event shapes.
EventCallback = Callable[[dict[str, Any]], None]


def run_mathcoach_pipeline(
    query: UserQuery,
    *,
    agents: PipelineAgents | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    on_event: EventCallback | None = None,
) -> PipelineResult:
    """Run the four-agent MathCoach workflow and return a UI/API friendly result.

    When ``on_event`` is provided it is called with progress events as each
    stage starts/completes, on verification retries, and on final completion.
    Events are plain JSON-serializable dicts (see ``_emit``).
    """
    started = time.monotonic()
    resolved_agents = agents or _build_default_agents(agent_kwargs or {})
    stages: list[PipelineStage] = []
    traces: list[AgentTraceSummary] = []

    run_results: list[AgentRunResult[Any]] = []

    u_result = _run_stage(
        key="understanding",
        agent=resolved_agents.understanding,
        input_data=query,
        stages=stages,
        traces=traces,
        on_event=on_event,
    )
    run_results.append(u_result)

    p_result = _run_stage(
        key="planning",
        agent=resolved_agents.planning,
        input_data=SolvingPlanningInput(
            analysis=u_result.output,
            original_question=query.question,
        ),
        stages=stages,
        traces=traces,
        attempt=1,
        on_event=on_event,
    )
    run_results.append(p_result)

    v_result = _run_stage(
        key="verification",
        agent=resolved_agents.verification,
        input_data=SolvingVerificationInput(
            analysis=u_result.output,
            plan=p_result.output,
            original_question=query.question,
        ),
        stages=stages,
        traces=traces,
        attempt=1,
        on_event=on_event,
    )
    run_results.append(v_result)

    planning_attempts = 1
    while (
        v_result.output.verification.status in _PLANNING_RETRY_STATES
        and planning_attempts <= _MAX_PLANNING_RETRIES
    ):
        attempt = planning_attempts + 1
        feedback = _build_planning_feedback(p_result.output, v_result)
        _emit(
            on_event,
            {
                "type": "retry",
                "attempt": attempt,
                "verification_status": v_result.output.verification.status,
                "detail": v_result.output.verification.detail,
                "failed_assertions": [
                    {
                        "description": fa.description,
                        "expr": fa.expr,
                        "expected": fa.expected,
                        "detail": fa.detail,
                    }
                    for fa in feedback.failed_assertions
                ],
            },
        )
        p_result = _run_stage(
            key="planning",
            agent=resolved_agents.planning,
            input_data=SolvingPlanningInput(
                analysis=u_result.output,
                original_question=query.question,
                feedback=feedback,
            ),
            stages=stages,
            traces=traces,
            attempt=attempt,
            on_event=on_event,
        )
        run_results.append(p_result)
        v_result = _run_stage(
            key="verification",
            agent=resolved_agents.verification,
            input_data=SolvingVerificationInput(
                analysis=u_result.output,
                plan=p_result.output,
                original_question=query.question,
            ),
            stages=stages,
            traces=traces,
            attempt=attempt,
            on_event=on_event,
        )
        run_results.append(v_result)
        planning_attempts += 1

    t_result = _run_stage(
        key="teaching",
        agent=resolved_agents.teaching,
        input_data=TeachingExplanationInput(
            analysis=u_result.output,
            plan=p_result.output,
            verification=v_result.output,
            original_question=query.question,
            student_level=query.student_level,
            explanation_style=query.explanation_style,
        ),
        stages=stages,
        traces=traces,
        attempt=1,
        on_event=on_event,
    )
    run_results.append(t_result)

    result = PipelineResult(
        question=query.question,
        student_level=query.student_level,
        explanation_style=query.explanation_style,
        analysis=u_result.output,
        plan=p_result.output,
        verification=v_result.output,
        teaching=t_result.output,
        stages=stages,
        trace=traces,
        usage=_sum_usage(run_results),
        duration_s=time.monotonic() - started,
        planning_attempts=planning_attempts,
    )
    _emit(on_event, {"type": "done", "result": result.model_dump()})
    return result


def _emit(on_event: EventCallback | None, event: dict[str, Any]) -> None:
    """Dispatch a progress event, ignoring the call when no callback is set."""
    if on_event is not None:
        on_event(event)


def _build_default_agents(agent_kwargs: dict[str, Any]) -> PipelineAgents:
    return PipelineAgents(
        understanding=ProblemUnderstandingAgent(**agent_kwargs),
        planning=SolvingPlanningAgent(**agent_kwargs),
        verification=SolvingVerificationAgent(**agent_kwargs),
        teaching=TeachingExplanationAgent(**agent_kwargs),
    )


def _run_stage(
    *,
    key: str,
    agent: Any,
    input_data: Any,
    stages: list[PipelineStage],
    traces: list[AgentTraceSummary],
    attempt: int = 1,
    on_event: EventCallback | None = None,
) -> AgentRunResult[Any]:
    started = time.monotonic()
    agent_name = getattr(agent, "name", agent.__class__.__name__)
    _emit(
        on_event,
        {
            "type": "stage_started",
            "key": key,
            "label": _STAGE_LABELS[key],
            "agent_name": agent_name,
            "attempt": attempt,
        },
    )
    try:
        result = agent.run_with_trace(input_data)
    except Exception as exc:  # noqa: BLE001 - normalize for API callers
        stage = PipelineStage(
            key=key,
            label=_STAGE_LABELS[key],
            agent_name=agent_name,
            status="failed",
            duration_s=time.monotonic() - started,
            attempt=attempt,
            error=f"{type(exc).__name__}: {exc}",
        )
        stages.append(stage)
        _emit(
            on_event,
            {
                "type": "stage_failed",
                "key": key,
                "label": _STAGE_LABELS[key],
                "agent_name": agent_name,
                "attempt": attempt,
                "error": stage.error,
            },
        )
        raise PipelineExecutionError(
            stage_key=key,
            message=stage.error or str(exc),
            stages=stages,
            trace=traces,
        ) from exc

    duration_s = time.monotonic() - started
    stages.append(
        PipelineStage(
            key=key,
            label=_STAGE_LABELS[key],
            agent_name=result.trace.agent_name,
            status="succeeded",
            duration_s=duration_s,
            attempt=attempt,
        )
    )
    traces.append(_summarize_trace(result))
    _emit(
        on_event,
        {
            "type": "stage_completed",
            "key": key,
            "label": _STAGE_LABELS[key],
            "agent_name": result.trace.agent_name,
            "attempt": attempt,
            "duration_s": duration_s,
            "output": result.output.model_dump(),
        },
    )
    return result


def _build_planning_feedback(
    plan: SolvingPlan,
    v_result: AgentRunResult[SolvingVerification],
) -> PlanningFeedback:
    """Assemble corrective feedback from a failed verification attempt."""
    verification = v_result.output.verification
    return PlanningFeedback(
        previous_method=plan.method,
        previous_steps=list(plan.steps),
        previous_solution_steps=list(v_result.output.solution_steps),
        verification_status=verification.status,
        verification_detail=verification.detail,
        failed_assertions=_extract_failed_assertions(v_result),
    )


def _extract_failed_assertions(
    v_result: AgentRunResult[SolvingVerification],
) -> list[FailedAssertion]:
    """Pull per-assertion failure detail from the SymPy verifier tool step."""
    failed: list[FailedAssertion] = []
    for step in v_result.trace.steps:
        if step.kind != "tool" or not step.parsed_payload:
            continue
        for item in step.parsed_payload.get("items", []):
            result = item.get("result") or {}
            if result.get("status") in _PLANNING_RETRY_STATES:
                failed.append(
                    FailedAssertion(
                        description=item.get("description"),
                        expr=str(item.get("expr", "")),
                        expected=item.get("expected"),
                        detail=result.get("detail"),
                    )
                )
    return failed


def _summarize_trace(result: AgentRunResult[Any]) -> AgentTraceSummary:
    return AgentTraceSummary(
        agent_name=result.trace.agent_name,
        steps=[_summarize_step(step) for step in result.trace.steps],
    )


def _summarize_step(step: LLMStepTrace) -> TraceStepSummary:
    return TraceStepSummary(
        kind=step.kind,
        attempt=step.attempt,
        status=step.status,
        model=step.model,
        prompt_tokens=step.prompt_tokens,
        completion_tokens=step.completion_tokens,
        total_tokens=step.total_tokens,
        parsed_payload=step.parsed_payload,
        validation_error=step.validation_error,
        raw_response_preview=_preview(step.raw_response),
    )


def _preview(value: str | None, limit: int = 1200) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def _sum_usage(results: list[AgentRunResult[Any]]) -> PipelineUsage:
    prompt = 0
    completion = 0
    total = 0
    for result in results:
        for step in result.trace.steps:
            if step.kind == "tool":
                continue
            prompt += step.prompt_tokens or 0
            completion += step.completion_tokens or 0
            total += step.total_tokens or 0
    return PipelineUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )
