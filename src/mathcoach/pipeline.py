"""Application-level orchestration for the full MathCoach pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

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


def run_mathcoach_pipeline(
    query: UserQuery,
    *,
    agents: PipelineAgents | None = None,
    agent_kwargs: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run the four-agent MathCoach workflow and return a UI/API friendly result."""
    started = time.monotonic()
    resolved_agents = agents or _build_default_agents(agent_kwargs or {})
    stages: list[PipelineStage] = []
    traces: list[AgentTraceSummary] = []

    u_result = _run_stage(
        key="understanding",
        agent=resolved_agents.understanding,
        input_data=query,
        stages=stages,
        traces=traces,
    )

    p_result = _run_stage(
        key="planning",
        agent=resolved_agents.planning,
        input_data=SolvingPlanningInput(
            analysis=u_result.output,
            original_question=query.question,
        ),
        stages=stages,
        traces=traces,
    )

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
    )

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
    )

    return PipelineResult(
        question=query.question,
        student_level=query.student_level,
        explanation_style=query.explanation_style,
        analysis=u_result.output,
        plan=p_result.output,
        verification=v_result.output,
        teaching=t_result.output,
        stages=stages,
        trace=traces,
        usage=_sum_usage([u_result, p_result, v_result, t_result]),
        duration_s=time.monotonic() - started,
    )


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
) -> AgentRunResult[Any]:
    started = time.monotonic()
    agent_name = getattr(agent, "name", agent.__class__.__name__)
    try:
        result = agent.run_with_trace(input_data)
    except Exception as exc:  # noqa: BLE001 - normalize for API callers
        stage = PipelineStage(
            key=key,
            label=_STAGE_LABELS[key],
            agent_name=agent_name,
            status="failed",
            duration_s=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        stages.append(stage)
        raise PipelineExecutionError(
            stage_key=key,
            message=stage.error or str(exc),
            stages=stages,
            trace=traces,
        ) from exc

    stages.append(
        PipelineStage(
            key=key,
            label=_STAGE_LABELS[key],
            agent_name=result.trace.agent_name,
            status="succeeded",
            duration_s=time.monotonic() - started,
        )
    )
    traces.append(_summarize_trace(result))
    return result


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
