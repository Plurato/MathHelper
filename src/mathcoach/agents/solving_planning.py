"""Solving planning agent implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from mathcoach.agents.base import BaseAgent
from mathcoach.prompts.solving_planning import (
    SOLVING_PLANNING_FEW_SHOT,
    SOLVING_PLANNING_SYSTEM_PROMPT,
)
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding


@dataclass(frozen=True)
class FailedAssertion:
    """A single assertion that the SymPy verifier rejected on a prior attempt."""

    description: str | None
    expr: str
    expected: object
    detail: str | None = None


@dataclass(frozen=True)
class PlanningFeedback:
    """Error feedback from a failed verification, fed back into re-planning.

    Carries enough context for the planner to diagnose *why* the previous
    attempt failed and produce a corrected plan: the previous plan, the
    solution steps that were executed, the overall verifier verdict, and the
    individual assertions that failed.
    """

    previous_method: str
    previous_steps: list[str] = field(default_factory=list)
    previous_solution_steps: list[str] = field(default_factory=list)
    verification_status: str | None = None
    verification_detail: str | None = None
    failed_assertions: list[FailedAssertion] = field(default_factory=list)


@dataclass(frozen=True)
class SolvingPlanningInput:
    """Input bundle for the solving planning agent."""

    analysis: ProblemUnderstanding
    original_question: str | None = None
    feedback: PlanningFeedback | None = None


class SolvingPlanningAgent(BaseAgent[SolvingPlanningInput, SolvingPlan]):
    """Create a solution plan from structured problem analysis."""

    name = "SolvingPlanningAgent"
    system_prompt = SOLVING_PLANNING_SYSTEM_PROMPT
    output_schema = SolvingPlan

    def build_user_prompt(self, input_data: SolvingPlanningInput) -> str:
        """Build the user prompt from structured analysis."""
        analysis_json = json.dumps(
            input_data.analysis.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
        sections = [f"Problem analysis:\n{analysis_json}"]
        if input_data.original_question:
            sections.append(f"Original question:\n{input_data.original_question.strip()}")

        if input_data.feedback is not None:
            sections.append(_build_feedback_section(input_data.feedback))

        few_shot = json.dumps(SOLVING_PLANNING_FEW_SHOT, ensure_ascii=False, indent=2)
        return (
            "\n\n".join(sections)
            + "\n\nExample:\n"
            + few_shot
            + "\n\nNow create a solving plan and return the JSON object."
        )


def _build_feedback_section(feedback: PlanningFeedback) -> str:
    """Render verification failure feedback as a corrective prompt section."""
    payload: dict[str, object] = {
        "previous_method": feedback.previous_method,
        "previous_steps": feedback.previous_steps,
        "previous_solution_steps": feedback.previous_solution_steps,
        "verification_status": feedback.verification_status,
        "verification_detail": feedback.verification_detail,
        "failed_assertions": [
            {
                "description": item.description,
                "expr": item.expr,
                "expected": item.expected,
                "detail": item.detail,
            }
            for item in feedback.failed_assertions
        ],
    }
    feedback_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        "The previous solving plan was executed but FAILED verification. "
        "Diagnose why it failed using the details below, then produce a "
        "CORRECTED plan that fixes the root cause. Do not repeat the same "
        "mistake.\n"
        f"Verification feedback:\n{feedback_json}"
    )
