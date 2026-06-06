"""Solving verification agent implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from mathcoach.agents.base import BaseAgent
from mathcoach.prompts.solving_verification import (
    SOLVING_VERIFICATION_FEW_SHOT,
    SOLVING_VERIFICATION_SYSTEM_PROMPT,
)
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.schemas.verification import SolvingVerification


@dataclass(frozen=True)
class SolvingVerificationInput:
    """Input bundle for the solving verification agent."""

    analysis: ProblemUnderstanding
    plan: SolvingPlan
    original_question: str | None = None


class SolvingVerificationAgent(BaseAgent[SolvingVerificationInput, SolvingVerification]):
    """Execute a detailed solution from the plan and verify the result."""

    name = "SolvingVerificationAgent"
    system_prompt = SOLVING_VERIFICATION_SYSTEM_PROMPT
    output_schema = SolvingVerification

    def build_user_prompt(self, input_data: SolvingVerificationInput) -> str:
        """Build the user prompt from analysis and plan."""
        combined = {
            "problem_type": input_data.analysis.problem_type,
            "knowledge_points": input_data.analysis.knowledge_points,
            "conditions": input_data.analysis.conditions,
            "goal": input_data.analysis.goal,
            "plan": {
                "method": input_data.plan.method,
                "steps": input_data.plan.steps,
            },
        }
        sections = [f"Problem and plan:\n{json.dumps(combined, ensure_ascii=False, indent=2)}"]
        if input_data.original_question:
            sections.append(f"Original question:\n{input_data.original_question.strip()}")

        few_shot = json.dumps(
            SOLVING_VERIFICATION_FEW_SHOT, ensure_ascii=False, indent=2
        )
        return (
            "\n\n".join(sections)
            + "\n\nExample:\n"
            + few_shot
            + "\n\nNow solve and verify the problem above, return the JSON object."
        )
