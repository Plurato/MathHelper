"""Solving planning agent implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from mathcoach.agents.base import BaseAgent
from mathcoach.prompts.solving_planning import (
    SOLVING_PLANNING_FEW_SHOT,
    SOLVING_PLANNING_SYSTEM_PROMPT,
)
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding


@dataclass(frozen=True)
class SolvingPlanningInput:
    """Input bundle for the solving planning agent."""

    analysis: ProblemUnderstanding
    original_question: str | None = None


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

        few_shot = json.dumps(SOLVING_PLANNING_FEW_SHOT, ensure_ascii=False, indent=2)
        return (
            "\n\n".join(sections)
            + "\n\nExample:\n"
            + few_shot
            + "\n\nNow create a solving plan and return the JSON object."
        )
