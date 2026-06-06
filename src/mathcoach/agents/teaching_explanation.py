"""Teaching explanation agent implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from mathcoach.agents.base import BaseAgent
from mathcoach.prompts.teaching_explanation import (
    TEACHING_EXPLANATION_FEW_SHOT,
    TEACHING_EXPLANATION_SYSTEM_PROMPT,
)
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.schemas.teaching import TeachingExplanation
from mathcoach.schemas.verification import SolvingVerification


@dataclass(frozen=True)
class TeachingExplanationInput:
    """Input bundle for the teaching explanation agent."""

    analysis: ProblemUnderstanding
    plan: SolvingPlan
    verification: SolvingVerification
    original_question: str | None = None
    student_level: str | None = None
    explanation_style: str | None = None


class TeachingExplanationAgent(BaseAgent[TeachingExplanationInput, TeachingExplanation]):
    """Convert solution results into a student-friendly teaching explanation."""

    name = "TeachingExplanationAgent"
    system_prompt = TEACHING_EXPLANATION_SYSTEM_PROMPT
    output_schema = TeachingExplanation

    def build_user_prompt(self, input_data: TeachingExplanationInput) -> str:
        """Build the user prompt from all upstream results."""
        combined = {
            "problem_type": input_data.analysis.problem_type,
            "knowledge_points": input_data.analysis.knowledge_points,
            "conditions": input_data.analysis.conditions,
            "goal": input_data.analysis.goal,
            "difficulty": input_data.analysis.difficulty,
            "method": input_data.plan.method,
            "answer": input_data.verification.answer,
        }
        sections = [
            f"Complete solution summary:\n{json.dumps(combined, ensure_ascii=False, indent=2)}"
        ]
        if input_data.original_question:
            sections.append(f"Original question:\n{input_data.original_question.strip()}")
        if input_data.student_level:
            sections.append(f"Student level: {input_data.student_level}")
        if input_data.explanation_style:
            sections.append(f"Explanation style: {input_data.explanation_style}")

        few_shot = json.dumps(
            TEACHING_EXPLANATION_FEW_SHOT, ensure_ascii=False, indent=2
        )
        return (
            "\n\n".join(sections)
            + "\n\nExample:\n"
            + few_shot
            + "\n\nNow create a teaching explanation and return the JSON object."
        )
