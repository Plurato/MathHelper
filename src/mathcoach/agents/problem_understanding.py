"""Problem understanding agent implementation."""

from __future__ import annotations

import json

from mathcoach.agents.base import BaseAgent
from mathcoach.prompts.problem_understanding import (
    PROBLEM_UNDERSTANDING_FEW_SHOT,
    PROBLEM_UNDERSTANDING_SYSTEM_PROMPT,
)
from mathcoach.schemas.inputs import UserQuery
from mathcoach.schemas.problem import ProblemUnderstanding


class ProblemUnderstandingAgent(BaseAgent[UserQuery, ProblemUnderstanding]):
    """Analyze a raw math question and return structured metadata."""

    name = "ProblemUnderstandingAgent"
    system_prompt = PROBLEM_UNDERSTANDING_SYSTEM_PROMPT
    output_schema = ProblemUnderstanding

    def build_user_prompt(self, input_data: UserQuery) -> str:
        """Build the user prompt from the raw question."""
        context_lines = [f"Question:\n{input_data.question.strip()}"]
        if input_data.student_level:
            context_lines.append(f"Student level: {input_data.student_level}")
        if input_data.explanation_style:
            context_lines.append(f"Explanation style: {input_data.explanation_style}")

        few_shot = json.dumps(PROBLEM_UNDERSTANDING_FEW_SHOT, ensure_ascii=False, indent=2)
        return (
            "\n\n".join(context_lines)
            + "\n\nExample:\n"
            + few_shot
            + "\n\nNow analyze the question above and return the JSON object."
        )
