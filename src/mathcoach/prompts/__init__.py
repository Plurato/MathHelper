"""Prompt templates for MathCoach agents."""

from mathcoach.prompts.shared import MATH_FORMAT_RULES
from mathcoach.prompts.problem_understanding import (
    PROBLEM_UNDERSTANDING_FEW_SHOT,
    PROBLEM_UNDERSTANDING_SYSTEM_PROMPT,
)
from mathcoach.prompts.solving_planning import (
    SOLVING_PLANNING_FEW_SHOT,
    SOLVING_PLANNING_SYSTEM_PROMPT,
)
from mathcoach.prompts.solving_verification import (
    SOLVING_VERIFICATION_FEW_SHOT,
    SOLVING_VERIFICATION_SYSTEM_PROMPT,
)
from mathcoach.prompts.teaching_explanation import (
    TEACHING_EXPLANATION_FEW_SHOT,
    TEACHING_EXPLANATION_SYSTEM_PROMPT,
)

__all__ = [
    "MATH_FORMAT_RULES",
    "PROBLEM_UNDERSTANDING_FEW_SHOT",
    "PROBLEM_UNDERSTANDING_SYSTEM_PROMPT",
    "SOLVING_PLANNING_FEW_SHOT",
    "SOLVING_PLANNING_SYSTEM_PROMPT",
    "SOLVING_VERIFICATION_FEW_SHOT",
    "SOLVING_VERIFICATION_SYSTEM_PROMPT",
    "TEACHING_EXPLANATION_FEW_SHOT",
    "TEACHING_EXPLANATION_SYSTEM_PROMPT",
]
