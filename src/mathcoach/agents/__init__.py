"""Agent implementations."""

from mathcoach.agents.base import BaseAgent
from mathcoach.agents.problem_understanding import ProblemUnderstandingAgent
from mathcoach.agents.solving_planning import SolvingPlanningAgent
from mathcoach.agents.solving_verification import (
    SolvingVerificationAgent,
    SolvingVerificationInput,
)
from mathcoach.agents.teaching_explanation import (
    TeachingExplanationAgent,
    TeachingExplanationInput,
)
from mathcoach.agents.trace import AgentRunResult, AgentRunTrace, LLMStepTrace

__all__ = [
    "AgentRunResult",
    "AgentRunTrace",
    "BaseAgent",
    "LLMStepTrace",
    "ProblemUnderstandingAgent",
    "SolvingPlanningAgent",
    "SolvingVerificationAgent",
    "SolvingVerificationInput",
    "TeachingExplanationAgent",
    "TeachingExplanationInput",
]
