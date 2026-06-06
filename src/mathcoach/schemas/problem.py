"""Problem understanding output schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProblemUnderstanding(BaseModel):
    """Structured analysis produced by the problem understanding agent."""

    problem_type: str = Field(..., description="Identified math problem category.")
    knowledge_points: list[str] = Field(
        default_factory=list,
        description="Relevant math concepts involved in the problem.",
    )
    conditions: dict[str, str] = Field(
        default_factory=dict,
        description="Known conditions extracted from the problem statement.",
    )
    goal: str = Field(..., description="What the problem asks the solver to find.")
    difficulty: Literal["简单", "中等", "困难"] = Field(
        ...,
        description="Estimated difficulty level.",
    )
