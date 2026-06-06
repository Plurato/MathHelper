"""User-facing input schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserQuery(BaseModel):
    """Raw user input for the math coaching pipeline."""

    question: str = Field(..., min_length=1, description="The math problem text.")
    student_level: str | None = Field(
        default=None,
        description="Optional student level, e.g. middle school, high school, college.",
    )
    explanation_style: str | None = Field(
        default=None,
        description="Optional explanation style preference for downstream agents.",
    )
