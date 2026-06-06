"""Teaching explanation output schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TeachingExplanation(BaseModel):
    """Teaching-oriented explanation produced by the teaching explanation agent."""

    explanation: str = Field(
        ...,
        description="Plain-language explanation of the solution approach and key ideas.",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Core knowledge points and takeaways for the student.",
    )
    common_mistakes: list[str] = Field(
        default_factory=list,
        description="Common errors and misconceptions to watch for.",
    )
    practice_questions: list[str] = Field(
        default_factory=list,
        description="Similar practice or variant questions for reinforcement.",
    )
    learning_advice: str | None = Field(
        default=None,
        description="Personalized study advice based on the problem difficulty and concepts.",
    )
