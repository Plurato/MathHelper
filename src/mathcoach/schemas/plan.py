"""Solving plan output schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SolvingPlan(BaseModel):
    """Step-by-step solution plan produced by the solving planning agent."""

    method: str = Field(..., description="Primary recommended solving method.")
    steps: list[str] = Field(
        default_factory=list,
        description="Ordered solution steps.",
    )
    alternative_method: str | None = Field(
        default=None,
        description="Optional backup solving approach.",
    )
    key_steps: list[str] = Field(
        default_factory=list,
        description="Critical steps that must not be skipped.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Common mistakes or pitfalls to watch for.",
    )
