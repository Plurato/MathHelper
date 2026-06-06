"""Solving verification output schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Verification subsection for the solving verification output."""

    method: str = Field(..., description="Verification method used, e.g. SymPy or numerical.")
    status: str = Field(..., description="Verification conclusion, e.g. passed / failed.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1.",
    )


class SolvingVerification(BaseModel):
    """Detailed solution and verification produced by the solving verification agent."""

    solution_steps: list[str] = Field(
        default_factory=list,
        description="Ordered, detailed solution steps with intermediate results.",
    )
    answer: dict[str, Any] = Field(
        default_factory=dict,
        description="Final answer expressed as key-value pairs.",
    )
    verification: VerificationResult = Field(
        ...,
        description="Verification result including method, status and confidence.",
    )
