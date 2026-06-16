"""Solving verification output schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

VerifiableKind = Literal[
    "equation_roots",
    "function_extrema",
    "trig_identity",
    "expression_value",
    "system_solution",
    "none",
]


class VerifiableArtifact(BaseModel):
    """Machine-checkable form of an answer for an external tool (e.g. SymPy).

    The LLM emits this alongside its narrative solution so a Python tool can
    verify the result independently. Each `kind` carries a different `payload`
    shape; see `mathcoach.tools.sympy_verifier` for accepted fields.

    All math expressions inside `payload` MUST be valid SymPy strings:
    use `*` for multiplication, `**` for power, lowercase function names
    (`sin`, `cos`, `sqrt`, `log`, `exp`), and `pi` / `E` for constants.
    Do NOT use LaTeX.
    """

    kind: VerifiableKind = Field(
        ..., description="Category of verification the tool should perform."
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Kind-specific arguments for the verifier.",
    )


class VerificationResult(BaseModel):
    """Verification subsection for the solving verification output."""

    method: str = Field(..., description="Verification method used, e.g. SymPy or numerical.")
    status: str = Field(
        ...,
        description=(
            "Verification conclusion: passed / failed / error / skipped / not_verifiable."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1.",
    )
    detail: str | None = Field(
        default=None,
        description="Optional human-readable detail produced by the verifier.",
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
    verifiable: VerifiableArtifact | None = Field(
        default=None,
        description="Machine-checkable form of the answer for external verification.",
    )
