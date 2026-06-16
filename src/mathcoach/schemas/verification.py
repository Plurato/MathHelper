"""Solving verification output schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Assertion(BaseModel):
    """A single machine-checkable claim emitted by the LLM during solving.

    All math expressions in `expr` and string-form `expected` must be valid
    SymPy syntax (no LaTeX). When `free_vars` is provided, the verifier
    samples each free variable in the given range; otherwise it tries a
    symbolic / numeric scalar comparison.
    """

    expr: str = Field(..., description="SymPy expression to evaluate.")
    expected: int | float | str | list[Any] | bool = Field(
        ...,
        description="What the LLM claims `expr` evaluates to.",
    )
    description: str | None = Field(default=None)
    free_vars: dict[str, list[float]] | None = Field(
        default=None,
        description="`{name: [low, high]}` sampling ranges for identity checks.",
    )
    tolerance: float | None = Field(default=None)


class AnswerItem(BaseModel):
    """One labeled answer with parallel display / tool / numeric forms.

    `latex`, `sympy`, and `numeric` MUST be semantically equivalent. The LLM
    is responsible for keeping them consistent.
    """

    label: str
    latex: str
    sympy: str | None = Field(default=None)
    numeric: float | None = Field(default=None)
    unit: str | None = Field(default=None)


class VerificationResult(BaseModel):
    method: str
    status: str = Field(
        ...,
        description="passed / failed / error / skipped / not_verifiable",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    detail: str | None = Field(default=None)


class SolvingVerification(BaseModel):
    solution_steps: list[str] = Field(default_factory=list)
    answer: list[AnswerItem] = Field(default_factory=list)
    verification: VerificationResult
    assertions: list[Assertion] | None = Field(default=None)
