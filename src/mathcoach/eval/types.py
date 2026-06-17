"""Data models used across the eval harness."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mathcoach.schemas.verification import AnswerItem

GroupLiteral = Literal["A", "B", "C"]
DifficultyLiteral = Literal["easy", "medium", "hard"]
ExpectedVerifierLiteral = Literal["should_pass", "edge_case", "unverifiable"]
GraderLayer = Literal["structure", "exact", "numeric", "set", "no_sympy", "error"]
PipelineStatus = Literal["ok", "failed"]
GraderStatus = Literal["ok", "no_sympy", "error"]


class TruthSpec(BaseModel):
    answer: list[AnswerItem]


class EvalProblem(BaseModel):
    id: str
    group: GroupLiteral
    type: str
    knowledge_points: list[str]
    difficulty: DifficultyLiteral
    expected_verifier: ExpectedVerifierLiteral
    question: str
    truth: TruthSpec
    notes: str | None = None


class GraderResult(BaseModel):
    correct: bool | None
    layer: GraderLayer
    reason: str = ""


class EvalRow(BaseModel):
    id: str
    group: GroupLiteral
    type: str
    expected_verifier: ExpectedVerifierLiteral
    correct: bool | None
    grader_status: GraderStatus
    grader_layer: GraderLayer
    grader_reason: str
    verifier_status: str | None
    verifier_confidence: float | None
    n_assertions: int
    n_passed: int
    n_failed: int
    prompt_tokens: int
    completion_tokens: int
    duration_s: float
    pipeline_status: PipelineStatus
    pipeline_error: str = ""
    pipeline_answer_repr: str = ""
    truth_answer_repr: str = ""


class RunMeta(BaseModel):
    timestamp: str
    git_sha: str
    model_name: str
    agents: list[str]
    n_problems: int
    problems_path: str = ""
    output_dir: str = ""


class AggregateMetrics(BaseModel):
    """Computed from rows; lives in report.md only (not a stored artifact)."""

    n_total: int
    n_correct: int
    n_incorrect: int
    n_grader_skipped: int = Field(
        default=0, description="correct=None due to no_sympy or grader error"
    )
    n_pipeline_failed: int = 0
    correct_rate: float = 0.0
    fp_rate: float = 0.0
    fn_rate: float = 0.0
