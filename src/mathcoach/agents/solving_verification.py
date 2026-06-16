"""Solving verification agent implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from mathcoach.agents.base import BaseAgent
from mathcoach.agents.trace import AgentRunResult, LLMStepTrace
from mathcoach.prompts.solving_verification import (
    SOLVING_VERIFICATION_FEW_SHOT,
    SOLVING_VERIFICATION_SYSTEM_PROMPT,
)
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.schemas.verification import (
    Assertion,
    SolvingVerification,
    VerificationResult,
)
from mathcoach.tools import sympy_verifier

# Cap on LLM self-reported confidence when no machine-checkable assertions are
# supplied; below the cap the LLM's value is preserved.
_LLM_SELF_REPORT_CONFIDENCE_CAP = 0.6
_MAX_FAILURE_DETAILS = 3


@dataclass(frozen=True)
class SolvingVerificationInput:
    analysis: ProblemUnderstanding
    plan: SolvingPlan
    original_question: str | None = None


class SolvingVerificationAgent(BaseAgent[SolvingVerificationInput, SolvingVerification]):
    """Execute the plan, then replace the LLM's self-reported verification
    with the SymPy verifier's aggregated judgment over its assertions list."""

    name = "SolvingVerificationAgent"
    system_prompt = SOLVING_VERIFICATION_SYSTEM_PROMPT
    output_schema = SolvingVerification

    def build_user_prompt(self, input_data: SolvingVerificationInput) -> str:
        combined = {
            "problem_type": input_data.analysis.problem_type,
            "knowledge_points": input_data.analysis.knowledge_points,
            "conditions": input_data.analysis.conditions,
            "goal": input_data.analysis.goal,
            "plan": {
                "method": input_data.plan.method,
                "steps": input_data.plan.steps,
            },
        }
        sections = [
            f"Problem and plan:\n{json.dumps(combined, ensure_ascii=False, indent=2)}"
        ]
        if input_data.original_question:
            sections.append(
                f"Original question:\n{input_data.original_question.strip()}"
            )

        few_shot = json.dumps(
            SOLVING_VERIFICATION_FEW_SHOT, ensure_ascii=False, indent=2
        )
        return (
            "\n\n".join(sections)
            + "\n\nExamples (each shows a different problem type with its assertions):\n"
            + few_shot
            + "\n\nNow solve and verify the problem above, return the JSON object."
        )

    def run_with_trace(
        self, input_data: SolvingVerificationInput
    ) -> AgentRunResult[SolvingVerification]:
        result = super().run_with_trace(input_data)
        assertions = result.output.assertions or []

        if not assertions:
            current = result.output.verification
            if current.confidence > _LLM_SELF_REPORT_CONFIDENCE_CAP:
                result.output.verification = current.model_copy(
                    update={
                        "confidence": _LLM_SELF_REPORT_CONFIDENCE_CAP,
                        "detail": _append_detail(
                            current.detail,
                            "Confidence capped: no assertions supplied for verification.",
                        ),
                    }
                )
            return result

        per_item: list[VerificationResult] = [
            sympy_verifier.verify(a) for a in assertions
        ]
        aggregated = _aggregate_results(per_item, assertions)
        result.output.verification = aggregated

        tool_payload = {
            "tool": "sympy_verifier",
            "n_total": len(assertions),
            "n_passed": sum(1 for r in per_item if r.status == "passed"),
            "n_failed": sum(1 for r in per_item if r.status == "failed"),
            "n_error": sum(1 for r in per_item if r.status == "error"),
            "n_skipped": sum(1 for r in per_item if r.status == "skipped"),
            "items": [
                {
                    "description": a.description,
                    "expr": a.expr,
                    "expected": a.expected,
                    "result": r.model_dump(),
                }
                for a, r in zip(assertions, per_item)
            ],
            "aggregated": aggregated.model_dump(),
        }
        result.trace.steps.append(
            LLMStepTrace(
                attempt=len(result.trace.steps) + 1,
                raw_response=json.dumps(
                    [a.model_dump() for a in assertions],
                    ensure_ascii=False,
                    indent=2,
                ),
                reasoning=None,
                parsed_payload=tool_payload,
                validation_error=None,
                status=("success" if aggregated.status != "error" else "failed"),
                model="sympy",
                kind="tool",
            )
        )
        return result


def _aggregate_results(
    results: list[VerificationResult], assertions: list[Assertion]
) -> VerificationResult:
    n = len(results)
    passed = [r for r in results if r.status == "passed"]
    failed_pairs = [
        (a, r) for a, r in zip(assertions, results) if r.status == "failed"
    ]
    error_pairs = [
        (a, r) for a, r in zip(assertions, results) if r.status == "error"
    ]

    if failed_pairs:
        n_failed = len(failed_pairs)
        sample = failed_pairs[:_MAX_FAILURE_DETAILS]
        details = "; ".join(
            f"[{a.description or a.expr[:40]}] {r.detail}" for a, r in sample
        )
        return VerificationResult(
            method=f"SymPy 多项断言验证（{n} 项）",
            status="failed",
            confidence=0.05,
            detail=f"{n_failed}/{n} failed | {details}",
        )

    if error_pairs:
        sample = error_pairs[0]
        return VerificationResult(
            method=f"SymPy 多项断言验证（{n} 项）",
            status="error",
            confidence=0.30,
            detail=(
                f"errors in {len(error_pairs)}/{n} items; "
                f"first: [{sample[0].description or sample[0].expr[:40]}] "
                f"{sample[1].detail}"
            ),
        )

    if not passed:
        first = results[0]
        return VerificationResult(
            method=first.method,
            status="skipped",
            confidence=first.confidence,
            detail=first.detail,
        )

    min_conf = min(r.confidence for r in passed)
    return VerificationResult(
        method=f"SymPy 多项断言验证（{n} 项全部通过）",
        status="passed",
        confidence=min_conf,
        detail=f"{len(passed)}/{n} passed",
    )


def _append_detail(existing: str | None, addition: str) -> str:
    if not existing:
        return addition
    return f"{existing} | {addition}"
