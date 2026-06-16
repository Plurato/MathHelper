"""Solving verification agent implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from mathcoach.agents.base import BaseAgent
from mathcoach.agents.trace import AgentRunResult, LLMStepTrace
from mathcoach.prompts.solving_verification import (
    SOLVING_VERIFICATION_FEW_SHOT,
    SOLVING_VERIFICATION_SYSTEM_PROMPT,
    SOLVING_VERIFICATION_VERIFIABLE_EXAMPLES,
)
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.schemas.verification import SolvingVerification
from mathcoach.tools import sympy_verifier

# Cap LLM self-reported confidence when no machine-checkable artifact is given.
_LLM_SELF_REPORT_CONFIDENCE_CAP = 0.6


@dataclass(frozen=True)
class SolvingVerificationInput:
    """Input bundle for the solving verification agent."""

    analysis: ProblemUnderstanding
    plan: SolvingPlan
    original_question: str | None = None


class SolvingVerificationAgent(BaseAgent[SolvingVerificationInput, SolvingVerification]):
    """Execute a detailed solution from the plan and verify the result.

    After the LLM produces structured output, we run a SymPy-backed verifier
    on the `verifiable` artifact and overwrite `verification` with the tool's
    independent judgment. If no artifact (or kind="none") is supplied, we cap
    the LLM's self-assessed confidence so downstream consumers don't trust an
    unverified 1.0.
    """

    name = "SolvingVerificationAgent"
    system_prompt = SOLVING_VERIFICATION_SYSTEM_PROMPT
    output_schema = SolvingVerification

    def build_user_prompt(self, input_data: SolvingVerificationInput) -> str:
        """Build the user prompt from analysis and plan."""
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
        sections = [f"Problem and plan:\n{json.dumps(combined, ensure_ascii=False, indent=2)}"]
        if input_data.original_question:
            sections.append(f"Original question:\n{input_data.original_question.strip()}")

        few_shot = json.dumps(
            SOLVING_VERIFICATION_FEW_SHOT, ensure_ascii=False, indent=2
        )
        verifiable_examples = json.dumps(
            SOLVING_VERIFICATION_VERIFIABLE_EXAMPLES, ensure_ascii=False, indent=2
        )
        return (
            "\n\n".join(sections)
            + "\n\nExample:\n"
            + few_shot
            + "\n\nAdditional `verifiable` shapes (illustrative only — adapt to the problem):\n"
            + verifiable_examples
            + "\n\nNow solve and verify the problem above, return the JSON object."
        )

    def run_with_trace(
        self, input_data: SolvingVerificationInput
    ) -> AgentRunResult[SolvingVerification]:
        result = super().run_with_trace(input_data)
        artifact = result.output.verifiable

        if artifact is None or artifact.kind == "none":
            # No machine-checkable form. Don't trust LLM self-confidence > cap.
            current = result.output.verification
            if current.confidence > _LLM_SELF_REPORT_CONFIDENCE_CAP:
                result.output.verification = current.model_copy(
                    update={
                        "confidence": _LLM_SELF_REPORT_CONFIDENCE_CAP,
                        "detail": (
                            (current.detail + " | " if current.detail else "")
                            + "Confidence capped: no machine-checkable artifact provided."
                        ),
                    }
                )
            return result

        tool_result = sympy_verifier.verify(artifact)
        result.output.verification = tool_result
        result.trace.steps.append(
            LLMStepTrace(
                attempt=len(result.trace.steps) + 1,
                raw_response=json.dumps(
                    artifact.model_dump(), ensure_ascii=False, indent=2
                ),
                reasoning=None,
                parsed_payload={
                    "tool": "sympy_verifier",
                    "kind": artifact.kind,
                    "result": tool_result.model_dump(),
                },
                validation_error=None,
                # 'success' here means the tool ran; the math verdict lives in tool_result.status.
                status="success" if tool_result.status != "error" else "failed",
                model="sympy",
            )
        )
        return result
