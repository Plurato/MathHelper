"""Integration tests for the SolvingVerificationAgent's verify-and-aggregate path.

These tests mock the parent BaseAgent.run_with_trace to avoid hitting a real
LLM, then assert that the agent's verifier+aggregation pipeline behaves as
specified.
"""

from __future__ import annotations

from unittest.mock import patch

import mathcoach.agents.base as base_mod
from mathcoach.agents.solving_verification import (
    SolvingVerificationAgent,
    SolvingVerificationInput,
)
from mathcoach.agents.trace import AgentRunResult, AgentRunTrace
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.schemas.verification import (
    AnswerItem,
    Assertion,
    SolvingVerification,
    VerificationResult,
)


def _build_input() -> SolvingVerificationInput:
    return SolvingVerificationInput(
        analysis=ProblemUnderstanding(
            problem_type="x",
            knowledge_points=[],
            conditions={},
            goal="x",
            difficulty="简单",
        ),
        plan=SolvingPlan(method="x", steps=[]),
    )


def _make_agent_with_output(output: SolvingVerification) -> SolvingVerificationAgent:
    """Stub the OpenRouter client and parent run_with_trace."""
    with patch("mathcoach.agents.base.OpenRouterClient"):
        agent = SolvingVerificationAgent()
    fake_trace = AgentRunTrace(
        agent_name="SolvingVerificationAgent",
        system_prompt="",
        user_prompt="",
    )

    def fake_super(self, input_data):  # noqa: ARG001 - signature parity
        return AgentRunResult(output=output, trace=fake_trace)

    # Patch the parent class's method via the imported module.
    patcher = patch.object(base_mod.BaseAgent, "run_with_trace", fake_super)
    patcher.start()
    return agent


# ---------------------------------------------------------------------------
# 1. assertions=[] → cap LLM confidence
# ---------------------------------------------------------------------------


def test_empty_assertions_caps_llm_confidence() -> None:
    output = SolvingVerification(
        solution_steps=["x"],
        answer=[AnswerItem(label="x", latex="$x$")],
        verification=VerificationResult(method="LLM 自检", status="passed", confidence=1.0),
        assertions=None,
    )
    agent = _make_agent_with_output(output)
    try:
        result = agent.run_with_trace(_build_input())
    finally:
        patch.stopall()

    assert result.output.verification.confidence == 0.6
    assert "no assertions supplied" in (result.output.verification.detail or "").lower()


def test_empty_assertions_keeps_low_confidence_unchanged() -> None:
    output = SolvingVerification(
        solution_steps=[],
        answer=[],
        verification=VerificationResult(method="x", status="passed", confidence=0.3),
        assertions=[],
    )
    agent = _make_agent_with_output(output)
    try:
        result = agent.run_with_trace(_build_input())
    finally:
        patch.stopall()

    # 0.3 is already below cap; should stay as-is.
    assert result.output.verification.confidence == 0.3


# ---------------------------------------------------------------------------
# 2. all-pass aggregation
# ---------------------------------------------------------------------------


def test_all_pass_aggregates_to_passed() -> None:
    output = SolvingVerification(
        solution_steps=[],
        answer=[],
        verification=VerificationResult(method="LLM", status="passed", confidence=1.0),
        assertions=[
            Assertion(expr="1 + 1", expected=2, description="trivial 1"),
            Assertion(expr="2 * 3", expected=6, description="trivial 2"),
        ],
    )
    agent = _make_agent_with_output(output)
    try:
        result = agent.run_with_trace(_build_input())
    finally:
        patch.stopall()

    v = result.output.verification
    assert v.status == "passed"
    assert v.confidence >= 0.94  # min of per-item confidences (post-recalibration)
    assert "2/2 passed" in (v.detail or "")
    # Trace got one tool step appended
    assert any(step.kind == "tool" for step in result.trace.steps)
    tool_step = next(step for step in result.trace.steps if step.kind == "tool")
    assert tool_step.model == "sympy"


# ---------------------------------------------------------------------------
# 3. mixed pass/fail aggregates to failed with locator
# ---------------------------------------------------------------------------


def test_mixed_pass_fail_locates_failure() -> None:
    output = SolvingVerification(
        solution_steps=[],
        answer=[],
        verification=VerificationResult(method="LLM", status="passed", confidence=1.0),
        assertions=[
            Assertion(expr="1 + 1", expected=2, description="step A"),
            Assertion(expr="2 * 3", expected=10, description="step B (wrong)"),
            Assertion(expr="3 + 4", expected=7, description="step C"),
        ],
    )
    agent = _make_agent_with_output(output)
    try:
        result = agent.run_with_trace(_build_input())
    finally:
        patch.stopall()

    v = result.output.verification
    assert v.status == "failed"
    assert v.confidence <= 0.1
    assert "1/3 failed" in (v.detail or "")
    # The failed item's description should appear in the aggregated detail
    assert "step B" in (v.detail or "")


# ---------------------------------------------------------------------------
# 4. error aggregation (parse error)
# ---------------------------------------------------------------------------


def test_mixed_tier_aggregation_stays_high() -> None:
    """A solution that adds a sampling check on top of symbolic checks should
    NOT see its confidence drop below 0.94 — verifying the recalibration
    removed the reverse incentive for thorough verification.
    """
    output = SolvingVerification(
        solution_steps=[],
        answer=[],
        verification=VerificationResult(method="LLM", status="passed", confidence=1.0),
        assertions=[
            # symbolic-tier item
            Assertion(expr="cos(5*pi/6)", expected="-sqrt(3)/2", description="answer"),
            # sampling-tier item (lowest base confidence)
            Assertion(
                expr="sin(x)**2 + cos(x)**2",
                expected=1,
                free_vars={"x": [0.1, 1.0]},
                description="pythagorean identity",
            ),
        ],
    )
    agent = _make_agent_with_output(output)
    try:
        result = agent.run_with_trace(_build_input())
    finally:
        patch.stopall()

    v = result.output.verification
    assert v.status == "passed"
    # min(0.98 symbolic, 0.94 sampling) = 0.94 — still high, NOT punished.
    assert v.confidence >= 0.94


def test_error_assertion_aggregates_to_error_when_no_failures() -> None:
    output = SolvingVerification(
        solution_steps=[],
        answer=[],
        verification=VerificationResult(method="LLM", status="passed", confidence=1.0),
        assertions=[
            Assertion(expr="1 + 1", expected=2),
            Assertion(expr="x ** ?", expected=0, description="parse error here"),
        ],
    )
    agent = _make_agent_with_output(output)
    try:
        result = agent.run_with_trace(_build_input())
    finally:
        patch.stopall()

    v = result.output.verification
    assert v.status == "error"
    assert v.confidence == 0.30
    assert "parse error here" in (v.detail or "")
