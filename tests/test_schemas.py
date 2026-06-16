"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from mathcoach.schemas.inputs import UserQuery
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.schemas.verification import (
    AnswerItem,
    Assertion,
    SolvingVerification,
    VerificationResult,
)


def test_user_query_requires_question() -> None:
    with pytest.raises(ValidationError):
        UserQuery(question="")


def test_problem_understanding_valid_payload() -> None:
    payload = ProblemUnderstanding(
        problem_type="函数最值问题",
        knowledge_points=["导数", "驻点"],
        conditions={"function": "x^3-3x+1", "interval": "[-2,2]"},
        goal="求最大值和最小值",
        difficulty="中等",
    )
    assert payload.problem_type == "函数最值问题"
    assert payload.difficulty == "中等"


def test_problem_understanding_rejects_invalid_difficulty() -> None:
    with pytest.raises(ValidationError):
        ProblemUnderstanding(
            problem_type="函数最值问题",
            knowledge_points=[],
            conditions={},
            goal="求最大值和最小值",
            difficulty="very hard",  # type: ignore[arg-type]
        )


def test_solving_plan_defaults() -> None:
    plan = SolvingPlan(
        method="导数法",
        steps=["求导", "求驻点"],
    )
    assert plan.alternative_method is None
    assert plan.key_steps == []
    assert plan.warnings == []


def test_solving_plan_full_payload() -> None:
    plan = SolvingPlan(
        method="导数法",
        steps=["求导", "求驻点", "比较端点与驻点"],
        alternative_method="图像法",
        key_steps=["比较端点与驻点"],
        warnings=["不要忘记比较端点"],
    )
    assert plan.method == "导数法"
    assert len(plan.steps) == 3
    assert plan.warnings == ["不要忘记比较端点"]


# ---------------------------------------------------------------------------
# Assertion / AnswerItem / SolvingVerification (post-refactor)
# ---------------------------------------------------------------------------


def test_assertion_minimal() -> None:
    a = Assertion(expr="x**2 - 4", expected=0)
    assert a.expected == 0
    assert a.description is None
    assert a.free_vars is None
    assert a.tolerance is None


def test_assertion_with_free_vars() -> None:
    a = Assertion(
        expr="sin(x)**2 + cos(x)**2",
        expected=1,
        description="毕达哥拉斯恒等式",
        free_vars={"x": [-3.14, 3.14]},
        tolerance=1e-9,
    )
    assert a.free_vars == {"x": [-3.14, 3.14]}
    assert a.description == "毕达哥拉斯恒等式"


def test_assertion_expected_accepts_string_number_list_bool() -> None:
    Assertion(expr="cos(B)", expected="-sqrt(3)/2")
    Assertion(expr="2+2", expected=4)
    Assertion(expr="solve(x**2-5*x+6, x)", expected=[2, 3])
    Assertion(expr="3 > 0", expected=True)


def test_answer_item_full_payload() -> None:
    item = AnswerItem(
        label="角B",
        latex="$\\dfrac{5\\pi}{6}$",
        sympy="5*pi/6",
        numeric=2.6179938779914944,
        unit="rad",
    )
    assert item.label == "角B"
    assert item.sympy == "5*pi/6"
    assert item.unit == "rad"


def test_answer_item_only_required_fields() -> None:
    item = AnswerItem(label="x", latex="$3$")
    assert item.sympy is None
    assert item.numeric is None
    assert item.unit is None


def test_solving_verification_uses_list_answer_and_assertions() -> None:
    sv = SolvingVerification(
        solution_steps=["step 1"],
        answer=[AnswerItem(label="最大值", latex="$3$", sympy="3", numeric=3.0)],
        verification=VerificationResult(method="LLM", status="passed", confidence=0.9),
        assertions=[Assertion(expr="x**2", expected=0)],
    )
    assert len(sv.answer) == 1
    assert sv.answer[0].label == "最大值"
    assert sv.assertions is not None and len(sv.assertions) == 1


def test_solving_verification_assertions_default_none() -> None:
    sv = SolvingVerification(
        solution_steps=[],
        answer=[],
        verification=VerificationResult(method="x", status="skipped", confidence=0.5),
    )
    assert sv.assertions is None
    assert sv.answer == []
