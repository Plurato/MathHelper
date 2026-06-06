"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from mathcoach.schemas.inputs import UserQuery
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding


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
