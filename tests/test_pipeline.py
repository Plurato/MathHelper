from __future__ import annotations

import pytest

from mathcoach.agents.trace import AgentRunResult, AgentRunTrace, LLMStepTrace
from mathcoach.pipeline import (
    PipelineAgents,
    PipelineExecutionError,
    run_mathcoach_pipeline,
)
from mathcoach.schemas.inputs import UserQuery
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.schemas.teaching import TeachingExplanation
from mathcoach.schemas.verification import (
    AnswerItem,
    SolvingVerification,
    VerificationResult,
)


def test_pipeline_runs_four_agents_and_shapes_response():
    agents = _fake_agents()
    query = UserQuery(
        question="解方程 x^2 - 5x + 6 = 0。",
        student_level="高中",
        explanation_style="详细版",
    )

    result = run_mathcoach_pipeline(query, agents=agents)

    assert result.question == query.question
    assert result.student_level == "高中"
    assert result.explanation_style == "详细版"
    assert result.analysis.problem_type == "一元二次方程"
    assert result.plan.method == "因式分解"
    assert result.verification.verification.status == "passed"
    assert result.teaching.explanation == "先因式分解，再代入验证。"
    assert [stage.key for stage in result.stages] == [
        "understanding",
        "planning",
        "verification",
        "teaching",
    ]
    assert all(stage.status == "succeeded" for stage in result.stages)
    assert result.usage.prompt_tokens == 40
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 60
    assert [entry.agent_name for entry in result.trace] == [
        "ProblemUnderstandingAgent",
        "SolvingPlanningAgent",
        "SolvingVerificationAgent",
        "TeachingExplanationAgent",
    ]
    assert agents.planning.seen_input.analysis == result.analysis
    assert agents.verification.seen_input.plan == result.plan
    assert agents.teaching.seen_input.verification == result.verification


def test_pipeline_failure_includes_stage_and_partial_stages():
    with pytest.raises(PipelineExecutionError) as exc_info:
        run_mathcoach_pipeline(
            UserQuery(question="1+1"),
            agents=_fake_agents(fail_at="planning"),
        )

    err = exc_info.value
    assert err.stage_key == "planning"
    assert "planner exploded" in err.message
    assert len(err.stages) == 2
    assert err.stages[0].status == "succeeded"
    assert err.stages[1].status == "failed"
    assert err.stages[1].agent_name == "SolvingPlanningAgent"


def _fake_agents(fail_at: str | None = None) -> PipelineAgents:
    return PipelineAgents(
        understanding=_FakeAgent(
            "ProblemUnderstandingAgent",
            ProblemUnderstanding(
                problem_type="一元二次方程",
                knowledge_points=["因式分解"],
                conditions={"方程": "x^2 - 5x + 6 = 0"},
                goal="求 x",
                difficulty="简单",
            ),
            fail=fail_at == "understanding",
        ),
        planning=_FakeAgent(
            "SolvingPlanningAgent",
            SolvingPlan(
                method="因式分解",
                steps=["化为 (x-2)(x-3)=0", "得到 x=2 或 x=3"],
                key_steps=["因式分解"],
                warnings=["不要漏根"],
            ),
            fail=fail_at == "planning",
            error="planner exploded",
        ),
        verification=_FakeAgent(
            "SolvingVerificationAgent",
            SolvingVerification(
                solution_steps=["(x-2)(x-3)=0", "x=2 或 x=3"],
                answer=[
                    AnswerItem(label="x_1", latex="$2$", sympy="2", numeric=2),
                    AnswerItem(label="x_2", latex="$3$", sympy="3", numeric=3),
                ],
                verification=VerificationResult(
                    method="SymPy", status="passed", confidence=0.95
                ),
                assertions=[],
            ),
            fail=fail_at == "verification",
        ),
        teaching=_FakeAgent(
            "TeachingExplanationAgent",
            TeachingExplanation(
                explanation="先因式分解，再代入验证。",
                key_points=["零因子性质"],
                common_mistakes=["漏写一个根"],
                practice_questions=["解方程 x^2-4x+3=0。"],
                learning_advice="多练习因式分解。",
            ),
            fail=fail_at == "teaching",
        ),
    )


class _FakeAgent:
    def __init__(
        self,
        name: str,
        output: object,
        *,
        fail: bool = False,
        error: str = "agent exploded",
    ) -> None:
        self.name = name
        self.output = output
        self.fail = fail
        self.error = error
        self.seen_input = None

    def run_with_trace(self, input_data: object):
        self.seen_input = input_data
        if self.fail:
            raise RuntimeError(self.error)
        return AgentRunResult(
            output=self.output,
            trace=AgentRunTrace(
                agent_name=self.name,
                system_prompt=f"{self.name} system",
                user_prompt=f"{self.name} user",
                steps=[
                    LLMStepTrace(
                        attempt=1,
                        raw_response='{"ok": true}',
                        reasoning=None,
                        parsed_payload={"ok": True},
                        validation_error=None,
                        status="success",
                        model="fake-model",
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    )
                ],
            ),
        )
