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


def test_pipeline_retries_planning_until_verification_passes():
    understanding = _FakeAgent(
        "ProblemUnderstandingAgent",
        ProblemUnderstanding(
            problem_type="一元二次方程",
            knowledge_points=["因式分解"],
            conditions={"方程": "x^2 - 5x + 6 = 0"},
            goal="求 x",
            difficulty="简单",
        ),
    )
    plan_a = SolvingPlan(method="错误方法", steps=["错误步骤"])
    plan_b = SolvingPlan(method="因式分解", steps=["(x-2)(x-3)=0", "x=2 或 x=3"])
    planning = _RecordingPlanningAgent([plan_a, plan_b])
    verification = _SequencedVerificationAgent(
        [
            _verification_output("failed", with_failed_item=True),
            _verification_output("passed"),
        ]
    )
    teaching = _FakeAgent(
        "TeachingExplanationAgent",
        TeachingExplanation(
            explanation="讲解",
            key_points=["k"],
            common_mistakes=["m"],
            practice_questions=["q"],
            learning_advice="advice",
        ),
    )
    agents = PipelineAgents(understanding, planning, verification, teaching)

    result = run_mathcoach_pipeline(UserQuery(question="解方程"), agents=agents)

    assert result.planning_attempts == 2
    assert result.verification.verification.status == "passed"
    assert result.plan.method == "因式分解"

    assert len(planning.seen_inputs) == 2
    assert planning.seen_inputs[0].feedback is None
    feedback = planning.seen_inputs[1].feedback
    assert feedback is not None
    assert feedback.previous_method == "错误方法"
    assert feedback.verification_status == "failed"
    assert feedback.failed_assertions
    assert feedback.failed_assertions[0].expr == "diff(x**2, x)"

    planning_stages = [s for s in result.stages if s.key == "planning"]
    verification_stages = [s for s in result.stages if s.key == "verification"]
    assert [s.attempt for s in planning_stages] == [1, 2]
    assert [s.attempt for s in verification_stages] == [1, 2]
    assert [stage.key for stage in result.stages] == [
        "understanding",
        "planning",
        "verification",
        "planning",
        "verification",
        "teaching",
    ]


def test_pipeline_keeps_last_result_when_retries_exhausted():
    understanding = _FakeAgent(
        "ProblemUnderstandingAgent",
        ProblemUnderstanding(
            problem_type="t",
            knowledge_points=["k"],
            conditions={},
            goal="g",
            difficulty="困难",
        ),
    )
    planning = _RecordingPlanningAgent(
        [SolvingPlan(method=f"方法{i}", steps=["s"]) for i in range(5)]
    )
    verification = _SequencedVerificationAgent(
        [_verification_output("failed", with_failed_item=True) for _ in range(5)]
    )
    teaching = _FakeAgent(
        "TeachingExplanationAgent",
        TeachingExplanation(
            explanation="讲解",
            key_points=["k"],
            common_mistakes=["m"],
            practice_questions=["q"],
            learning_advice="advice",
        ),
    )
    agents = PipelineAgents(understanding, planning, verification, teaching)

    result = run_mathcoach_pipeline(UserQuery(question="解方程"), agents=agents)

    assert result.planning_attempts == 3
    assert len(planning.seen_inputs) == 3
    assert result.verification.verification.status == "failed"

    teaching_stages = [s for s in result.stages if s.key == "teaching"]
    assert len(teaching_stages) == 1
    assert teaching_stages[0].status == "succeeded"
    assert len([s for s in result.stages if s.key == "planning"]) == 3


def test_pipeline_emits_progress_events_in_order():
    events: list[dict] = []
    run_mathcoach_pipeline(
        UserQuery(question="解方程"),
        agents=_fake_agents(),
        on_event=events.append,
    )

    types = [e["type"] for e in events]
    assert types == [
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
        "done",
    ]
    started_keys = [e["key"] for e in events if e["type"] == "stage_started"]
    assert started_keys == ["understanding", "planning", "verification", "teaching"]
    completed = next(e for e in events if e["type"] == "stage_completed")
    assert completed["output"]["problem_type"] == "一元二次方程"
    done = events[-1]
    assert done["result"]["verification"]["verification"]["status"] == "passed"


def test_pipeline_emits_retry_event_on_failed_verification():
    understanding = _FakeAgent(
        "ProblemUnderstandingAgent",
        ProblemUnderstanding(
            problem_type="t",
            knowledge_points=["k"],
            conditions={},
            goal="g",
            difficulty="中等",
        ),
    )
    planning = _RecordingPlanningAgent(
        [
            SolvingPlan(method="错误方法", steps=["s"]),
            SolvingPlan(method="正确方法", steps=["s"]),
        ]
    )
    verification = _SequencedVerificationAgent(
        [
            _verification_output("failed", with_failed_item=True),
            _verification_output("passed"),
        ]
    )
    teaching = _FakeAgent(
        "TeachingExplanationAgent",
        TeachingExplanation(
            explanation="讲解",
            key_points=["k"],
            common_mistakes=["m"],
            practice_questions=["q"],
            learning_advice="advice",
        ),
    )
    events: list[dict] = []
    run_mathcoach_pipeline(
        UserQuery(question="解方程"),
        agents=PipelineAgents(understanding, planning, verification, teaching),
        on_event=events.append,
    )

    retries = [e for e in events if e["type"] == "retry"]
    assert len(retries) == 1
    assert retries[0]["attempt"] == 2
    assert retries[0]["verification_status"] == "failed"
    assert retries[0]["failed_assertions"][0]["expr"] == "diff(x**2, x)"
    second_planning = [
        e
        for e in events
        if e["type"] == "stage_started" and e["key"] == "planning"
    ]
    assert [e["attempt"] for e in second_planning] == [1, 2]


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
            trace=_simple_trace(self.name),
        )


class _RecordingPlanningAgent:
    """Planning agent that returns queued plans and records each input."""

    name = "SolvingPlanningAgent"

    def __init__(self, outputs: list[SolvingPlan]) -> None:
        self._outputs = list(outputs)
        self.seen_inputs: list[object] = []

    def run_with_trace(self, input_data: object):
        self.seen_inputs.append(input_data)
        output = self._outputs[min(len(self.seen_inputs) - 1, len(self._outputs) - 1)]
        return AgentRunResult(output=output, trace=_simple_trace(self.name))


class _SequencedVerificationAgent:
    """Verification agent that returns queued (output, trace) results in order."""

    name = "SolvingVerificationAgent"

    def __init__(self, results: list[AgentRunResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def run_with_trace(self, input_data: object):
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return result


def _simple_trace(name: str) -> AgentRunTrace:
    return AgentRunTrace(
        agent_name=name,
        system_prompt=f"{name} system",
        user_prompt=f"{name} user",
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
    )


def _verification_output(
    status: str, *, with_failed_item: bool = False
) -> AgentRunResult:
    """Build a verification AgentRunResult with the given verdict.

    When ``with_failed_item`` is set, a SymPy tool step carrying a failed
    assertion item is attached so the pipeline can extract rich feedback.
    """
    output = SolvingVerification(
        solution_steps=["step-1", "step-2"],
        answer=[AnswerItem(label="x", latex="$2$", sympy="2", numeric=2)],
        verification=VerificationResult(
            method="SymPy",
            status=status,
            confidence=0.9 if status == "passed" else 0.05,
            detail=None if status == "passed" else "1/1 failed",
        ),
        assertions=[],
    )
    steps = [_simple_trace("SolvingVerificationAgent").steps[0]]
    if with_failed_item:
        steps.append(
            LLMStepTrace(
                attempt=2,
                raw_response="[]",
                reasoning=None,
                parsed_payload={
                    "tool": "sympy_verifier",
                    "items": [
                        {
                            "description": "求导",
                            "expr": "diff(x**2, x)",
                            "expected": "2*x",
                            "result": {
                                "status": "failed",
                                "detail": "expected 2*x, got 2",
                            },
                        }
                    ],
                },
                validation_error=None,
                status="failed",
                model="sympy",
                kind="tool",
            )
        )
    return AgentRunResult(
        output=output,
        trace=AgentRunTrace(
            agent_name="SolvingVerificationAgent",
            system_prompt="v system",
            user_prompt="v user",
            steps=steps,
        ),
    )
