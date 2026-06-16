"""Tests for trace printing utilities."""

from mathcoach.agents.trace import AgentRunResult, AgentRunTrace, LLMStepTrace
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.utils.trace_printer import print_agent_trace


def test_print_agent_trace_renders_sections(capsys) -> None:
    output = ProblemUnderstanding(
        problem_type="函数最值问题",
        knowledge_points=["导数"],
        conditions={"function": "x^3-3x+1"},
        goal="求最大值和最小值",
        difficulty="中等",
    )
    trace = AgentRunTrace(
        agent_name="ProblemUnderstandingAgent",
        system_prompt="You are a math expert.",
        user_prompt="Question:\n求最值",
        steps=[
            LLMStepTrace(
                attempt=1,
                raw_response='{"problem_type":"函数最值问题","goal":"求最值"}',
                reasoning="First identify the problem type.",
                parsed_payload={"problem_type": "函数最值问题", "goal": "求最值"},
                validation_error=None,
                status="success",
                model="openai/gpt-4o-mini",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            )
        ],
    )
    result = AgentRunResult(output=output, trace=trace)

    print_agent_trace(result)
    captured = capsys.readouterr().out

    assert "ProblemUnderstandingAgent" in captured
    assert "[System Prompt]" in captured
    assert "[User Prompt]" in captured
    assert "[Reasoning]" in captured
    assert "First identify the problem type." in captured
    assert "[Raw Response]" in captured
    assert "[Parsed JSON]" in captured
    assert "--- Validated Output ---" in captured
    assert "Tokens:" in captured


def test_print_agent_trace_can_hide_prompts(capsys) -> None:
    output = ProblemUnderstanding(
        problem_type="函数最值问题",
        knowledge_points=[],
        conditions={},
        goal="求最值",
        difficulty="简单",
    )
    trace = AgentRunTrace(
        agent_name="ProblemUnderstandingAgent",
        system_prompt="hidden system",
        user_prompt="hidden user",
        steps=[
            LLMStepTrace(
                attempt=1,
                raw_response="{}",
                reasoning=None,
                parsed_payload={},
                validation_error=None,
                status="success",
            )
        ],
    )
    result = AgentRunResult(output=output, trace=trace)

    print_agent_trace(result, show_prompts=False)
    captured = capsys.readouterr().out

    assert "[System Prompt]" not in captured
    assert "[User Prompt]" not in captured
    assert "hidden system" not in captured


def test_print_agent_trace_distinguishes_tool_step(capsys) -> None:
    """Tool steps should print under '--- Tool Call ... ---' not 'LLM Call'."""
    output = ProblemUnderstanding(
        problem_type="函数最值问题",
        knowledge_points=[],
        conditions={},
        goal="求",
        difficulty="简单",
    )
    trace = AgentRunTrace(
        agent_name="SolvingVerificationAgent",
        system_prompt="...",
        user_prompt="...",
        steps=[
            LLMStepTrace(
                attempt=1,
                raw_response="{}",
                reasoning=None,
                parsed_payload={"a": 1},
                validation_error=None,
                status="success",
                model="glm-5.1",
                kind="llm",
            ),
            LLMStepTrace(
                attempt=2,
                raw_response='[{"expr":"x**2","expected":0}]',
                reasoning=None,
                parsed_payload={
                    "tool": "sympy_verifier",
                    "n_total": 2,
                    "n_passed": 1,
                    "n_failed": 1,
                    "n_error": 0,
                    "n_skipped": 0,
                    "items": [
                        {
                            "description": "step A",
                            "expr": "x**2",
                            "expected": 0,
                            "result": {
                                "status": "passed",
                                "method": "SymPy 符号验证",
                                "confidence": 0.95,
                                "detail": "ok",
                            },
                        },
                        {
                            "description": "step B",
                            "expr": "x**3",
                            "expected": 1,
                            "result": {
                                "status": "failed",
                                "method": "SymPy 验证",
                                "confidence": 0.05,
                                "detail": "diff != 0",
                            },
                        },
                    ],
                    "aggregated": {
                        "status": "failed",
                        "confidence": 0.05,
                        "method": "SymPy 多项断言验证（2 项）",
                        "detail": "1/2 failed",
                    },
                },
                validation_error=None,
                status="failed",
                model="sympy",
                kind="tool",
            ),
        ],
    )
    result = AgentRunResult(output=output, trace=trace)

    print_agent_trace(result)
    captured = capsys.readouterr().out

    # The LLM step is rendered with the LLM header
    assert "--- LLM Call (attempt 1" in captured
    # The tool step is rendered with the Tool header (and NOT 'LLM Call' for it)
    assert "--- Tool Call (sympy_verifier" in captured
    # Per-item rendering surfaces both items
    assert "step A" in captured
    assert "step B" in captured
    # Failure detail appears
    assert "diff != 0" in captured
    # Aggregated summary appears
    assert "1/2 failed" in captured
