"""Tests for eval runner with mocked LLM client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mathcoach.eval import runner
from mathcoach.eval.types import EvalProblem
from mathcoach.llm.chat_result import ChatCompletionResult


def _problem(
    pid: str = "X01",
    sympy: str = "1",
    label: str = "x",
) -> EvalProblem:
    return EvalProblem(
        id=pid,
        group="A",
        type="test",
        knowledge_points=["t"],
        difficulty="easy",
        expected_verifier="should_pass",
        question="解 $x=1$",
        truth={
            "answer": [
                {"label": label, "latex": "$1$", "sympy": sympy, "numeric": 1.0, "unit": None}
            ]
        },
    )


_FAKE_UNDERSTANDING = {
    "problem_type": "解方程",
    "knowledge_points": ["代数"],
    "conditions": {"equation": "x=1"},
    "goal": "求 x",
    "difficulty": "简单",
}

_FAKE_PLAN = {
    "method": "代入",
    "steps": ["代入"],
    "key_steps": ["代入"],
    "warnings": [],
}

_FAKE_VERIFICATION = {
    "solution_steps": ["x=1"],
    "answer": [
        {"label": "x", "latex": "$1$", "sympy": "1", "numeric": 1.0, "unit": None}
    ],
    "verification": {
        "method": "trivial",
        "status": "passed",
        "confidence": 0.98,
        "detail": "1/1 passed",
    },
    "assertions": [
        {"expr": "1", "expected": 1, "description": "trivial"}
    ],
}


class _FakeClient:
    """Returns a canned JSON payload based on the system prompt's agent name."""

    def __init__(self, *, fail_on: str | None = None):
        self._fail_on = fail_on

    def chat_with_details(self, messages, *, model=None, **kwargs):
        system = messages[0]["content"]
        if self._fail_on and self._fail_on in system:
            return ChatCompletionResult(
                content="not json at all",
                model="fake",
            )
        if "problem understanding" in system:
            payload = _FAKE_UNDERSTANDING
        elif "solving strategist" in system:
            payload = _FAKE_PLAN
        else:
            payload = _FAKE_VERIFICATION
        return ChatCompletionResult(
            content=json.dumps(payload, ensure_ascii=False),
            model="fake-model",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def _patch_clients(monkeypatch, client_factory):
    """Replace agents' OpenRouterClient with the fake one."""
    from mathcoach.agents import (
        problem_understanding,
        solving_planning,
        solving_verification,
    )

    for module in (problem_understanding, solving_planning, solving_verification):
        # BaseAgent calls OpenRouterClient() if llm_client is None; we patch
        # the symbol that base.py imports.
        pass

    from mathcoach.agents import base as base_module

    monkeypatch.setattr(base_module, "OpenRouterClient", client_factory)


def test_runner_all_pass(monkeypatch, tmp_path: Path):
    _patch_clients(monkeypatch, lambda: _FakeClient())

    problems = [_problem(pid="A01"), _problem(pid="A02")]
    rows = runner.run(problems, traces_dir=tmp_path / "traces")

    assert len(rows) == 2
    for row in rows:
        assert row.pipeline_status == "ok"
        assert row.correct is True
        assert row.grader_status == "ok"
        assert row.verifier_status == "passed"
    # Trace files were written
    assert (tmp_path / "traces" / "A01.log").exists()
    assert (tmp_path / "traces" / "A02.log").exists()


def test_runner_isolates_single_failure(monkeypatch, tmp_path: Path):
    """A single agent failure must not break the batch."""

    state = {"first_seen": False}

    class _FlakyClient:
        def chat_with_details(self, messages, **kwargs):
            system = messages[0]["content"]
            if not state["first_seen"]:
                state["first_seen"] = True
                # First call always succeeds (Understanding)
                return ChatCompletionResult(
                    content=json.dumps(_FAKE_UNDERSTANDING),
                    model="fake",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            # Subsequent calls return garbage to force agent failure
            return ChatCompletionResult(content="garbage", model="fake")

    _patch_clients(monkeypatch, lambda: _FlakyClient())

    problems = [_problem(pid="A01")]
    rows = runner.run(problems, traces_dir=tmp_path / "traces")
    assert len(rows) == 1
    assert rows[0].pipeline_status == "failed"
    assert rows[0].correct is None
    assert rows[0].grader_status == "error"


def test_runner_grader_skip_on_no_sympy(monkeypatch, tmp_path: Path):
    """When truth.sympy is None, grader returns None and runner records no_sympy."""
    _patch_clients(monkeypatch, lambda: _FakeClient())

    p = EvalProblem(
        id="C01",
        group="C",
        type="proof",
        knowledge_points=["t"],
        difficulty="medium",
        expected_verifier="unverifiable",
        question="证明...",
        truth={
            "answer": [
                {"label": "结论", "latex": "$\\square$", "sympy": None, "numeric": None, "unit": None}
            ]
        },
    )
    rows = runner.run([p], traces_dir=tmp_path / "traces")
    assert len(rows) == 1
    # Pipeline returns AnswerItem(label="x", sympy="1") but truth.label="结论"
    # → label mismatch falls back to positional pairing → truth.sympy=None → skip
    assert rows[0].correct is None
    assert rows[0].grader_status == "no_sympy"
