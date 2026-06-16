"""Unit tests for the SymPy verifier tool.

These tests do not call any LLM. They construct VerifiableArtifact instances
directly and assert that the tool's status / confidence agree with reality.
"""

from __future__ import annotations

import pytest

from mathcoach.schemas.verification import VerifiableArtifact
from mathcoach.tools.sympy_verifier import verify

pytest.importorskip("sympy")


# ---------------------------------------------------------------------------
# equation_roots
# ---------------------------------------------------------------------------


def test_equation_roots_passed():
    artifact = VerifiableArtifact(
        kind="equation_roots",
        payload={"expr": "x**2 - 5*x + 6", "var": "x", "roots": [2, 3]},
    )
    result = verify(artifact)
    assert result.status == "passed"
    assert result.confidence >= 0.9


def test_equation_roots_failed_when_roots_wrong():
    artifact = VerifiableArtifact(
        kind="equation_roots",
        payload={"expr": "x**2 - 5*x + 6", "var": "x", "roots": [1, 4]},
    )
    result = verify(artifact)
    assert result.status == "failed"
    assert result.confidence <= 0.2
    assert "residual" in (result.detail or "")


def test_equation_roots_accepts_symbolic_root():
    # x**2 - 2 = 0 has roots ±sqrt(2); pass them as SymPy strings.
    artifact = VerifiableArtifact(
        kind="equation_roots",
        payload={"expr": "x**2 - 2", "var": "x", "roots": ["sqrt(2)", "-sqrt(2)"]},
    )
    assert verify(artifact).status == "passed"


# ---------------------------------------------------------------------------
# expression_value
# ---------------------------------------------------------------------------


def test_expression_value_passed_numeric():
    artifact = VerifiableArtifact(
        kind="expression_value",
        payload={
            "expr": "x**3 - 3*x + 1",
            "substitute": {"x": "-1"},
            "expected": 3,
        },
    )
    assert verify(artifact).status == "passed"


def test_expression_value_passed_symbolic():
    artifact = VerifiableArtifact(
        kind="expression_value",
        payload={
            "expr": "cos(B)",
            "substitute": {"B": "5*pi/6"},
            "expected": "-sqrt(3)/2",
        },
    )
    assert verify(artifact).status == "passed"


def test_expression_value_failed_when_wrong():
    artifact = VerifiableArtifact(
        kind="expression_value",
        payload={
            "expr": "x**2",
            "substitute": {"x": "3"},
            "expected": 10,
        },
    )
    result = verify(artifact)
    assert result.status == "failed"
    assert "diff" in (result.detail or "")


def test_expression_value_missing_expected_raises_error_status():
    artifact = VerifiableArtifact(
        kind="expression_value",
        payload={"expr": "x**2", "substitute": {"x": "1"}},
    )
    assert verify(artifact).status == "error"


# ---------------------------------------------------------------------------
# Error handling and stubs
# ---------------------------------------------------------------------------


def test_parse_error_returns_error_status():
    artifact = VerifiableArtifact(
        kind="equation_roots",
        payload={"expr": "x ** ?", "var": "x", "roots": [0]},
    )
    result = verify(artifact)
    assert result.status == "error"
    assert result.confidence <= 0.4


def test_none_kind_marks_not_verifiable():
    artifact = VerifiableArtifact(kind="none", payload={"reason": "open proof"})
    result = verify(artifact)
    assert result.status == "not_verifiable"
    # Confidence is intentionally neutral, not high.
    assert 0.4 <= result.confidence <= 0.6


def test_unimplemented_kind_returns_skipped():
    artifact = VerifiableArtifact(kind="function_extrema", payload={})
    result = verify(artifact)
    assert result.status == "skipped"
    assert "未实现" in result.method
