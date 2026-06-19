"""Tests for the assertion-based SymPy verifier.

These tests construct `Assertion` instances directly and assert that the
verifier's status / confidence agree with reality. No LLM calls.
"""

from __future__ import annotations

import pytest

from mathcoach.schemas.verification import Assertion
from mathcoach.tools.sympy_verifier import verify

pytest.importorskip("sympy")


# ---------------------------------------------------------------------------
# Symbolic & numeric scalar paths
# ---------------------------------------------------------------------------


def test_numeric_passed_when_evalf_matches() -> None:
    a = Assertion(expr="(x**3 - 3*x + 1).subs(x, -1)", expected=3)
    r = verify(a)
    assert r.status == "passed"
    assert r.confidence >= 0.96  # symbolic 0.98 or numerical 0.96


def test_symbolic_passed_simplifies_to_zero() -> None:
    # Both sides are symbolic; simplify reduces (cos(5pi/6) - (-sqrt(3)/2)) to 0.
    a = Assertion(expr="cos(5*pi/6)", expected="-sqrt(3)/2")
    r = verify(a)
    assert r.status == "passed"
    assert r.confidence == 0.98  # symbolic tier


def test_failed_when_value_differs() -> None:
    a = Assertion(expr="2 + 2", expected=5)
    r = verify(a)
    assert r.status == "failed"
    assert r.confidence <= 0.1


# ---------------------------------------------------------------------------
# List / set comparison
# ---------------------------------------------------------------------------


def test_list_expected_passes_regardless_of_order() -> None:
    # solve returns [2, 3]; expected lists [3, 2]. Set-style match passes.
    a = Assertion(
        expr="solve(x**2 - 5*x + 6, x)",
        expected=[3, 2],
        description="roots set",
    )
    r = verify(a)
    assert r.status == "passed"


def test_list_expected_fails_when_member_missing() -> None:
    a = Assertion(
        expr="solve(x**2 - 5*x + 6, x)",
        expected=[2, 4],
    )
    r = verify(a)
    assert r.status == "failed"


def test_list_length_mismatch_fails() -> None:
    a = Assertion(
        expr="solve(x**2 - 5*x + 6, x)",
        expected=[2, 3, 4],
    )
    r = verify(a)
    assert r.status == "failed"


def test_list_with_symbolic_member_passes() -> None:
    a = Assertion(
        expr="solve(x**2 - 2, x)",
        expected=["sqrt(2)", "-sqrt(2)"],
    )
    r = verify(a)
    assert r.status == "passed"


def test_linear_system_solve_dict_matches_expected_values() -> None:
    """SymPy solve([...], [x, y]) returns a dict for determined systems.

    The verifier should compare the solved values against an expected answer
    list instead of treating dict keys as the actual values.
    """
    a = Assertion(
        expr="solve([y - 10*x, y - x - 288], [x, y])",
        expected=[32, 320],
        description="一次方程组求解",
    )
    r = verify(a)
    assert r.status == "passed"


def test_linear_system_solve_dict_true_matches_expected_mapping() -> None:
    """Regression for verifier error on solve(..., dict=True).

    SymPy returns list[dict[Symbol, value]], while LLM output often expresses
    expected mappings as JSON dicts inside the expected list.
    """
    a = Assertion(
        expr="solve([y - 10*x, y - x - 288], [x, y], dict=True)",
        expected=[{"x": 32, "y": 320}],
        description="一次方程组求解",
    )
    r = verify(a)
    assert r.status == "passed"


def test_nested_list_expected_parses_recursively() -> None:
    a = Assertion(expr="[[2, 3]]", expected=[[3, 2]])
    r = verify(a)
    assert r.status == "passed"


# ---------------------------------------------------------------------------
# list-valued expr vs scalar expected (solve() returns a singleton list)
# ---------------------------------------------------------------------------


def test_solve_singleton_list_vs_scalar_expected_passes() -> None:
    """`solve(eq, x)` returns a 1-element list even for a unique solution.

    When the LLM writes `expected` as the scalar answer, the verifier must
    unwrap the singleton list and compare scalar-to-scalar instead of crashing
    with `unsupported operand type(s) for -: 'list' and ...`.
    """
    a = Assertion(
        expr="solve(Eq(x/v1, (D - x)/v2), x)",
        expected="D*v1/(v1 + v2)",
        description="相向到达河岸时间相等求 x 与 D 的关系",
    )
    r = verify(a)
    assert r.status == "passed"


def test_solve_singleton_list_vs_scalar_with_free_vars_passes() -> None:
    a = Assertion(
        expr="solve(Eq(2*x, y), x)",
        expected="y/2",
        free_vars={"y": [-5.0, 5.0]},
    )
    r = verify(a)
    assert r.status == "passed"


def test_singleton_list_literal_vs_scalar_passes() -> None:
    a = Assertion(expr="[255]", expected=255)
    r = verify(a)
    assert r.status == "passed"


def test_singleton_list_vs_scalar_fails_when_wrong() -> None:
    a = Assertion(expr="solve(x - 3, x)", expected=4)
    r = verify(a)
    assert r.status == "failed"


def test_multi_solution_list_vs_scalar_membership_passes() -> None:
    """`solve(x**2-4, x)` returns [-2, 2]; expected scalar 2 is one root."""
    a = Assertion(expr="solve(x**2 - 4, x)", expected=2)
    r = verify(a)
    assert r.status == "passed"


def test_multi_solution_list_vs_scalar_membership_fails_when_absent() -> None:
    a = Assertion(expr="solve(x**2 - 4, x)", expected=5)
    r = verify(a)
    assert r.status == "failed"


def test_empty_solve_list_vs_scalar_fails_gracefully() -> None:
    a = Assertion(expr="solve(Eq(x, x + 1), x)", expected=0)
    r = verify(a)
    assert r.status == "failed"


# ---------------------------------------------------------------------------
# Boolean comparison
# ---------------------------------------------------------------------------


def test_bool_assertion_passes() -> None:
    a = Assertion(expr="5*pi/6 > 0", expected=True)
    r = verify(a)
    assert r.status == "passed"


def test_bool_assertion_fails_when_wrong() -> None:
    a = Assertion(expr="5*pi/6 < 0", expected=True)
    r = verify(a)
    assert r.status == "failed"


# ---------------------------------------------------------------------------
# Sampling path (free_vars)
# ---------------------------------------------------------------------------


def test_pythagorean_identity_passes_via_sampling() -> None:
    a = Assertion(
        expr="sin(x)**2 + cos(x)**2",
        expected=1,
        free_vars={"x": [-3.14, 3.14]},
    )
    r = verify(a)
    assert r.status == "passed"
    # Expect either symbolic (simplify reduces) or sampling confidence.
    assert r.confidence >= 0.94  # sampling floor


def test_non_identity_with_free_vars_fails() -> None:
    a = Assertion(
        expr="sin(x)",
        expected="cos(x)",
        free_vars={"x": [0, 1.0]},
    )
    r = verify(a)
    assert r.status == "failed"


def test_default_sampling_range_when_not_given() -> None:
    # free_vars value is empty list → default range (-10, 10) is used.
    a = Assertion(
        expr="x - x",
        expected=0,
        free_vars={"x": []},
    )
    r = verify(a)
    assert r.status == "passed"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_parse_error_returns_error_status() -> None:
    a = Assertion(expr="x ** ?", expected=0)
    r = verify(a)
    assert r.status == "error"
    assert r.confidence <= 0.4


def test_unbound_symbol_diff_returns_failed_or_error() -> None:
    # Expected has free var that expr doesn't have a way to reduce against.
    a = Assertion(expr="x", expected="y")
    r = verify(a)
    assert r.status in {"failed", "error"}


# ---------------------------------------------------------------------------
# Integration with a real-world style assertion (sanity)
# ---------------------------------------------------------------------------


def test_substitution_into_polynomial() -> None:
    a = Assertion(
        expr="(x**3 - 3*x + 1).subs(x, 2)",
        expected=3,
        description="f(2) for f(x)=x^3-3x+1",
    )
    r = verify(a)
    assert r.status == "passed"


def test_max_min_helpers() -> None:
    assert verify(Assertion(expr="Max(-1, 3, -1, 3)", expected=3)).status == "passed"
    assert verify(Assertion(expr="Min(-1, 3, -1, 3)", expected=-1)).status == "passed"


def test_verify_bool_handles_relational_expected() -> None:
    """Regression for B05 baseline crash: when expr parses to a Relational
    (e.g. `a > 0`) and expected is `true`, `bool(Relational)` raises in SymPy.
    The verifier must catch this and return error, not propagate TypeError."""
    a = Assertion(expr="a > 0", expected=True, description="discriminant condition")
    r = verify(a)
    assert r.status == "error"
    assert "bool" in (r.detail or "").lower()
