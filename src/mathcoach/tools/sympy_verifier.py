"""Independent verification of solving-agent output via SymPy.

The solving verification agent emits a `VerifiableArtifact` describing what
should be checked and the candidate answer in a machine-parseable form. This
module turns that artifact into a real numerical/symbolic check using SymPy
and reports back a `VerificationResult` whose `confidence` reflects the
*tool's* judgment rather than the LLM's self-assessment.

Design notes:
- SymPy is imported lazily so the package still works without the [solve]
  extra installed.
- Expressions are parsed with `parse_expr` against a restricted local_dict to
  avoid arbitrary attribute access. `eval` is never used.
- Each verifier returns one of: passed / failed / error / skipped /
  not_verifiable / not_implemented.
- P0 implements `equation_roots` and `expression_value`. Other kinds return
  a `skipped` result with confidence 0.5 until P1 lands.
"""

from __future__ import annotations

from typing import Any, Callable

from mathcoach.schemas.verification import (
    VerifiableArtifact,
    VerificationResult,
)

_NUMERIC_TOLERANCE = 1e-9


def verify(artifact: VerifiableArtifact) -> VerificationResult:
    """Dispatch a `VerifiableArtifact` to the matching SymPy verifier."""
    try:
        import sympy  # noqa: F401  - lazy import; we re-import inside helpers
    except ImportError:
        return VerificationResult(
            method="SymPy 未安装，跳过验证",
            status="skipped",
            confidence=0.5,
            detail="Install the [solve] extra to enable verification.",
        )

    dispatcher: dict[str, Callable[[dict[str, Any]], VerificationResult]] = {
        "equation_roots": _verify_equation_roots,
        "expression_value": _verify_expression_value,
        "function_extrema": _not_implemented("function_extrema"),
        "trig_identity": _not_implemented("trig_identity"),
        "system_solution": _not_implemented("system_solution"),
        "none": _not_verifiable,
    }
    handler = dispatcher.get(artifact.kind)
    if handler is None:
        return VerificationResult(
            method=f"未知 kind: {artifact.kind}",
            status="error",
            confidence=0.3,
        )
    try:
        return handler(artifact.payload)
    except Exception as exc:  # noqa: BLE001 - surface any SymPy/parser failure
        return VerificationResult(
            method=f"SymPy 验证异常: {type(exc).__name__}",
            status="error",
            confidence=0.3,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Individual verifiers
# ---------------------------------------------------------------------------


def _verify_equation_roots(payload: dict[str, Any]) -> VerificationResult:
    """Check that each candidate root makes `expr` evaluate to 0.

    Payload:
        expr (str)   — SymPy expression that should equal zero at each root.
        var  (str)   — symbol name representing the variable.
        roots (list) — candidate roots; each may be a number or a SymPy string.
    """
    import sympy

    expr_str = _require(payload, "expr", str)
    var_name = _require(payload, "var", str)
    roots = _require(payload, "roots", list)

    expr = _parse(expr_str, sympy)
    var = sympy.Symbol(var_name)

    failures: list[str] = []
    for raw_root in roots:
        root_expr = _parse(str(raw_root), sympy)
        residual = sympy.simplify(expr.subs(var, root_expr))
        if residual != 0:
            failures.append(f"{raw_root} -> residual={residual}")

    if failures:
        return VerificationResult(
            method="SymPy 方程根代入验证",
            status="failed",
            confidence=0.05,
            detail="; ".join(failures),
        )
    return VerificationResult(
        method="SymPy 方程根代入验证",
        status="passed",
        confidence=0.95,
        detail=f"All {len(roots)} root(s) satisfy the equation.",
    )


def _verify_expression_value(payload: dict[str, Any]) -> VerificationResult:
    """Check that `expr` evaluates to `expected` under the given substitutions.

    Payload:
        expr (str)        — SymPy expression to evaluate.
        substitute (dict) — variable -> SymPy string mapping (may be empty).
        expected          — number or SymPy string representing the target.
        tolerance (float, optional) — absolute tolerance for numeric comparison.
    """
    import sympy

    expr_str = _require(payload, "expr", str)
    substitute = payload.get("substitute", {}) or {}
    if not isinstance(substitute, dict):
        raise TypeError("`substitute` must be a dict of name -> expression string")
    expected_raw = payload["expected"] if "expected" in payload else _missing("expected")
    tolerance = float(payload.get("tolerance", _NUMERIC_TOLERANCE))

    expr = _parse(expr_str, sympy)
    sub_pairs = {sympy.Symbol(k): _parse(str(v), sympy) for k, v in substitute.items()}
    actual_sym = sympy.simplify(expr.subs(sub_pairs))
    expected_sym = _parse(str(expected_raw), sympy)

    diff = sympy.simplify(actual_sym - expected_sym)
    if diff == 0:
        return VerificationResult(
            method="SymPy 表达式取值验证",
            status="passed",
            confidence=0.95,
            detail=f"actual = expected = {actual_sym}",
        )

    # Symbolic diff is non-zero — fall back to numeric tolerance check.
    try:
        diff_value = float(diff.evalf())
    except (TypeError, ValueError):
        return VerificationResult(
            method="SymPy 表达式取值验证",
            status="failed",
            confidence=0.05,
            detail=f"actual={actual_sym}, expected={expected_sym}, diff={diff}",
        )
    if abs(diff_value) <= tolerance:
        return VerificationResult(
            method="SymPy 表达式取值验证（数值容差）",
            status="passed",
            confidence=0.9,
            detail=f"|actual - expected| = {abs(diff_value):.3e} <= {tolerance}",
        )
    return VerificationResult(
        method="SymPy 表达式取值验证",
        status="failed",
        confidence=0.05,
        detail=f"actual={actual_sym}, expected={expected_sym}, diff={diff_value:.6g}",
    )


def _not_verifiable(_payload: dict[str, Any]) -> VerificationResult:
    return VerificationResult(
        method="无法机器验证（kind=none）",
        status="not_verifiable",
        confidence=0.5,
        detail="LLM marked this problem as not amenable to automated verification.",
    )


def _not_implemented(kind: str) -> Callable[[dict[str, Any]], VerificationResult]:
    def _handler(_payload: dict[str, Any]) -> VerificationResult:
        return VerificationResult(
            method=f"SymPy 验证器未实现: {kind}",
            status="skipped",
            confidence=0.5,
            detail="This artifact kind is reserved for a later phase.",
        )

    return _handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Whitelisted SymPy names that LLM-emitted expressions may reference. parse_expr
# resolves bare identifiers against this dict, so anything not listed here will
# be treated as a fresh symbol — which keeps `__import__`, `os.system`, etc.
# unreachable.
def _build_local_dict(sympy_module: Any) -> dict[str, Any]:
    names = (
        "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
        "exp", "log", "ln", "Abs",
        "pi", "E", "I", "oo",
        "Rational", "Symbol", "Integer", "Float",
        "Min", "Max",
    )
    local: dict[str, Any] = {}
    for n in names:
        attr = getattr(sympy_module, n, None)
        if attr is not None:
            local[n] = attr
    # Common aliases
    local.setdefault("ln", sympy_module.log)
    return local


def _parse(expr_str: str, sympy_module: Any) -> Any:
    from sympy.parsing.sympy_parser import parse_expr

    if not isinstance(expr_str, str) or not expr_str.strip():
        raise ValueError("expression must be a non-empty string")
    return parse_expr(
        expr_str,
        local_dict=_build_local_dict(sympy_module),
        evaluate=True,
    )


def _require(payload: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in payload:
        _missing(key)
    value = payload[key]
    if not isinstance(value, expected_type):
        raise TypeError(
            f"payload['{key}'] must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def _missing(key: str) -> None:
    raise KeyError(f"payload missing required key: '{key}'")
