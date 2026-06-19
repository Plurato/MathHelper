"""Independent verification of LLM-emitted assertions via SymPy."""

from __future__ import annotations

import random
from typing import Any

from mathcoach.schemas.verification import Assertion, VerificationResult

_DEFAULT_TOLERANCE = 1e-9
_DEFAULT_SAMPLE_RANGE = (-10.0, 10.0)
_DEFAULT_SAMPLE_COUNT = 10
_MAX_SINGULAR_RETRIES = 3


def verify(assertion: Assertion) -> VerificationResult:
    try:
        import sympy  # noqa: F401  - sympy is optional; fall through gracefully if absent
    except ImportError:
        return VerificationResult(
            method="SymPy 未安装，跳过验证",
            status="skipped",
            confidence=0.5,
            detail="Install the [solve] extra to enable verification.",
        )

    tolerance = (
        assertion.tolerance if assertion.tolerance is not None else _DEFAULT_TOLERANCE
    )

    try:
        expr_obj = _parse(assertion.expr)
        expected_obj = _parse_expected(assertion.expected)
    except Exception as exc:  # noqa: BLE001
        return VerificationResult(
            method=f"SymPy 解析异常: {type(exc).__name__}",
            status="error",
            confidence=0.30,
            detail=str(exc),
        )

    if isinstance(assertion.expected, list):
        return _verify_list(expr_obj, expected_obj, tolerance)

    if isinstance(assertion.expected, bool) or _is_sympy_bool(expected_obj):
        return _verify_bool(expr_obj, expected_obj)

    if assertion.free_vars:
        try:
            return _verify_via_sampling(
                expr_obj, expected_obj, assertion.free_vars, tolerance
            )
        except Exception as exc:  # noqa: BLE001
            return VerificationResult(
                method=f"SymPy 抽样异常: {type(exc).__name__}",
                status="error",
                confidence=0.30,
                detail=str(exc),
            )

    try:
        return _verify_scalar(expr_obj, expected_obj, tolerance)
    except Exception as exc:  # noqa: BLE001
        return VerificationResult(
            method=f"SymPy 验证异常: {type(exc).__name__}",
            status="error",
            confidence=0.30,
            detail=str(exc),
        )


def _verify_scalar(expr_obj: Any, expected_obj: Any, tolerance: float) -> VerificationResult:
    import sympy

    diff = sympy.simplify(expr_obj - expected_obj)
    if diff == 0:
        return VerificationResult(
            method="SymPy 符号验证",
            status="passed",
            confidence=0.98,
            detail=f"expr == expected == {expr_obj}",
        )

    try:
        diff_value = float(diff.evalf())
    except (TypeError, ValueError):
        return VerificationResult(
            method="SymPy 验证",
            status="failed",
            confidence=0.05,
            detail=f"diff cannot be evaluated to a real number: {diff}",
        )

    if abs(diff_value) <= tolerance:
        return VerificationResult(
            method="SymPy 数值验证（容差）",
            status="passed",
            confidence=0.96,
            detail=f"|expr - expected| = {abs(diff_value):.3e} <= {tolerance}",
        )
    return VerificationResult(
        method="SymPy 验证",
        status="failed",
        confidence=0.05,
        detail=f"diff={diff} (numeric={diff_value:.6g}) exceeds tolerance",
    )


def _verify_list(
    expr_obj: Any, expected_obj: Any, tolerance: float
) -> VerificationResult:
    import sympy

    if not isinstance(expected_obj, (list, tuple)):
        return VerificationResult(
            method="SymPy 列表验证",
            status="error",
            confidence=0.30,
            detail="expected was not a list after parsing",
        )

    expected_items = list(expected_obj)

    actual_items: list[Any]
    if isinstance(expr_obj, dict):
        if len(expected_items) == 1 and isinstance(expected_items[0], dict):
            actual_items = [expr_obj]
        else:
            actual_items = list(expr_obj.values())
    elif isinstance(expr_obj, (list, tuple)):
        actual_items = list(expr_obj)
    elif isinstance(expr_obj, sympy.FiniteSet):
        actual_items = list(expr_obj)
    elif isinstance(expr_obj, sympy.Basic):
        try:
            actual_items = list(expr_obj)  # type: ignore[arg-type]
        except TypeError:
            return VerificationResult(
                method="SymPy 列表验证",
                status="failed",
                confidence=0.05,
                detail=f"expr did not evaluate to a list-like value: got {expr_obj}",
            )
    elif hasattr(expr_obj, "__iter__"):
        actual_items = list(expr_obj)
    else:
        return VerificationResult(
            method="SymPy 列表验证",
            status="failed",
            confidence=0.05,
            detail=f"expr did not evaluate to a list-like value: got {type(expr_obj).__name__}",
        )

    if len(actual_items) != len(expected_items):
        return VerificationResult(
            method="SymPy 列表验证",
            status="failed",
            confidence=0.05,
            detail=f"length mismatch: actual={len(actual_items)}, expected={len(expected_items)}",
        )

    actual_pool = list(actual_items)
    for ev in expected_items:
        match_idx = None
        for i, av in enumerate(actual_pool):
            if _values_equal(av, ev, tolerance):
                match_idx = i
                break
        if match_idx is None:
            return VerificationResult(
                method="SymPy 列表验证",
                status="failed",
                confidence=0.05,
                detail=f"expected item {ev} not found in actual {actual_items}",
            )
        actual_pool.pop(match_idx)

    return VerificationResult(
        method="SymPy 列表验证（顺序无关）",
        status="passed",
        confidence=0.98,
        detail=f"all {len(expected_items)} item(s) matched",
    )


def _verify_bool(expr_obj: Any, expected_obj: Any) -> VerificationResult:
    import sympy

    actual = sympy.simplify(expr_obj) if hasattr(expr_obj, "free_symbols") else expr_obj
    expected = (
        sympy.simplify(expected_obj)
        if hasattr(expected_obj, "free_symbols")
        else expected_obj
    )

    try:
        a_bool = bool(actual)
        e_bool = bool(expected)
    except TypeError as exc:
        return VerificationResult(
            method="SymPy 布尔验证",
            status="error",
            confidence=0.30,
            detail=f"cannot reduce to bool: {exc}",
        )

    if a_bool == e_bool:
        return VerificationResult(
            method="SymPy 布尔验证",
            status="passed",
            confidence=0.98,
            detail=f"actual={actual}, expected={expected}",
        )
    return VerificationResult(
        method="SymPy 布尔验证",
        status="failed",
        confidence=0.05,
        detail=f"actual={actual}, expected={expected}",
    )


def _verify_via_sampling(
    expr_obj: Any,
    expected_obj: Any,
    free_vars: dict[str, list[float]],
    tolerance: float,
) -> VerificationResult:
    import sympy

    diff = expr_obj - expected_obj

    rng = random.Random(0)
    n_samples = _DEFAULT_SAMPLE_COUNT
    valid_evals = 0
    failed_at: tuple[dict[str, float], float] | None = None
    ranges = _resolve_sampling_ranges(free_vars)

    def _draw_binding() -> dict[Any, Any]:
        b: dict[Any, Any] = {}
        for name, (low, high) in ranges.items():
            b[sympy.Symbol(name)] = sympy.Float(rng.uniform(low, high))
        return b

    for _ in range(n_samples):
        binding = _draw_binding()
        value_f: float | None = None

        for _retry in range(_MAX_SINGULAR_RETRIES + 1):
            try:
                value = diff.subs(binding).evalf()
                cand = float(value)
                if cand != cand:
                    raise ValueError("NaN")
                if cand in (float("inf"), float("-inf")):
                    raise ValueError("inf")
                value_f = cand
                break
            except Exception:
                binding = _draw_binding()
                continue

        if value_f is None:
            continue

        if abs(value_f) > tolerance:
            failed_at = ({str(k): float(v) for k, v in binding.items()}, value_f)
            break
        valid_evals += 1

    if failed_at is not None:
        binding_str = ", ".join(f"{k}={v:.4f}" for k, v in failed_at[0].items())
        return VerificationResult(
            method="SymPy 抽样验证",
            status="failed",
            confidence=0.05,
            detail=f"counterexample at {binding_str}: residual={failed_at[1]:.6g}",
        )

    if valid_evals == 0:
        return VerificationResult(
            method="SymPy 抽样验证",
            status="error",
            confidence=0.30,
            detail="all sample points hit singularities",
        )

    return VerificationResult(
        method="SymPy 抽样验证",
        status="passed",
        confidence=0.94,
        detail=f"{valid_evals}/{n_samples} samples within tolerance {tolerance}",
    )


def _resolve_sampling_ranges(
    free_vars: dict[str, list[float]],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for name, rng in free_vars.items():
        if not rng or len(rng) < 2:
            out[name] = _DEFAULT_SAMPLE_RANGE
        else:
            low, high = float(rng[0]), float(rng[1])
            if low > high:
                low, high = high, low
            out[name] = (low, high)
    return out


def _values_equal(a: Any, b: Any, tolerance: float) -> bool:
    import sympy

    if isinstance(a, dict) or isinstance(b, dict):
        return _dicts_equal(a, b, tolerance)

    if _is_sequence(a) or _is_sequence(b):
        return _sequences_equal(a, b, tolerance)

    try:
        diff = sympy.simplify(a - b)
        if diff == 0:
            return True
        try:
            return abs(float(diff.evalf())) <= tolerance
        except (TypeError, ValueError):
            return False
    except (TypeError, AttributeError):
        return a == b


def _dicts_equal(a: Any, b: Any, tolerance: float) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if len(a) != len(b):
        return False

    unmatched = list(b.items())
    for ak, av in a.items():
        match_idx = None
        for i, (bk, bv) in enumerate(unmatched):
            if _values_equal(ak, bk, tolerance) and _values_equal(av, bv, tolerance):
                match_idx = i
                break
        if match_idx is None:
            return False
        unmatched.pop(match_idx)
    return True


def _sequences_equal(a: Any, b: Any, tolerance: float) -> bool:
    if not _is_sequence(a) or not _is_sequence(b):
        return False
    a_items = list(a)
    b_items = list(b)
    if len(a_items) != len(b_items):
        return False

    unmatched = list(b_items)
    for av in a_items:
        match_idx = None
        for i, bv in enumerate(unmatched):
            if _values_equal(av, bv, tolerance):
                match_idx = i
                break
        if match_idx is None:
            return False
        unmatched.pop(match_idx)
    return True


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _is_sympy_bool(obj: Any) -> bool:
    import sympy

    return obj is sympy.true or obj is sympy.false


def _build_local_dict() -> dict[str, Any]:
    # Whitelist limits parse_expr to safe SymPy names; bare identifiers outside
    # this list become free symbols, so __import__/os/etc. cannot be reached.
    import sympy

    names = (
        "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
        "exp", "log", "ln", "Abs",
        "pi", "E", "I", "oo", "true", "false",
        "Rational", "Symbol", "Integer", "Float",
        "Eq",
        "Min", "Max", "FiniteSet",
        "simplify", "expand", "factor", "trigsimp",
        "solve", "diff", "integrate", "limit", "summation",
        "binomial", "factorial",
    )
    local: dict[str, Any] = {}
    for n in names:
        attr = getattr(sympy, n, None)
        if attr is not None:
            local[n] = attr
    local.setdefault("ln", sympy.log)
    return local


def _parse(expr_value: Any) -> Any:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr

    if isinstance(expr_value, bool):
        return sympy.true if expr_value else sympy.false
    if isinstance(expr_value, (int, float)):
        return sympy.sympify(expr_value)
    if isinstance(expr_value, list):
        return [_parse(item) for item in expr_value]
    if isinstance(expr_value, tuple):
        return tuple(_parse(item) for item in expr_value)
    if isinstance(expr_value, dict):
        return {_parse_mapping_key(k): _parse(v) for k, v in expr_value.items()}
    if isinstance(expr_value, str):
        if not expr_value.strip():
            raise ValueError("expression must be a non-empty string")
        return parse_expr(expr_value, local_dict=_build_local_dict(), evaluate=True)
    raise TypeError(f"unsupported expr type: {type(expr_value).__name__}")


def _parse_mapping_key(key: Any) -> Any:
    if not isinstance(key, str):
        return _parse(key)
    try:
        return _parse(key)
    except Exception:  # noqa: BLE001 - non-symbol labels remain comparable strings
        return key


def _parse_expected(expected: Any) -> Any:
    return _parse(expected)
