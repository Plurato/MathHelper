"""Compare pipeline answers against ground truth via SymPy."""

from __future__ import annotations

from typing import Any

from mathcoach.eval.types import GraderResult
from mathcoach.schemas.verification import AnswerItem
from mathcoach.tools.sympy_verifier import _parse, _values_equal

_NUMERIC_TOLERANCE = 1e-6


def compare(
    pipeline: list[AnswerItem], truth: list[AnswerItem]
) -> GraderResult:
    if len(pipeline) != len(truth):
        flat_outcome = _try_flatten_compare(pipeline, truth)
        if flat_outcome is not None:
            return flat_outcome
        return GraderResult(
            correct=False,
            layer="structure",
            reason=f"item count mismatch: pipeline={len(pipeline)}, truth={len(truth)}",
        )

    pipeline_pairs = _pair_by_label(pipeline, truth)
    if pipeline_pairs is None:
        pipeline_pairs = list(zip(pipeline, truth))

    skipped_layer: str | None = None
    layer_used: str = "exact"

    for p_item, t_item in pipeline_pairs:
        if t_item.sympy is None:
            skipped_layer = "no_sympy"
            continue

        if p_item.sympy is None:
            return GraderResult(
                correct=False,
                layer="structure",
                reason=f"pipeline item '{p_item.label}' missing sympy form",
            )

        try:
            outcome = _compare_item(p_item, t_item)
        except _StructureMismatch as exc:
            return GraderResult(
                correct=False,
                layer="structure",
                reason=f"item '{t_item.label}': {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return GraderResult(
                correct=None,
                layer="error",
                reason=f"{type(exc).__name__}: {exc}",
            )

        if outcome is False:
            return GraderResult(
                correct=False,
                layer="exact",
                reason=(
                    f"item '{t_item.label}': pipeline={p_item.sympy!r} != "
                    f"truth={t_item.sympy!r}"
                ),
            )
        layer_used = outcome  # type: ignore[assignment]

    if skipped_layer is not None:
        return GraderResult(
            correct=None,
            layer="no_sympy",
            reason="at least one truth item has sympy=null; cannot grade programmatically",
        )

    return GraderResult(correct=True, layer=layer_used, reason="")  # type: ignore[arg-type]


def _pair_by_label(
    pipeline: list[AnswerItem], truth: list[AnswerItem]
) -> list[tuple[AnswerItem, AnswerItem]] | None:
    by_label = {p.label: p for p in pipeline}
    if len(by_label) != len(pipeline):
        return None
    if not all(t.label in by_label for t in truth):
        return None
    return [(by_label[t.label], t) for t in truth]


def _compare_item(p_item: AnswerItem, t_item: AnswerItem) -> str | bool:
    """Return layer name on value match, False on value mismatch.

    Raises ``_StructureMismatch`` when the two sides parse to incompatible
    SymPy shapes (e.g. list vs scalar), so the caller can report
    ``layer="structure"`` instead of a misleading value mismatch.
    Routes by shape: list → set-equal, Set → ``==``, Boolean → normalized
    bool compare, else ``simplify(a-b)==0`` with a numeric fallback.
    """
    p_sym = _parse(p_item.sympy or "")
    t_sym = _parse(t_item.sympy or "")

    p_kind = _shape_kind(p_sym)
    t_kind = _shape_kind(t_sym)
    if p_kind != t_kind:
        raise _StructureMismatch(p_kind, t_kind)

    if p_kind == "list":
        return "set" if _set_equal(p_sym, t_sym) else False
    if p_kind == "set":
        return "exact" if p_sym == t_sym else False
    if p_kind == "bool":
        return "exact" if _to_python_bool(p_sym) == _to_python_bool(t_sym) else False

    if _symbolic_equal(p_sym, t_sym):
        return "exact"
    if _numeric_equal(p_item.numeric, t_item.numeric):
        return "numeric"
    if _numeric_equal_via_evalf(p_sym, t_sym):
        return "numeric"

    return False


class _StructureMismatch(Exception):
    """Raised by ``_compare_item`` when pipeline and truth shapes disagree."""

    def __init__(self, p_kind: str, t_kind: str) -> None:
        super().__init__(f"shape mismatch: pipeline={p_kind}, truth={t_kind}")
        self.p_kind = p_kind
        self.t_kind = t_kind


def _shape_kind(obj: Any) -> str:
    if isinstance(obj, list):
        return "list"
    if _is_sympy_set(obj):
        return "set"
    if _is_boolean_like(obj):
        return "bool"
    return "scalar"


def _try_flatten_compare(
    pipeline: list[AnswerItem], truth: list[AnswerItem]
) -> GraderResult | None:
    """Handle "1×list-sympy vs N×scalar-sympy" mismatch by flattening both sides
    to a flat list of SymPy values and comparing as a set.

    Returns None when neither side can be flattened cleanly — caller falls back
    to its own structure-mismatch handling.
    """
    flat_p = _try_flatten(pipeline)
    flat_t = _try_flatten(truth)

    if flat_p is None or flat_t is None:
        return None
    if len(flat_p) != len(flat_t):
        return None

    if _set_equal(flat_p, flat_t):
        return GraderResult(correct=True, layer="set", reason="flatten matched")
    return None


def _try_flatten(items: list[AnswerItem]) -> list[Any] | None:
    if any(it.sympy is None for it in items):
        return None

    if len(items) == 1:
        try:
            parsed = _parse(items[0].sympy or "")
        except Exception:  # noqa: BLE001
            return None
        if isinstance(parsed, list):
            return parsed
        return [parsed]

    flat: list[Any] = []
    for it in items:
        try:
            parsed = _parse(it.sympy or "")
        except Exception:  # noqa: BLE001
            return None
        if isinstance(parsed, list):
            return None  # mixed shape — refuse
        flat.append(parsed)
    return flat


def _is_sympy_set(obj: Any) -> bool:
    import sympy

    return isinstance(obj, sympy.Set)


def _is_boolean_like(obj: Any) -> bool:
    import sympy

    if isinstance(obj, bool):
        return True
    return isinstance(obj, sympy.logic.boolalg.Boolean) and not isinstance(
        obj, sympy.logic.boolalg.BooleanFunction
    ) and obj in (sympy.true, sympy.false)


def _to_python_bool(obj: Any) -> bool:
    import sympy

    if isinstance(obj, bool):
        return obj
    if obj is sympy.true:
        return True
    if obj is sympy.false:
        return False
    return bool(obj)


def _symbolic_equal(a: Any, b: Any) -> bool:
    import sympy

    try:
        diff = sympy.simplify(a - b)
        return diff == 0
    except (TypeError, AttributeError):
        return False


def _numeric_equal(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= _NUMERIC_TOLERANCE


def _numeric_equal_via_evalf(a: Any, b: Any) -> bool:
    try:
        diff = a - b
        return abs(float(diff.evalf())) <= _NUMERIC_TOLERANCE
    except (TypeError, ValueError, AttributeError):
        return False


def _set_equal(a: list[Any], b: list[Any]) -> bool:
    if len(a) != len(b):
        return False
    pool = list(b)
    for av in a:
        match_idx = None
        for i, bv in enumerate(pool):
            if _values_equal(av, bv, _NUMERIC_TOLERANCE):
                match_idx = i
                break
        if match_idx is None:
            return False
        pool.pop(match_idx)
    return True
