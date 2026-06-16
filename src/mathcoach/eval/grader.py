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
    """Return layer name on match, False on mismatch.

    Tries set → exact → numeric in turn.
    """
    p_sym = _parse(p_item.sympy or "")
    t_sym = _parse(t_item.sympy or "")

    if isinstance(p_sym, list) or isinstance(t_sym, list):
        if not (isinstance(p_sym, list) and isinstance(t_sym, list)):
            return False
        if _set_equal(p_sym, t_sym):
            return "set"
        return False

    if _symbolic_equal(p_sym, t_sym):
        return "exact"

    if _numeric_equal(p_item.numeric, t_item.numeric):
        return "numeric"
    if _numeric_equal_via_evalf(p_sym, t_sym):
        return "numeric"

    return False


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
