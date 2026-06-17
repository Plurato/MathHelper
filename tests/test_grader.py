"""Tests for the grader's equivalence layers."""

from __future__ import annotations

from mathcoach.eval.grader import compare
from mathcoach.schemas.verification import AnswerItem


def _item(label: str, sympy: str | None, numeric: float | None = None) -> AnswerItem:
    return AnswerItem(label=label, latex=f"${sympy}$" if sympy else "$?$", sympy=sympy, numeric=numeric, unit=None)


def test_structure_layer_length_mismatch():
    pipe = [_item("x", "1")]
    truth = [_item("x", "1"), _item("y", "2")]
    res = compare(pipe, truth)
    assert res.correct is False
    assert res.layer == "structure"


def test_exact_layer_simplify_equivalent():
    pipe = [_item("v", "sqrt(12)")]
    truth = [_item("v", "2*sqrt(3)")]
    res = compare(pipe, truth)
    assert res.correct is True
    assert res.layer == "exact"


def test_exact_layer_pi_form():
    pipe = [_item("B", "5*pi/6")]
    truth = [_item("B", "5*pi/6")]
    res = compare(pipe, truth)
    assert res.correct is True
    assert res.layer == "exact"


def test_numeric_layer_via_evalf():
    pipe = [_item("v", "sqrt(2)", numeric=1.4142135623730951)]
    truth = [_item("v", "sqrt(2)", numeric=1.4142135623730951)]
    res = compare(pipe, truth)
    assert res.correct is True
    assert res.layer in ("exact", "numeric")


def test_set_layer_order_independent():
    pipe = [_item("x", "[3, 2]")]
    truth = [_item("x", "[2, 3]")]
    res = compare(pipe, truth)
    assert res.correct is True
    assert res.layer == "set"


def test_set_layer_mismatch():
    pipe = [_item("x", "[2, 3]")]
    truth = [_item("x", "[2, 4]")]
    res = compare(pipe, truth)
    assert res.correct is False


def test_label_pairing_reorders():
    pipe = [_item("min", "-1"), _item("max", "3")]
    truth = [_item("max", "3"), _item("min", "-1")]
    res = compare(pipe, truth)
    assert res.correct is True


def test_pipeline_missing_sympy_when_truth_has_it():
    pipe = [_item("x", None)]
    truth = [_item("x", "1")]
    res = compare(pipe, truth)
    assert res.correct is False
    assert res.layer == "structure"


def test_truth_no_sympy_returns_skip():
    pipe = [_item("x", "1")]
    truth = [_item("x", None)]
    res = compare(pipe, truth)
    assert res.correct is None
    assert res.layer == "no_sympy"


def test_parse_error_returns_grader_error():
    pipe = [_item("x", "x ** ?")]
    truth = [_item("x", "1")]
    res = compare(pipe, truth)
    assert res.correct is None
    assert res.layer == "error"


def test_genuine_mismatch():
    pipe = [_item("v", "5")]
    truth = [_item("v", "6")]
    res = compare(pipe, truth)
    assert res.correct is False


def test_set_equality_intervals_with_canonical_form():
    """Interval.open(...) ↔ Interval(..., True, True) are the same SymPy
    object; grader should route by Set type and use ==."""
    pipe = [_item("解集", "Union(Interval(-oo, 1, True, True), Interval(3, oo, True, True))")]
    truth = [_item("解集", "Union(Interval.open(-oo, 1), Interval.open(3, oo))")]
    res = compare(pipe, truth)
    assert res.correct is True
    assert res.layer == "exact"


def test_set_inequality_different_intervals():
    pipe = [_item("解集", "Interval.open(0, 1)")]
    truth = [_item("解集", "Interval.open(0, 2)")]
    res = compare(pipe, truth)
    assert res.correct is False


def test_bool_python_vs_sympy_normalized():
    """Python False (Python bool) vs sympy false (BooleanFalse) must compare equal."""
    pipe = [_item("命题真假", "False")]
    truth = [_item("命题真假", "false")]
    res = compare(pipe, truth)
    assert res.correct is True
    assert res.layer == "exact"


def test_bool_true_false_mismatch():
    pipe = [_item("p", "true")]
    truth = [_item("p", "false")]
    res = compare(pipe, truth)
    assert res.correct is False


def test_shape_flatten_one_list_vs_n_scalars():
    """truth is 1×list-sympy, pipeline is N×scalar-sympy → flatten to set compare."""
    pipe = [_item("x_1", "2"), _item("x_2", "3")]
    truth = [_item("x", "[2, 3]")]
    res = compare(pipe, truth)
    assert res.correct is True
    assert res.layer == "set"


def test_shape_flatten_with_param_symbols():
    """Same shape mismatch for parametric solutions like B02."""
    pipe = [_item("x_1", "-m"), _item("x_2", "1")]
    truth = [_item("x", "[1, -m]")]
    res = compare(pipe, truth)
    assert res.correct is True
    assert res.layer == "set"


def test_shape_flatten_unrelated_lengths_returns_structure_fail():
    """1×list-sympy with 2 elements vs N×scalar-sympy with 3 elements is a real
    mismatch; flatten correctly identifies length mismatch."""
    pipe = [_item("x_1", "2"), _item("x_2", "3"), _item("x_3", "5")]
    truth = [_item("x", "[2, 3]")]
    res = compare(pipe, truth)
    assert res.correct is False
    assert res.layer == "structure"


def test_shape_flatten_mixed_types_returns_structure_fail():
    """Pipeline has 2 items where one is itself a list — flatten should refuse
    and fall back to structure failure."""
    pipe = [_item("x_1", "[1, 2]"), _item("x_2", "3")]
    truth = [_item("x", "[1, 2, 3]")]
    res = compare(pipe, truth)
    assert res.correct is False
    assert res.layer == "structure"


def test_type_mismatch_list_vs_scalar_returns_structure():
    """1-item list answer compared against 1-item scalar answer is a shape
    mismatch, not a value mismatch — layer must be `structure`."""
    pipe = [_item("x", "2")]
    truth = [_item("x", "[2]")]
    res = compare(pipe, truth)
    assert res.correct is False
    assert res.layer == "structure"


def test_type_mismatch_set_vs_scalar_returns_structure():
    pipe = [_item("解集", "2")]
    truth = [_item("解集", "Interval.open(0, 5)")]
    res = compare(pipe, truth)
    assert res.correct is False
    assert res.layer == "structure"


def test_type_mismatch_bool_vs_scalar_returns_structure():
    pipe = [_item("p", "1")]
    truth = [_item("p", "true")]
    res = compare(pipe, truth)
    assert res.correct is False
    assert res.layer == "structure"
