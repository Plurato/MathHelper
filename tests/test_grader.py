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
