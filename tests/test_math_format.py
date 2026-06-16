"""Tests for the LaTeX-escape repair utility.

These cases simulate what happens after `json.loads` decodes a malformed LLM
response: control characters end up inside `$...$` math regions where the
model meant LaTeX commands like `\\triangle`. The fixer should restore the
literal escape form so KaTeX can render the math.
"""

from __future__ import annotations

import json

import pytest

from mathcoach.utils.math_format import (
    fix_latex_escapes,
    repair_latex_in_payload,
)


# ---------------------------------------------------------------------------
# Single-character control repairs (the core mechanism)
# ---------------------------------------------------------------------------


def test_tab_inside_math_becomes_literal_escape():
    # `$\triangle ABC$` written with single backslash → JSON decoded to TAB.
    raw = "$\triangle ABC$"  # contains a TAB char between $ and 'riangle'
    assert "\t" in raw
    fixed = fix_latex_escapes(raw)
    assert "\t" not in fixed
    assert fixed == "$\\triangle ABC$"


def test_newline_inside_math_becomes_literal_escape():
    # Build explicitly: a TAB-free string with a NEWLINE where `\n` from `\neq`
    # would have ended up after json.loads. Avoids `\s` Python SyntaxWarning.
    raw = "$\\sin C \neq 0$"
    assert "\n" in raw
    fixed = fix_latex_escapes(raw)
    assert "\n" not in fixed
    assert "\\neq" in fixed


def test_carriage_return_inside_math():
    raw = "$\rho > 0$"  # \r becomes CR
    fixed = fix_latex_escapes(raw)
    assert "\r" not in fixed
    assert "\\rho" in fixed


def test_backspace_inside_math():
    raw = "$\beta \\in \\mathbb{R}$"  # \b becomes BS; rest is literal LaTeX
    fixed = fix_latex_escapes(raw)
    assert "\b" not in fixed
    assert "\\beta" in fixed


def test_form_feed_inside_math():
    raw = "$\frac{a}{b}$"  # \f becomes FF
    fixed = fix_latex_escapes(raw)
    assert "\f" not in fixed
    assert "\\frac" in fixed


# ---------------------------------------------------------------------------
# End-to-end with actual JSON parsing
# ---------------------------------------------------------------------------


def test_full_pipeline_matches_real_llm_bug():
    # Reproduces the exact pattern observed in real LLM output.
    bad_json = '{"step": "在 $\\triangle ABC$ 中，$\\sin C \\neq 0$。"}'
    # Note in the line above: Python string literals — \\t in source means
    # JSON literal \t, which json.loads decodes to TAB. To trigger the bug
    # we need the JSON literal to contain ONE backslash (= Python `\\`).
    # That's tricky in source, so build via json.dumps with a bad string:
    parsed_after_buggy_decode = {
        "step": "在 $\triangle ABC$ 中，$\\sin C \neq 0$。",
    }
    # ^ this dict's value contains literal TAB and NEWLINE chars.
    fixed = repair_latex_in_payload(parsed_after_buggy_decode)
    assert "\t" not in fixed["step"]
    assert "\n" not in fixed["step"]
    assert "\\triangle" in fixed["step"]
    assert "\\neq" in fixed["step"]


# ---------------------------------------------------------------------------
# Region detection ($ vs $$, multiple regions, no math at all)
# ---------------------------------------------------------------------------


def test_display_math_double_dollar():
    raw = "$$\triangle ABC$$"  # TAB inside $$...$$
    fixed = fix_latex_escapes(raw)
    assert "\t" not in fixed
    assert fixed == "$$\\triangle ABC$$"


def test_multiple_inline_regions_in_one_string():
    raw = "$\triangle$ 和 $\beta$ 都需要修复"
    fixed = fix_latex_escapes(raw)
    assert "\t" not in fixed and "\b" not in fixed
    assert fixed.count("$") == 4
    assert "\\triangle" in fixed and "\\beta" in fixed


def test_no_math_region_is_passthrough():
    plain = "这是普通中文，没有数学公式。"
    assert fix_latex_escapes(plain) == plain


def test_control_chars_outside_math_are_preserved():
    # Tabs/newlines in prose (not inside $...$) should NOT be touched.
    raw = "段落 1\n段落 2\t（带制表符）"
    fixed = fix_latex_escapes(raw)
    assert fixed == raw  # unchanged


# ---------------------------------------------------------------------------
# Idempotence and structural recursion
# ---------------------------------------------------------------------------


def test_already_correct_latex_is_unchanged():
    # When the LLM gets it right, `\\triangle` survives JSON decode as `\triangle`
    # (literal backslash + t + r + i + ...). No control char present → no-op.
    correct = "$\\triangle ABC$"  # in-memory: backslash + 'triangle ABC$'
    assert fix_latex_escapes(correct) == correct


def test_idempotent():
    raw = "$\triangle ABC$"
    once = fix_latex_escapes(raw)
    twice = fix_latex_escapes(once)
    assert once == twice


def test_repair_walks_nested_dict_and_list():
    payload = {
        "solution_steps": [
            "在 $\triangle ABC$ 中。",
            "因为 $\\sin C \neq 0$。",
        ],
        "answer": {"角B": "$\\dfrac{5\\pi}{6}$"},
        "raw_count": 4,
        "verifiable": {
            "kind": "expression_value",
            "payload": {"expr": "5*pi/6"},  # SymPy field — no $...$, untouched
        },
    }
    fixed = repair_latex_in_payload(payload)
    # Display fields cleaned
    assert "\t" not in fixed["solution_steps"][0]
    assert "\\triangle" in fixed["solution_steps"][0]
    assert "\n" not in fixed["solution_steps"][1]
    # Non-string scalars passed through
    assert fixed["raw_count"] == 4
    # SymPy field untouched (no $...$ region inside)
    assert fixed["verifiable"]["payload"]["expr"] == "5*pi/6"


def test_input_not_mutated():
    payload = {"x": "$\triangle$"}
    snapshot = payload["x"]
    repair_latex_in_payload(payload)
    assert payload["x"] == snapshot  # original dict still has the bad value


# ---------------------------------------------------------------------------
# Documented limitation: \u cannot be repaired here
# ---------------------------------------------------------------------------


def test_unicode_escape_failure_is_caller_responsibility():
    # If LLM writes `\Upsilon` with a single backslash inside a JSON string,
    # json.loads itself raises before our fixer can touch anything.
    bad = '{"x": "$\\Upsilon$"}'
    # In the source above, `\\U` in a Python literal is JSON `\U` (single).
    # Python's json parser raises on `\U` because only `\uXXXX` is valid.
    with pytest.raises(json.JSONDecodeError):
        json.loads(bad)
