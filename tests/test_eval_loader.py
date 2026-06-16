"""Tests for eval problem loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathcoach.eval.loader import check_truth_parsable, load_problems


def test_load_real_dev_set():
    problems = load_problems("data/eval/dev.json")
    assert len(problems) == 20
    assert {p.id for p in problems} == {
        f"A0{i}" for i in range(1, 10)
    } | {"A10"} | {f"B0{i}" for i in range(1, 6)} | {f"C0{i}" for i in range(1, 6)}


def test_real_dev_set_truth_parses():
    problems = load_problems("data/eval/dev.json")
    failures = check_truth_parsable(problems)
    assert failures == [], f"truth.sympy parse failures: {failures}"


def test_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_problems(tmp_path / "nope.json")


def test_top_level_not_object(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="problems"):
        load_problems(p)


def test_missing_required_field(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(
        json.dumps(
            {
                "problems": [
                    {
                        "id": "X01",
                        "group": "A",
                        "type": "test",
                        "knowledge_points": [],
                        "difficulty": "easy",
                        "expected_verifier": "should_pass",
                        "question": "Q?",
                        # truth missing
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="X01"):
        load_problems(p)


def test_duplicate_ids_rejected(tmp_path: Path):
    item = {
        "id": "DUP",
        "group": "A",
        "type": "t",
        "knowledge_points": [],
        "difficulty": "easy",
        "expected_verifier": "should_pass",
        "question": "Q?",
        "truth": {
            "answer": [
                {"label": "x", "latex": "$1$", "sympy": "1", "numeric": 1.0, "unit": None}
            ]
        },
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"problems": [item, item]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_problems(p)
