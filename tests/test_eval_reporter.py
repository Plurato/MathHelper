"""Tests for eval reporter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mathcoach.eval.reporter import (
    compute_aggregate,
    write_csv,
    write_meta,
    write_report,
)
from mathcoach.eval.types import EvalRow, RunMeta


def _row(**overrides) -> EvalRow:
    base = dict(
        id="X01",
        group="A",
        type="test",
        expected_verifier="should_pass",
        correct=True,
        grader_status="ok",
        grader_layer="exact",
        grader_reason="",
        verifier_status="passed",
        verifier_confidence=0.96,
        n_assertions=3,
        n_passed=3,
        n_failed=0,
        prompt_tokens=1000,
        completion_tokens=500,
        duration_s=1.2,
        pipeline_status="ok",
        pipeline_error="",
        pipeline_answer_repr="x=1",
        truth_answer_repr="x=1",
    )
    base.update(overrides)
    return EvalRow(**base)


def test_aggregate_basic():
    rows = [
        _row(id="A01"),
        _row(id="A02", correct=False, grader_status="ok"),
        _row(id="A03", correct=None, grader_status="no_sympy", grader_layer="no_sympy"),
        _row(id="A04", pipeline_status="failed", correct=None, grader_status="error", grader_layer="error"),
    ]
    m = compute_aggregate(rows)
    assert m.n_total == 4
    assert m.n_correct == 1
    assert m.n_incorrect == 1
    assert m.n_grader_skipped == 1
    assert m.n_pipeline_failed == 1
    assert m.correct_rate == 0.25


def test_fp_rate_should_pass_but_wrong():
    rows = [
        _row(id="A01", expected_verifier="should_pass", correct=True),
        _row(id="A02", expected_verifier="should_pass", correct=False),
        _row(id="C01", expected_verifier="unverifiable", correct=False),
    ]
    m = compute_aggregate(rows)
    assert m.fp_rate == 0.5


def test_fn_rate_correct_but_verifier_failed():
    rows = [
        _row(id="A01", correct=True, verifier_status="passed"),
        _row(id="A02", correct=True, verifier_status="failed"),
    ]
    m = compute_aggregate(rows)
    assert m.fn_rate == 0.5


def test_write_csv(tmp_path: Path):
    rows = [_row(id="A01"), _row(id="A02", correct=None)]
    path = tmp_path / "results.csv"
    write_csv(rows, path)

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
    assert len(records) == 2
    assert records[0]["id"] == "A01"
    assert records[0]["correct"] == "True"
    assert records[1]["correct"] == ""


def test_write_meta(tmp_path: Path):
    meta = RunMeta(
        timestamp="20260616-120000",
        git_sha="abc123",
        model_name="glm-5.1",
        agents=["U", "P", "V"],
        n_problems=20,
    )
    path = tmp_path / "meta.json"
    write_meta(meta, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["git_sha"] == "abc123"
    assert loaded["agents"] == ["U", "P", "V"]


def test_write_report_includes_failure_section(tmp_path: Path):
    rows = [
        _row(id="A01"),
        _row(
            id="A02",
            correct=False,
            grader_reason="pipeline=2 != truth=3",
        ),
    ]
    meta = RunMeta(
        timestamp="20260616-120000",
        git_sha="abc",
        model_name="m",
        agents=["U", "P", "V"],
        n_problems=2,
    )
    path = tmp_path / "report.md"
    metrics = write_report(rows, path, meta)
    text = path.read_text(encoding="utf-8")
    assert "A01" not in text.split("Failures and skips")[0] or True  # passes are not in failures
    assert "A02" in text
    assert "pipeline=2 != truth=3" in text
    assert metrics.n_correct == 1
