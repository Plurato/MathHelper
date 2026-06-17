"""Write eval results as CSV, Markdown report, and run metadata."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from mathcoach.eval.types import AggregateMetrics, EvalRow, RunMeta

_CSV_FIELDS = [
    "id",
    "group",
    "type",
    "expected_verifier",
    "correct",
    "grader_status",
    "grader_layer",
    "grader_reason",
    "verifier_status",
    "verifier_confidence",
    "n_assertions",
    "n_passed",
    "n_failed",
    "prompt_tokens",
    "completion_tokens",
    "duration_s",
    "pipeline_status",
    "pipeline_error",
    "pipeline_answer_repr",
    "truth_answer_repr",
]


def write_csv(rows: list[EvalRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            d = row.model_dump()
            d["correct"] = "" if d["correct"] is None else str(d["correct"])
            writer.writerow(d)


def write_meta(meta: RunMeta, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_report(rows: list[EvalRow], path: Path, meta: RunMeta) -> AggregateMetrics:
    metrics = compute_aggregate(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Eval Report — {meta.timestamp}")
    lines.append("")
    lines.append(f"- Model: `{meta.model_name}`")
    lines.append(f"- Agents: {' → '.join(meta.agents)}")
    lines.append(f"- Problems: {meta.n_problems} (from `{meta.problems_path}`)")
    lines.append(f"- Git SHA: `{meta.git_sha}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- correct: **{metrics.n_correct}/{metrics.n_total} ({metrics.correct_rate*100:.1f}%)**")
    lines.append(f"- incorrect: {metrics.n_incorrect}")
    lines.append(f"- grader skipped (no_sympy / error): {metrics.n_grader_skipped}")
    lines.append(f"- pipeline failed: {metrics.n_pipeline_failed}")
    lines.append(f"- FP rate (expected should_pass but answer wrong): **{metrics.fp_rate*100:.1f}%**")
    lines.append(f"- FN rate (verifier failed but answer correct): **{metrics.fn_rate*100:.1f}%**")
    lines.append("")
    lines.append("## By group")
    lines.append("")
    lines.append("| Group | N | correct | incorrect | skipped | failed |")
    lines.append("|------|---|---------|-----------|---------|--------|")
    by_group = _bucket_by_group(rows)
    for grp in ("A", "B", "C"):
        bucket = by_group.get(grp, [])
        if not bucket:
            continue
        n = len(bucket)
        nc = sum(1 for r in bucket if r.correct is True)
        ni = sum(1 for r in bucket if r.correct is False)
        ns = sum(1 for r in bucket if r.correct is None and r.pipeline_status == "ok")
        nf = sum(1 for r in bucket if r.pipeline_status == "failed")
        lines.append(f"| {grp} | {n} | {nc} | {ni} | {ns} | {nf} |")
    lines.append("")

    failures = [
        r for r in rows
        if r.correct is False
        or r.correct is None
        or r.pipeline_status == "failed"
    ]
    lines.append(f"## Failures and skips ({len(failures)})")
    lines.append("")
    if not failures:
        lines.append("_None._")
    for r in failures:
        lines.append(f"### {r.id} [{r.group}] — {r.type}")
        lines.append("")
        lines.append(f"- expected_verifier: `{r.expected_verifier}`")
        lines.append(f"- pipeline_status: `{r.pipeline_status}`")
        if r.pipeline_error:
            lines.append(f"- pipeline_error: `{r.pipeline_error}`")
        lines.append(f"- grader: `{r.grader_status}` (layer={r.grader_layer})")
        if r.grader_reason:
            lines.append(f"- grader_reason: {r.grader_reason}")
        lines.append(f"- verifier: status=`{r.verifier_status}`, confidence=`{r.verifier_confidence}`")
        if r.pipeline_answer_repr:
            lines.append(f"- pipeline_answer: `{r.pipeline_answer_repr}`")
        lines.append(f"- truth_answer: `{r.truth_answer_repr}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return metrics


def compute_aggregate(rows: list[EvalRow]) -> AggregateMetrics:
    n = len(rows)
    n_correct = sum(1 for r in rows if r.correct is True)
    n_incorrect = sum(1 for r in rows if r.correct is False)
    n_skipped = sum(1 for r in rows if r.correct is None and r.pipeline_status == "ok")
    n_failed = sum(1 for r in rows if r.pipeline_status == "failed")

    n_should_pass = sum(1 for r in rows if r.expected_verifier == "should_pass")
    n_fp = sum(
        1 for r in rows
        if r.expected_verifier == "should_pass" and r.correct is False
    )
    n_correct_total = max(1, n_correct + n_incorrect)
    n_fn = sum(
        1 for r in rows
        if r.correct is True and r.verifier_status == "failed"
    )

    return AggregateMetrics(
        n_total=n,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        n_grader_skipped=n_skipped,
        n_pipeline_failed=n_failed,
        correct_rate=(n_correct / n) if n else 0.0,
        fp_rate=(n_fp / n_should_pass) if n_should_pass else 0.0,
        fn_rate=(n_fn / n_correct_total) if n_correct_total else 0.0,
    )


def _bucket_by_group(rows: list[EvalRow]) -> dict[str, list[EvalRow]]:
    out: dict[str, list[EvalRow]] = defaultdict(list)
    for r in rows:
        out[r.group].append(r)
    return out


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"
