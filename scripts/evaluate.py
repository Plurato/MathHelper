#!/usr/bin/env python3
"""Run end-to-end evaluation over a JSON problem set."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathcoach.eval import reporter, runner
from mathcoach.eval.loader import load_problems
from mathcoach.eval.types import RunMeta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MathCoach end-to-end eval.")
    parser.add_argument(
        "--problems", default="data/eval/dev.json", help="Path to problem set JSON"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run first N problems")
    parser.add_argument(
        "--full", action="store_true", help="Include Teaching agent (4-agent run)"
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="output/eval",
        help="Root output dir; results land in <root>/<timestamp>/",
    )
    parser.add_argument("--model", default=None, help="OpenRouter model override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    problems = load_problems(args.problems)
    if args.limit:
        problems = problems[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    traces_dir = run_dir / "traces"

    agents_used = ["Understanding", "Planning", "Verification"]
    if args.full:
        agents_used.append("Teaching")

    print(f"Eval starting: {len(problems)} problems → {run_dir}")
    rows = runner.run(
        problems,
        full_pipeline=args.full,
        concurrency=args.concurrency,
        traces_dir=traces_dir,
        agent_kwargs={"model": args.model} if args.model else None,
    )

    meta = RunMeta(
        timestamp=timestamp,
        git_sha=reporter.get_git_sha(),
        model_name=args.model or "default",
        agents=agents_used,
        concurrency=args.concurrency,
        n_problems=len(problems),
        problems_path=str(Path(args.problems).resolve()),
        output_dir=str(run_dir.resolve()),
    )

    reporter.write_csv(rows, run_dir / "results.csv")
    metrics = reporter.write_report(rows, run_dir / "report.md", meta)
    reporter.write_meta(meta, run_dir / "meta.json")

    print(
        f"Eval done: {metrics.n_correct}/{metrics.n_total} correct, "
        f"FP={metrics.fp_rate*100:.1f}%, FN={metrics.fn_rate*100:.1f}%, "
        f"output={run_dir}"
    )


if __name__ == "__main__":
    main()
