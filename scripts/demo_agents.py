#!/usr/bin/env python3
"""CLI demo for the first two MathCoach agents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathcoach.agents.problem_understanding import ProblemUnderstandingAgent
from mathcoach.agents.solving_planning import (
    SolvingPlanningAgent,
    SolvingPlanningInput,
)
from mathcoach.schemas.inputs import UserQuery
from mathcoach.utils.trace_printer import print_agent_trace

DEFAULT_QUESTION = (
    "求函数 f(x)=x^3-3x+1 在区间 [-2,2] 上的最大值和最小值。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run problem understanding and solving planning agents."
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Math question to analyze.",
    )
    parser.add_argument(
        "--student-level",
        default=None,
        help="Optional student level hint.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional OpenRouter model override.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print validated final outputs, without execution traces.",
    )
    parser.add_argument(
        "--hide-prompts",
        action="store_true",
        help="Hide system/user prompts in the execution trace.",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Do not request reasoning content from the model provider.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_query = UserQuery(
        question=args.question,
        student_level=args.student_level,
    )

    agent_kwargs = {
        "model": args.model,
        "include_reasoning": not args.no_reasoning,
    }
    understanding_agent = ProblemUnderstandingAgent(**agent_kwargs)
    planning_agent = SolvingPlanningAgent(**agent_kwargs)

    print("=== Pipeline Input ===")
    print(json.dumps(user_query.model_dump(), ensure_ascii=False, indent=2))

    understanding_result = understanding_agent.run_with_trace(user_query)
    if args.quiet:
        print("\n=== Problem Understanding Agent ===")
        print(
            json.dumps(
                understanding_result.output.model_dump(),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_agent_trace(
            understanding_result,
            show_prompts=not args.hide_prompts,
        )

    planning_input = SolvingPlanningInput(
        analysis=understanding_result.output,
        original_question=user_query.question,
    )
    planning_result = planning_agent.run_with_trace(planning_input)
    if args.quiet:
        print("\n=== Solving Planning Agent ===")
        print(
            json.dumps(
                planning_result.output.model_dump(),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_agent_trace(
            planning_result,
            show_prompts=not args.hide_prompts,
        )


if __name__ == "__main__":
    main()
