#!/usr/bin/env python3
"""CLI demo for the full MathCoach agent pipeline (4 agents)."""

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
from mathcoach.agents.solving_verification import (
    SolvingVerificationAgent,
    SolvingVerificationInput,
)
from mathcoach.agents.teaching_explanation import (
    TeachingExplanationAgent,
    TeachingExplanationInput,
)
from mathcoach.schemas.inputs import UserQuery
from mathcoach.utils.trace_printer import print_agent_trace

DEFAULT_QUESTION = (
    "求函数 f(x)=x^3-3x+1 在区间 [-2,2] 上的最大值和最小值。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full 4-agent MathCoach pipeline."
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

    # --- Agent 1: Problem Understanding ---
    understanding_agent = ProblemUnderstandingAgent(**agent_kwargs)
    # --- Agent 2: Solving Planning ---
    planning_agent = SolvingPlanningAgent(**agent_kwargs)
    # --- Agent 3: Solving Verification ---
    verification_agent = SolvingVerificationAgent(**agent_kwargs)
    # --- Agent 4: Teaching Explanation ---
    teaching_agent = TeachingExplanationAgent(**agent_kwargs)

    print("=== Pipeline Input ===")
    print(json.dumps(user_query.model_dump(), ensure_ascii=False, indent=2))

    # Agent 1
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

    # Agent 2
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

    # Agent 3
    verification_input = SolvingVerificationInput(
        analysis=understanding_result.output,
        plan=planning_result.output,
        original_question=user_query.question,
    )
    verification_result = verification_agent.run_with_trace(verification_input)
    if args.quiet:
        print("\n=== Solving Verification Agent ===")
        print(
            json.dumps(
                verification_result.output.model_dump(),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_agent_trace(
            verification_result,
            show_prompts=not args.hide_prompts,
        )

    # Agent 4
    teaching_input = TeachingExplanationInput(
        analysis=understanding_result.output,
        plan=planning_result.output,
        verification=verification_result.output,
        original_question=user_query.question,
        student_level=user_query.student_level,
        explanation_style=user_query.explanation_style,
    )
    teaching_result = teaching_agent.run_with_trace(teaching_input)
    if args.quiet:
        print("\n=== Teaching Explanation Agent ===")
        print(
            json.dumps(
                teaching_result.output.model_dump(),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_agent_trace(
            teaching_result,
            show_prompts=not args.hide_prompts,
        )


if __name__ == "__main__":
    main()
