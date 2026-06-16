"""Load eval problems from a JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mathcoach.eval.types import EvalProblem


def load_problems(path: str | Path) -> list[EvalProblem]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"problems file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw: Any = json.load(f)

    if not isinstance(raw, dict) or "problems" not in raw:
        raise ValueError(f"{p}: top-level must be an object with a 'problems' key")
    items = raw["problems"]
    if not isinstance(items, list):
        raise ValueError(f"{p}: 'problems' must be a list")

    out: list[EvalProblem] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{p}: problem #{idx} is not an object")
        prob_id = item.get("id", f"<idx={idx}>")
        try:
            out.append(EvalProblem.model_validate(item))
        except ValidationError as exc:
            raise ValueError(
                f"{p}: problem '{prob_id}' failed schema validation: {exc}"
            ) from exc

    seen: set[str] = set()
    for prob in out:
        if prob.id in seen:
            raise ValueError(f"{p}: duplicate problem id '{prob.id}'")
        seen.add(prob.id)

    return out


def check_truth_parsable(problems: list[EvalProblem]) -> list[tuple[str, str, str]]:
    """Return (problem_id, item_label, error) triples for any truth.sympy that
    fails to parse with the SymPy sandbox. Empty list means all OK."""
    from mathcoach.tools.sympy_verifier import _parse  # local import to keep lazy

    failures: list[tuple[str, str, str]] = []
    for prob in problems:
        for item in prob.truth.answer:
            if item.sympy is None:
                continue
            try:
                _parse(item.sympy)
            except Exception as exc:  # noqa: BLE001
                failures.append((prob.id, item.label, f"{type(exc).__name__}: {exc}"))
    return failures


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m mathcoach.eval.loader <path-to-problems.json> [--check]")
        sys.exit(2)

    problems = load_problems(sys.argv[1])
    print(f"Loaded {len(problems)} problems from {sys.argv[1]}")

    if "--check" in sys.argv[2:]:
        failures = check_truth_parsable(problems)
        if failures:
            print(f"\n{len(failures)} truth.sympy parse failures:")
            for pid, label, err in failures:
                print(f"  {pid} [{label}]: {err}")
            sys.exit(1)
        print("All truth.sympy values parse OK.")
