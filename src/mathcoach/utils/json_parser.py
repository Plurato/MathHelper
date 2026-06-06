"""Robust JSON extraction helpers for LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from raw LLM text.

    Supports:
    - Pure JSON strings
    - Markdown ```json fenced blocks
    - JSON embedded in surrounding prose
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Cannot extract JSON from empty text.")

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced_match:
        return _loads_object(fenced_match.group(1))

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response.")

    candidate = _extract_balanced_object(stripped[start:])
    return _loads_object(candidate)


def _extract_balanced_object(text: str) -> str:
    """Return the first balanced {...} substring."""
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[: index + 1]

    raise ValueError("Unbalanced JSON object in LLM response.")


def _loads_object(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object at the top level.")
    return parsed
