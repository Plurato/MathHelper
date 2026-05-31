"""Structured chat completion results from the LLM client."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatCompletionResult:
    """Detailed result from a single chat completion request."""

    content: str
    model: str
    reasoning: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
