"""OpenRouter API client wrapper."""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from mathcoach.config import Settings, get_settings
from mathcoach.llm.chat_result import ChatCompletionResult


class OpenRouterClient:
    """Thin wrapper around the OpenAI SDK for OpenRouter chat completions."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openrouter_api_key:
            raise RuntimeError(
                "Please set OPENROUTER_API_KEY in .env before calling the LLM."
            )
        self._client = OpenAI(
            api_key=self._settings.openrouter_api_key,
            base_url=self._settings.openrouter_base_url,
        )

    @property
    def settings(self) -> Settings:
        return self._settings

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, str] | None = None,
        include_reasoning: bool = True,
        max_retries: int = 3,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        return self.chat_with_details(
            messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
            include_reasoning=include_reasoning,
            max_retries=max_retries,
        ).content

    def chat_with_details(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, str] | None = None,
        include_reasoning: bool = True,
        max_retries: int = 3,
    ) -> ChatCompletionResult:
        """Send a chat completion request and return detailed response metadata."""
        resolved_model = model or self._settings.default_model
        resolved_temperature = (
            temperature if temperature is not None else self._settings.default_temperature
        )

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": resolved_model,
                    "messages": messages,
                    "temperature": resolved_temperature,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if include_reasoning:
                    kwargs["extra_body"] = {"include_reasoning": True}

                response = self._client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = message.content
                if not content:
                    raise RuntimeError("LLM returned an empty response.")

                usage = response.usage
                return ChatCompletionResult(
                    content=content,
                    model=response.model or resolved_model,
                    reasoning=_extract_reasoning(message),
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                    total_tokens=usage.total_tokens if usage else None,
                )
            except Exception as exc:  # noqa: BLE001 - retry on transient API errors
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(
                    f"OpenRouter chat completion failed after {max_retries} attempts."
                ) from last_error

        raise RuntimeError("Unexpected chat completion failure.")  # pragma: no cover


def _extract_reasoning(message: Any) -> str | None:
    """Extract reasoning text from provider-specific message fields."""
    for attr in ("reasoning", "reasoning_content", "reasoning_details"):
        value = getattr(message, attr, None)
        if not value:
            continue
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)
    return None
