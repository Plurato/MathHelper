"""Reusable base class for MathCoach agents."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from mathcoach.agents.trace import AgentRunResult, AgentRunTrace, LLMStepTrace
from mathcoach.llm.chat_result import ChatCompletionResult
from mathcoach.llm.openrouter_client import OpenRouterClient
from mathcoach.utils.json_parser import extract_json
from mathcoach.utils.math_format import repair_latex_in_payload

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput", bound=BaseModel)


class BaseAgent(ABC, Generic[TInput, TOutput]):
    """Abstract agent that calls an LLM and validates structured JSON output."""

    name: str
    system_prompt: str
    output_schema: type[TOutput]
    max_parse_retries: int = 2

    def __init__(
        self,
        *,
        llm_client: OpenRouterClient | None = None,
        model: str | None = None,
        temperature: float | None = None,
        include_reasoning: bool = True,
    ) -> None:
        self._llm_client = llm_client or OpenRouterClient()
        self._model = model
        self._temperature = temperature
        self._include_reasoning = include_reasoning

    @abstractmethod
    def build_user_prompt(self, input_data: TInput) -> str:
        """Build the user prompt from typed input."""

    def run(self, input_data: TInput) -> TOutput:
        """Execute the agent and return validated structured output."""
        return self.run_with_trace(input_data).output

    def run_with_trace(self, input_data: TInput) -> AgentRunResult[TOutput]:
        """Execute the agent and return output plus an execution trace."""
        user_prompt = self.build_user_prompt(input_data)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        trace = AgentRunTrace(
            agent_name=self.name,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )
        steps: list[LLMStepTrace] = []

        last_error: Exception | None = None
        for attempt in range(self.max_parse_retries + 1):
            completion = self._llm_client.chat_with_details(
                messages,
                model=self._model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                include_reasoning=self._include_reasoning,
            )
            try:
                payload = extract_json(completion.content)
                payload = repair_latex_in_payload(payload)
                output = self.output_schema.model_validate(payload)
                steps.append(
                    _build_step_trace(
                        attempt=attempt + 1,
                        completion=completion,
                        parsed_payload=payload,
                        status="success",
                    )
                )
                trace = AgentRunTrace(
                    agent_name=trace.agent_name,
                    system_prompt=trace.system_prompt,
                    user_prompt=trace.user_prompt,
                    steps=steps,
                )
                return AgentRunResult(output=output, trace=trace)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                steps.append(
                    _build_step_trace(
                        attempt=attempt + 1,
                        completion=completion,
                        parsed_payload=_safe_extract_payload(completion.content),
                        status="retry" if attempt < self.max_parse_retries else "failed",
                        validation_error=str(exc),
                    )
                )
                if attempt >= self.max_parse_retries:
                    break
                messages.append(
                    {
                        "role": "assistant",
                        "content": completion.content,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": self._build_repair_prompt(exc),
                    }
                )

        trace = AgentRunTrace(
            agent_name=trace.agent_name,
            system_prompt=trace.system_prompt,
            user_prompt=trace.user_prompt,
            steps=steps,
        )
        raise RuntimeError(
            f"{self.name} failed to produce valid structured output."
        ) from last_error

    def _build_repair_prompt(self, error: Exception) -> str:
        """Ask the model to fix invalid JSON output."""
        return (
            "Your previous response could not be parsed or validated.\n"
            f"Error: {error}\n"
            "Return ONLY a valid JSON object that matches the required schema. "
            "Do not include markdown fences or extra commentary."
        )


def _build_step_trace(
    *,
    attempt: int,
    completion: ChatCompletionResult,
    parsed_payload: dict[str, Any] | None,
    status: str,
    validation_error: str | None = None,
) -> LLMStepTrace:
    return LLMStepTrace(
        attempt=attempt,
        raw_response=completion.content,
        reasoning=completion.reasoning,
        parsed_payload=parsed_payload,
        validation_error=validation_error,
        status=status,  # type: ignore[arg-type]
        model=completion.model,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        total_tokens=completion.total_tokens,
    )


def _safe_extract_payload(raw_response: str) -> dict[str, Any] | None:
    if not raw_response.strip():
        return None
    try:
        return repair_latex_in_payload(extract_json(raw_response))
    except ValueError:
        return None
