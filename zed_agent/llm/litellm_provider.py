"""
LiteLLM universal provider.

Uses the litellm library to support 100+ LLM providers through a single
interface. This is the easiest way to get started - just specify a model
string and litellm handles the rest.

Supported providers include: OpenAI, Anthropic, Google, Mistral, Cohere,
Ollama, Azure, AWS Bedrock, Hugging Face, and many more.

See https://docs.litellm.ai/docs/providers for the full list.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

from zed_agent.core.types import StopReason, TokenUsage, ToolCall
from zed_agent.llm.base import (
    CompletionEvent,
    CompletionRequest,
    LLMProvider,
    LLMProviderInfo,
    StopEvent,
    TextChunk,
    ToolUseEvent,
    UsageEvent,
)

logger = logging.getLogger(__name__)


class LiteLLMProvider(LLMProvider):
    """Universal LLM provider using litellm.

    Requires the `litellm` package: pip install litellm

    Usage:
        # Anthropic
        provider = LiteLLMProvider(model="claude-sonnet-4-20250514")

        # OpenAI
        provider = LiteLLMProvider(model="gpt-4o")

        # Ollama (local)
        provider = LiteLLMProvider(model="ollama/llama3")

        # Google
        provider = LiteLLMProvider(model="gemini/gemini-pro")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 200_000,
        max_output_tokens: int = 16_384,
        **kwargs: Any,
    ):
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._max_tokens = max_tokens
        self._max_output_tokens = max_output_tokens
        self._extra_kwargs = kwargs

    @property
    def info(self) -> LLMProviderInfo:
        provider_name = self._model.split("/")[0] if "/" in self._model else "litellm"
        return LLMProviderInfo(
            id=f"litellm-{provider_name}",
            name=f"LiteLLM ({provider_name})",
            model_id=self._model,
            model_name=self._model,
            supports_tools=True,
            supports_images=True,
            supports_thinking=False,
            supports_streaming_tools=True,
            max_tokens=self._max_tokens,
            max_output_tokens=self._max_output_tokens,
        )

    async def stream_completion(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionEvent]:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "The 'litellm' package is required for LiteLLMProvider. "
                "Install it with: pip install litellm"
            )

        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            tool_results = msg.get("tool_results", [])

            if tool_results:
                for tr in tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.get("tool_call_id", ""),
                            "content": tr.get("content", ""),
                        }
                    )
            elif tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": json.dumps(tc.get("input", {})),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
            else:
                messages.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            **self._extra_kwargs,
        }

        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]

        try:
            response = await litellm.acompletion(**kwargs)
            tool_calls_in_progress: dict[int, dict[str, Any]] = {}

            async for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None
                if choice and choice.delta:
                    delta = choice.delta
                    if delta.content:
                        yield TextChunk(text=delta.content)

                    if getattr(delta, "tool_calls", None):
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_in_progress:
                                tool_calls_in_progress[idx] = {
                                    "id": tc.id or "",
                                    "name": getattr(tc.function, "name", "") or "",
                                    "arguments": "",
                                }
                            if tc.id:
                                tool_calls_in_progress[idx]["id"] = tc.id
                            if getattr(tc.function, "name", None):
                                tool_calls_in_progress[idx]["name"] = tc.function.name
                            if getattr(tc.function, "arguments", None):
                                tool_calls_in_progress[idx]["arguments"] += (
                                    tc.function.arguments
                                )

                    if choice.finish_reason:
                        for _idx, tc_data in sorted(
                            tool_calls_in_progress.items()
                        ):
                            try:
                                parsed = json.loads(tc_data["arguments"])
                            except json.JSONDecodeError:
                                parsed = {}
                            yield ToolUseEvent(
                                tool_call=ToolCall(
                                    id=tc_data["id"],
                                    name=tc_data["name"],
                                    raw_input=tc_data["arguments"],
                                    input=parsed,
                                    is_input_complete=True,
                                )
                            )
                        tool_calls_in_progress.clear()

                        reason_map = {
                            "stop": StopReason.END_TURN,
                            "length": StopReason.MAX_TOKENS,
                            "tool_calls": StopReason.TOOL_USE,
                        }
                        yield StopEvent(
                            reason=reason_map.get(
                                choice.finish_reason, StopReason.END_TURN
                            )
                        )

                if hasattr(chunk, "usage") and chunk.usage:
                    yield UsageEvent(
                        usage=TokenUsage(
                            input_tokens=getattr(
                                chunk.usage, "prompt_tokens", 0
                            )
                            or 0,
                            output_tokens=getattr(
                                chunk.usage, "completion_tokens", 0
                            )
                            or 0,
                        )
                    )

        except Exception as e:
            logger.error(f"LiteLLM API error: {e}")
            raise
