"""
OpenAI LLM provider.

Implements the LLMProvider interface for OpenAI's API.
Also works with any OpenAI-compatible API (Ollama, vLLM, etc.) via base_url.
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
    ThinkingChunk,
    ToolUseEvent,
    UsageEvent,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI provider.

    Also works with OpenAI-compatible APIs by setting base_url.

    Requires the `openai` package: pip install openai
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        max_tokens: int = 128_000,
        max_output_tokens: int = 16_384,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._max_output_tokens = max_output_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for OpenAIProvider. "
                    "Install it with: pip install openai"
                )
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    @property
    def info(self) -> LLMProviderInfo:
        return LLMProviderInfo(
            id="openai",
            name="OpenAI",
            model_id=self._model,
            model_name=self._model,
            supports_tools=True,
            supports_images=True,
            supports_thinking=self._model.startswith("o"),
            supports_streaming_tools=True,
            max_tokens=self._max_tokens,
            max_output_tokens=self._max_output_tokens,
        )

    async def stream_completion(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionEvent]:
        client = self._get_client()

        # Build messages list
        messages: list[dict[str, Any]] = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.extend(self._convert_messages(request.messages))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

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
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]

        try:
            stream = await client.chat.completions.create(**kwargs)
            tool_calls_in_progress: dict[int, dict[str, Any]] = {}

            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None

                if choice and choice.delta:
                    delta = choice.delta

                    # Text content
                    if delta.content:
                        yield TextChunk(text=delta.content)

                    # Tool calls
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_in_progress:
                                tool_calls_in_progress[idx] = {
                                    "id": tc.id or "",
                                    "name": getattr(tc.function, "name", "") or "",
                                    "arguments": "",
                                }
                            else:
                                if tc.id:
                                    tool_calls_in_progress[idx]["id"] = tc.id
                                if getattr(tc.function, "name", None):
                                    tool_calls_in_progress[idx]["name"] = tc.function.name

                            if getattr(tc.function, "arguments", None):
                                tool_calls_in_progress[idx]["arguments"] += tc.function.arguments

                    # Check for stop
                    if choice.finish_reason:
                        # First, emit any completed tool calls
                        for _idx, tc_data in sorted(tool_calls_in_progress.items()):
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

                # Usage info
                if chunk.usage:
                    yield UsageEvent(
                        usage=TokenUsage(
                            input_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                        )
                    )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert our message format to OpenAI's format."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            tool_results = msg.get("tool_results", [])

            if tool_results:
                for tr in tool_results:
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.get("tool_call_id", ""),
                            "content": tr.get("content", ""),
                        }
                    )
            elif tool_calls:
                msg_dict: dict[str, Any] = {
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
                result.append(msg_dict)
            else:
                result.append({"role": role, "content": content})

        return result
