"""
Anthropic (Claude) LLM provider.

Implements the LLMProvider interface for Anthropic's API.
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


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider.

    Requires the `anthropic` package: pip install anthropic
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 200_000,
        max_output_tokens: int = 16_384,
    ):
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._max_output_tokens = max_output_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for AnthropicProvider. "
                    "Install it with: pip install anthropic"
                )
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    @property
    def info(self) -> LLMProviderInfo:
        return LLMProviderInfo(
            id="anthropic",
            name="Anthropic",
            model_id=self._model,
            model_name=self._model,
            supports_tools=True,
            supports_images=True,
            supports_thinking="thinking" in self._model or "claude-3" in self._model,
            supports_streaming_tools=True,
            max_tokens=self._max_tokens,
            max_output_tokens=self._max_output_tokens,
        )

    async def stream_completion(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionEvent]:
        client = self._get_client()

        # Build the API request
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_tokens or self._max_output_tokens,
            "stream": True,
        }

        if request.system_prompt:
            kwargs["system"] = request.system_prompt

        # Convert messages
        kwargs["messages"] = self._convert_messages(request.messages)

        # Add tools if provided
        if request.tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences

        # Handle thinking/extended thinking
        if request.thinking_enabled:
            budget = min(
                (request.max_tokens or self._max_output_tokens) - 1000,
                32000,
            )
            if budget > 0:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }

        try:
            async with client.messages.stream(**kwargs) as stream:
                current_tool_call: Optional[dict] = None

                async for event in stream:
                    event_type = event.type

                    if event_type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool_call = {
                                "id": block.id,
                                "name": block.name,
                                "raw_input": "",
                            }
                        elif block.type == "thinking":
                            pass  # Will get content via delta events

                    elif event_type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield TextChunk(text=delta.text)
                        elif delta.type == "thinking_delta":
                            yield ThinkingChunk(text=delta.thinking)
                        elif delta.type == "input_json_delta":
                            if current_tool_call is not None:
                                current_tool_call["raw_input"] += delta.partial_json

                    elif event_type == "content_block_stop":
                        if current_tool_call is not None:
                            try:
                                parsed_input = json.loads(
                                    current_tool_call["raw_input"]
                                )
                            except json.JSONDecodeError:
                                parsed_input = {}

                            yield ToolUseEvent(
                                tool_call=ToolCall(
                                    id=current_tool_call["id"],
                                    name=current_tool_call["name"],
                                    raw_input=current_tool_call["raw_input"],
                                    input=parsed_input,
                                    is_input_complete=True,
                                )
                            )
                            current_tool_call = None

                    elif event_type == "message_delta":
                        if hasattr(event, "usage") and event.usage:
                            yield UsageEvent(
                                usage=TokenUsage(
                                    output_tokens=getattr(
                                        event.usage, "output_tokens", 0
                                    ),
                                )
                            )
                        stop = getattr(event.delta, "stop_reason", None)
                        if stop:
                            reason_map = {
                                "end_turn": StopReason.END_TURN,
                                "max_tokens": StopReason.MAX_TOKENS,
                                "tool_use": StopReason.TOOL_USE,
                            }
                            yield StopEvent(
                                reason=reason_map.get(stop, StopReason.END_TURN)
                            )

                    elif event_type == "message_start":
                        msg = getattr(event, "message", None)
                        if msg and hasattr(msg, "usage"):
                            yield UsageEvent(
                                usage=TokenUsage(
                                    input_tokens=getattr(
                                        msg.usage, "input_tokens", 0
                                    ),
                                    cache_creation_input_tokens=getattr(
                                        msg.usage,
                                        "cache_creation_input_tokens",
                                        0,
                                    )
                                    or 0,
                                    cache_read_input_tokens=getattr(
                                        msg.usage, "cache_read_input_tokens", 0
                                    )
                                    or 0,
                                )
                            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert our message format to Anthropic's format."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            tool_results = msg.get("tool_results", [])

            if tool_results:
                # Tool results go as user messages
                blocks = []
                for tr in tool_results:
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.get("tool_call_id", ""),
                            "content": tr.get("content", ""),
                            "is_error": tr.get("is_error", False),
                        }
                    )
                result.append({"role": "user", "content": blocks})
            elif tool_calls:
                # Tool calls go as assistant messages
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("name", ""),
                            "input": tc.get("input", {}),
                        }
                    )
                result.append({"role": "assistant", "content": blocks})
            else:
                result.append({"role": role, "content": content})

        return result

    async def count_tokens(self, request: CompletionRequest) -> int:
        """Use Anthropic's token counting API."""
        try:
            client = self._get_client()
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": self._convert_messages(request.messages),
            }
            if request.system_prompt:
                kwargs["system"] = request.system_prompt
            if request.tools:
                kwargs["tools"] = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    }
                    for t in request.tools
                ]
            result = await client.messages.count_tokens(**kwargs)
            return result.input_tokens
        except Exception:
            return await super().count_tokens(request)
