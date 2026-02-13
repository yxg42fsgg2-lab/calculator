"""Anthropic provider — mirrors crates/anthropic/."""
from __future__ import annotations
import json, logging
from typing import Any, AsyncIterator, Optional
from zed_agent.language_model.model import (
    LanguageModel, LanguageModelId, LanguageModelProviderId,
    LanguageModelCompletionEvent, LanguageModelToolUse, LanguageModelToolUseId,
    StopReason, TokenUsage,
    TextEvent, ThinkingEvent, ToolUseEvent, UsageUpdateEvent, StopEvent, StartMessageEvent,
)
from zed_agent.language_model.request import (
    LanguageModelRequest, LanguageModelRequestMessage,
    TextContent, ThinkingContent, RedactedThinkingContent,
    ToolUseContent, ToolResultContent, MessageContent,
)

logger = logging.getLogger(__name__)

class AnthropicModel(LanguageModel):
    """Anthropic Claude. Requires `pip install anthropic`."""
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", max_output: int = 16384):
        self._api_key = api_key
        self._model = model
        self._max_output = max_output
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    def id(self) -> LanguageModelId:
        return LanguageModelId(self._model)
    def name(self) -> str:
        return self._model
    def provider_id(self) -> LanguageModelProviderId:
        return LanguageModelProviderId("anthropic")
    def supports_images(self) -> bool:
        return True
    def supports_tools(self) -> bool:
        return True
    def supports_streaming_tools(self) -> bool:
        return True
    def max_output_tokens(self) -> Optional[int]:
        return self._max_output

    async def stream_completion(self, request: LanguageModelRequest) -> AsyncIterator[LanguageModelCompletionEvent]:
        client = self._get_client()
        kwargs: dict[str, Any] = {"model": self._model, "max_tokens": self._max_output, "stream": True}

        # System prompt
        system_parts = []
        api_messages = []
        for msg in request.messages:
            if msg.role.value == "system":
                system_parts.append(msg.string_contents())
            else:
                api_messages.extend(_convert_message(msg))
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        kwargs["messages"] = api_messages

        if request.tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in request.tools]
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.thinking_allowed:
            budget = min(self._max_output - 1000, 32000)
            if budget > 0:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        current_tc: Optional[dict] = None
        async with client.messages.stream(**kwargs) as stream:
            async for ev in stream:
                t = ev.type
                if t == "message_start":
                    msg = getattr(ev, "message", None)
                    if msg and hasattr(msg, "usage"):
                        yield UsageUpdateEvent(usage=TokenUsage(
                            input_tokens=getattr(msg.usage, "input_tokens", 0),
                            cache_creation_input_tokens=getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
                            cache_read_input_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                        ))
                elif t == "content_block_start":
                    b = ev.content_block
                    if b.type == "tool_use":
                        current_tc = {"id": b.id, "name": b.name, "raw": ""}
                elif t == "content_block_delta":
                    d = ev.delta
                    if d.type == "text_delta":
                        yield TextEvent(text=d.text)
                    elif d.type == "thinking_delta":
                        yield ThinkingEvent(text=d.thinking)
                    elif d.type == "input_json_delta" and current_tc is not None:
                        current_tc["raw"] += d.partial_json
                elif t == "content_block_stop":
                    if current_tc is not None:
                        try:
                            parsed = json.loads(current_tc["raw"])
                        except json.JSONDecodeError:
                            parsed = {}
                        yield ToolUseEvent(tool_use=LanguageModelToolUse(
                            id=LanguageModelToolUseId(current_tc["id"]),
                            name=current_tc["name"], raw_input=current_tc["raw"],
                            input=parsed, is_input_complete=True,
                        ))
                        current_tc = None
                elif t == "message_delta":
                    if hasattr(ev, "usage") and ev.usage:
                        yield UsageUpdateEvent(usage=TokenUsage(output_tokens=getattr(ev.usage, "output_tokens", 0)))
                    stop = getattr(ev.delta, "stop_reason", None)
                    if stop:
                        m = {"end_turn": StopReason.EndTurn, "max_tokens": StopReason.MaxTokens, "tool_use": StopReason.ToolUse}
                        yield StopEvent(reason=m.get(stop, StopReason.EndTurn))


def _convert_message(msg: LanguageModelRequestMessage) -> list[dict[str, Any]]:
    """Convert our message format to Anthropic API format."""
    results = []
    role = msg.role.value
    # Check for tool results
    tool_result_blocks = [c for c in msg.content if isinstance(c, ToolResultContent)]
    tool_use_blocks = [c for c in msg.content if isinstance(c, ToolUseContent)]
    text_blocks = [c for c in msg.content if isinstance(c, TextContent)]

    if tool_result_blocks:
        blocks = []
        for trc in tool_result_blocks:
            tr = trc.tool_result
            blocks.append({"type": "tool_result", "tool_use_id": str(tr.tool_use_id),
                           "content": tr.content.text, "is_error": tr.is_error})
        results.append({"role": "user", "content": blocks})
    elif tool_use_blocks:
        blocks: list[Any] = []
        for tc in text_blocks:
            blocks.append({"type": "text", "text": tc.text})
        for tuc in tool_use_blocks:
            tu = tuc.tool_use
            blocks.append({"type": "tool_use", "id": str(tu.id), "name": tu.name, "input": tu.input})
        results.append({"role": "assistant", "content": blocks})
    else:
        text = msg.string_contents()
        results.append({"role": role, "content": text})
    return results
