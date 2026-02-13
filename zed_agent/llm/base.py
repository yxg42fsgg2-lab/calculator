"""
Abstract LLM provider interface.

This is modeled after Zed's LanguageModel trait, which defines the contract
that any language model must implement to work with the agent system.

Key design decisions (from Zed):
- Streaming is the primary interface (stream_completion)
- Token counting is separate from completion
- Models declare their capabilities (tools, images, thinking)
- Completion events are a typed enum, not raw strings
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from zed_agent.core.types import (
    Message,
    Role,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolSchema,
)


@dataclass
class LLMProviderInfo:
    """Information about an LLM provider."""
    id: str
    name: str
    model_id: str
    model_name: str
    supports_tools: bool = True
    supports_images: bool = False
    supports_thinking: bool = False
    supports_streaming_tools: bool = False
    max_tokens: int = 200_000
    max_output_tokens: Optional[int] = None


@dataclass
class CompletionRequest:
    """A request to the LLM for a completion.

    This mirrors Zed's LanguageModelRequest structure.
    """
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[ToolSchema] = field(default_factory=list)
    system_prompt: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: list[str] = field(default_factory=list)
    thinking_enabled: bool = False
    thinking_effort: Optional[str] = None


class CompletionEvent:
    """Base class for completion stream events."""
    pass


@dataclass
class TextChunk(CompletionEvent):
    """A chunk of text from the model."""
    text: str


@dataclass
class ThinkingChunk(CompletionEvent):
    """A chunk of thinking/reasoning from the model."""
    text: str
    signature: Optional[str] = None


@dataclass
class ToolUseEvent(CompletionEvent):
    """The model wants to use a tool."""
    tool_call: ToolCall


@dataclass
class UsageEvent(CompletionEvent):
    """Token usage update."""
    usage: TokenUsage


@dataclass
class StopEvent(CompletionEvent):
    """The model stopped generating."""
    reason: StopReason


class LLMProvider(abc.ABC):
    """Abstract interface for language model providers.

    This is the key abstraction that decouples the agent from any specific
    LLM provider. Implement this interface to add support for a new provider.

    Modeled after Zed's `LanguageModel` trait:
    - stream_completion() -> async iterator of CompletionEvents
    - count_tokens() -> estimated token count for a request
    - info -> provider metadata and capabilities
    """

    @property
    @abc.abstractmethod
    def info(self) -> LLMProviderInfo:
        """Return provider information and capabilities."""
        ...

    @abc.abstractmethod
    async def stream_completion(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionEvent]:
        """Stream a completion from the model.

        This is the primary interface for getting completions. It yields
        a stream of CompletionEvent objects that the agent processes.

        Args:
            request: The completion request with messages, tools, etc.

        Yields:
            CompletionEvent objects (TextChunk, ThinkingChunk, ToolUseEvent, etc.)
        """
        ...

    async def count_tokens(self, request: CompletionRequest) -> int:
        """Estimate the token count for a request.

        Override this for accurate token counting. The default implementation
        provides a rough character-based estimate.
        """
        total_chars = sum(
            len(str(m.get("content", ""))) for m in request.messages
        )
        return total_chars // 4  # rough estimate
