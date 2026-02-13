"""
Faithful port of crates/language_model/src/language_model.rs.

Key types:
- LanguageModelCompletionEvent  (the stream event enum)
- StopReason, TokenUsage
- LanguageModelToolUse, LanguageModelToolUseId
- LanguageModelToolResult, LanguageModelToolResultContent
- LanguageModel (the trait)
- LanguageModelCompletionError

Rust→Python idiom map:
  Arc<dyn LanguageModel>        →  LanguageModel (ABC instance)
  BoxFuture<'static, Result<…>> →  async method
  BoxStream<'static, …>         →  AsyncIterator
  SharedString                  →  str
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

__all__ = [
    "Role",
    "StopReason",
    "TokenUsage",
    "LanguageModelToolUseId",
    "LanguageModelToolUse",
    "LanguageModelToolResultContent",
    "LanguageModelToolResult",
    "LanguageModelCompletionEvent",
    "LanguageModelCompletionError",
    "LanguageModel",
    "LanguageModelId",
    "LanguageModelProviderId",
]


# ── mirrors crates/language_model/src/role.rs ──────────────

class Role(str, Enum):
    User = "user"
    Assistant = "assistant"
    System = "system"


# ── mirrors language_model.rs StopReason ───────────────────

class StopReason(str, Enum):
    EndTurn = "end_turn"
    MaxTokens = "max_tokens"
    ToolUse = "tool_use"
    Refusal = "refusal"


# ── mirrors language_model.rs TokenUsage ───────────────────

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


# ── mirrors language_model.rs LanguageModelToolUseId ───────

class LanguageModelToolUseId(str):
    """Newtype wrapper around str, mirrors Rust's LanguageModelToolUseId(Arc<str>)."""
    pass


# ── mirrors language_model.rs LanguageModelToolUse ─────────

@dataclass
class LanguageModelToolUse:
    id: LanguageModelToolUseId
    name: str
    raw_input: str = ""
    input: Any = field(default_factory=dict)  # serde_json::Value
    is_input_complete: bool = False
    thought_signature: Optional[str] = None


# ── mirrors request.rs LanguageModelToolResultContent ──────

@dataclass
class LanguageModelToolResultContent:
    """Text or Image result content."""
    text: str = ""
    # For images, this would carry base64 data; omitted for now.

    def is_empty(self) -> bool:
        return not self.text or self.text.isspace()

    @staticmethod
    def from_str(s: str) -> LanguageModelToolResultContent:
        return LanguageModelToolResultContent(text=s)


# ── mirrors language_model.rs LanguageModelToolResult ──────

@dataclass
class LanguageModelToolResult:
    tool_use_id: LanguageModelToolUseId
    tool_name: str
    is_error: bool = False
    content: LanguageModelToolResultContent = field(
        default_factory=LanguageModelToolResultContent
    )
    output: Optional[Any] = None  # serde_json::Value for replay


# ── mirrors language_model.rs LanguageModelCompletionEvent ─

@dataclass
class _EventBase:
    pass


@dataclass
class QueuedEvent(_EventBase):
    """Model request is queued."""
    position: int = 0


@dataclass
class StartedEvent(_EventBase):
    """Model started processing."""
    pass


@dataclass
class StartMessageEvent(_EventBase):
    """A new message block started."""
    message_id: str = ""


@dataclass
class TextEvent(_EventBase):
    text: str = ""


@dataclass
class ThinkingEvent(_EventBase):
    text: str = ""
    signature: Optional[str] = None


@dataclass
class RedactedThinkingEvent(_EventBase):
    data: str = ""


@dataclass
class ToolUseEvent(_EventBase):
    tool_use: LanguageModelToolUse = field(default_factory=lambda: LanguageModelToolUse(id=LanguageModelToolUseId(""), name=""))


@dataclass
class ToolUseJsonParseErrorEvent(_EventBase):
    id: LanguageModelToolUseId = LanguageModelToolUseId("")
    tool_name: str = ""
    raw_input: str = ""
    json_parse_error: str = ""


@dataclass
class ReasoningDetailsEvent(_EventBase):
    details: Any = None


@dataclass
class UsageUpdateEvent(_EventBase):
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class StopEvent(_EventBase):
    reason: StopReason = StopReason.EndTurn


# The union — mirrors Rust enum LanguageModelCompletionEvent
LanguageModelCompletionEvent = (
    QueuedEvent
    | StartedEvent
    | StartMessageEvent
    | TextEvent
    | ThinkingEvent
    | RedactedThinkingEvent
    | ToolUseEvent
    | ToolUseJsonParseErrorEvent
    | ReasoningDetailsEvent
    | UsageUpdateEvent
    | StopEvent
)


# ── mirrors language_model.rs LanguageModelCompletionError ─

class LanguageModelCompletionError(Exception):
    """Base for all completion errors."""
    pass


class PromptTooLargeError(LanguageModelCompletionError):
    def __init__(self, tokens: Optional[int] = None):
        self.tokens = tokens
        super().__init__(f"Prompt too large (tokens={tokens})")


class RateLimitExceededError(LanguageModelCompletionError):
    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__("Rate limit exceeded")


class ServerOverloadedError(LanguageModelCompletionError):
    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__("Server overloaded")


class AuthenticationError(LanguageModelCompletionError):
    pass


class ApiInternalServerError(LanguageModelCompletionError):
    pass


# ── ID newtypes ────────────────────────────────────────────

class LanguageModelId(str):
    """Mirrors LanguageModelId(SharedString)."""
    pass


class LanguageModelProviderId(str):
    """Mirrors LanguageModelProviderId(SharedString)."""
    pass


# ── mirrors language_model.rs trait LanguageModel ──────────

class LanguageModel(abc.ABC):
    """Faithful port of the Rust LanguageModel trait.

    Every method here corresponds to a method on the Rust trait.
    """

    @abc.abstractmethod
    def id(self) -> LanguageModelId:
        ...

    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def provider_id(self) -> LanguageModelProviderId:
        ...

    def provider_name(self) -> str:
        return str(self.provider_id())

    def telemetry_id(self) -> str:
        return str(self.id())

    def supports_thinking(self) -> bool:
        return False

    @abc.abstractmethod
    def supports_images(self) -> bool:
        ...

    @abc.abstractmethod
    def supports_tools(self) -> bool:
        ...

    def supports_streaming_tools(self) -> bool:
        return False

    def max_token_count(self) -> int:
        return 200_000

    def max_output_tokens(self) -> Optional[int]:
        return None

    @abc.abstractmethod
    async def stream_completion(
        self,
        request: "LanguageModelRequest",
    ) -> AsyncIterator[LanguageModelCompletionEvent]:
        """Mirrors stream_completion(&self, request, cx) -> BoxFuture<BoxStream<…>>."""
        ...
        # Must be an async generator in implementations: yield events

    async def count_tokens(self, request: "LanguageModelRequest") -> int:
        """Estimate token count. Default: character / 4."""
        total = sum(len(m.string_contents()) for m in request.messages)
        return total // 4
