"""
Core data types for the agent system.

These types are the shared vocabulary between the backend and any frontend.
They are intentionally simple, serializable dataclasses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(str, Enum):
    """Message role in the conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class StopReason(str, Enum):
    """Why the agent stopped generating."""
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    CANCELLED = "cancelled"
    ERROR = "error"


class ToolKind(str, Enum):
    """Category of tool - helps UIs render appropriate affordances."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    SEARCH = "search"
    NAVIGATE = "navigate"
    OTHER = "other"


@dataclass
class TokenUsage:
    """Token usage statistics for a completion."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
        )


@dataclass
class ToolCallInfo:
    """Metadata about a tool call for the UI."""
    id: str
    name: str
    kind: ToolKind
    title: str = ""
    input_preview: str = ""


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""
    id: str
    name: str
    raw_input: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    is_input_complete: bool = False

    @staticmethod
    def new_id() -> str:
        return f"toolu_{uuid.uuid4().hex[:24]}"


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_call_id: str
    tool_name: str
    content: str = ""
    is_error: bool = False
    output: Any = None


@dataclass
class UserMessage:
    """A message from the user."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""

    @property
    def role(self) -> Role:
        return Role.USER


@dataclass
class AgentMessage:
    """A message from the agent, which may include text and tool calls."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    @property
    def role(self) -> Role:
        return Role.ASSISTANT


@dataclass
class Message:
    """A conversation message - either from user or agent."""
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    thinking: str = ""


@dataclass
class ToolPermissionRequest:
    """Asks the frontend whether a tool call should proceed."""
    tool_call_id: str
    tool_name: str
    tool_kind: ToolKind
    title: str
    description: str


class PermissionDecision(str, Enum):
    """Result of a permission check."""
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_ALWAYS = "allow_always"
    DENY_ALWAYS = "deny_always"


@dataclass
class ToolSchema:
    """Schema describing a tool's capabilities for the LLM."""
    name: str
    description: str
    input_schema: dict[str, Any]
