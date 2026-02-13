"""
Faithful port of crates/language_model/src/request.rs.

Mirrors:
- MessageContent (enum: Text, Thinking, RedactedThinking, Image, ToolUse, ToolResult)
- LanguageModelRequestMessage
- LanguageModelRequestTool
- LanguageModelRequest
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from zed_agent.language_model.model import (
    LanguageModelToolResult,
    LanguageModelToolUse,
    Role,
)

__all__ = [
    "MessageContent",
    "LanguageModelRequestMessage",
    "LanguageModelRequestTool",
    "LanguageModelRequest",
]


# ── mirrors request.rs MessageContent enum ─────────────────

@dataclass
class TextContent:
    text: str


@dataclass
class ThinkingContent:
    text: str
    signature: Optional[str] = None


@dataclass
class RedactedThinkingContent:
    data: str


@dataclass
class ToolUseContent:
    tool_use: LanguageModelToolUse


@dataclass
class ToolResultContent:
    tool_result: LanguageModelToolResult


MessageContent = (
    TextContent | ThinkingContent | RedactedThinkingContent | ToolUseContent | ToolResultContent
)


# ── mirrors request.rs LanguageModelRequestMessage ─────────

@dataclass
class LanguageModelRequestMessage:
    role: Role
    content: list[MessageContent] = field(default_factory=list)
    cache: bool = False
    reasoning_details: Optional[Any] = None

    def string_contents(self) -> str:
        buf = ""
        for c in self.content:
            if isinstance(c, TextContent):
                buf += c.text
            elif isinstance(c, ThinkingContent):
                buf += c.text
        return buf

    def contents_empty(self) -> bool:
        return all(_content_is_empty(c) for c in self.content)


def _content_is_empty(c: MessageContent) -> bool:
    if isinstance(c, TextContent):
        return not c.text or c.text.isspace()
    if isinstance(c, ThinkingContent):
        return not c.text or c.text.isspace()
    return False


# ── mirrors request.rs LanguageModelRequestTool ────────────

@dataclass
class LanguageModelRequestTool:
    name: str
    description: str
    input_schema: Any  # serde_json::Value (dict)


# ── mirrors request.rs LanguageModelRequest ────────────────

@dataclass
class LanguageModelRequest:
    thread_id: Optional[str] = None
    prompt_id: Optional[str] = None
    messages: list[LanguageModelRequestMessage] = field(default_factory=list)
    tools: list[LanguageModelRequestTool] = field(default_factory=list)
    stop: list[str] = field(default_factory=list)
    temperature: Optional[float] = None
    thinking_allowed: bool = False
    thinking_effort: Optional[str] = None
