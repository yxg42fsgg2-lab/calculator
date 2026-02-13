"""
Agent events - the communication protocol between backend and frontend.

These events flow FROM the agent TO the interface. Any UI that wants to
display agent activity must handle these events.

This is modeled after Zed's ThreadEvent enum, which provides a clean
stream of events that the UI layer processes to update its display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from zed_agent.core.types import (
    StopReason,
    TokenUsage,
    ToolCall,
    ToolCallInfo,
    ToolKind,
    ToolPermissionRequest,
    ToolResult,
)


@dataclass
class AgentEvent:
    """Base class for all events emitted by the agent."""
    pass


@dataclass
class TextEvent(AgentEvent):
    """Agent is streaming text content."""
    text: str
    is_complete: bool = False


@dataclass
class ThinkingEvent(AgentEvent):
    """Agent is streaming thinking/reasoning content."""
    text: str
    is_complete: bool = False


@dataclass
class ToolCallEvent(AgentEvent):
    """Agent wants to call a tool.

    Emitted when the model requests a tool call. The frontend can use the
    ToolCallInfo to render appropriate UI (file icon for read, terminal
    icon for execute, etc.)
    """
    tool_call: ToolCall
    info: ToolCallInfo


@dataclass
class ToolCallStartEvent(AgentEvent):
    """Tool execution has started."""
    tool_call_id: str
    tool_name: str
    title: str = ""


@dataclass
class ToolCallProgressEvent(AgentEvent):
    """Progress update during tool execution."""
    tool_call_id: str
    content: str = ""
    locations: list[str] = field(default_factory=list)


@dataclass
class ToolResultEvent(AgentEvent):
    """Tool has finished executing."""
    result: ToolResult


@dataclass
class PermissionRequestEvent(AgentEvent):
    """Agent needs user permission to proceed with a tool call.

    The frontend must respond by calling agent.grant_permission() or
    agent.deny_permission() with the request_id.
    """
    request: ToolPermissionRequest


@dataclass
class RetryEvent(AgentEvent):
    """Agent is retrying after an error."""
    attempt: int
    max_attempts: int
    delay_seconds: float
    reason: str = ""


@dataclass
class ErrorEvent(AgentEvent):
    """An error occurred during agent processing."""
    error: str
    is_retryable: bool = False


@dataclass
class StopEvent(AgentEvent):
    """Agent has stopped generating."""
    reason: StopReason
    token_usage: Optional[TokenUsage] = None


@dataclass
class TitleEvent(AgentEvent):
    """Thread title was generated or updated."""
    title: str


@dataclass
class StatusEvent(AgentEvent):
    """Agent status change (e.g., 'thinking', 'executing tool', etc.)."""
    status: str
    detail: str = ""
