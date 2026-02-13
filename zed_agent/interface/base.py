"""
Abstract interface protocol.

THIS IS THE KEY FILE FOR UI IMPLEMENTORS.

To plug the agent backend into any UI, you implement this single abstract class.
The interface has two responsibilities:

1. DISPLAY: Handle events from the agent (text streaming, tool calls, etc.)
2. INPUT: Provide user messages and permission decisions back to the agent

The event handler methods correspond to the events the agent emits.
Each method receives a typed event object and should update the UI accordingly.

MINIMAL IMPLEMENTATION:
    You only need to implement:
    - on_text_event(): Display streaming text
    - on_tool_call_event(): Show tool is being called
    - on_tool_result_event(): Show tool result
    - on_error_event(): Show errors
    - on_stop_event(): Mark turn as complete
    - get_user_input(): Get text from the user
    - get_permission(): Ask user for tool permission

    All other methods have sensible defaults.
"""

from __future__ import annotations

import abc
from typing import Optional

from zed_agent.core.events import (
    ErrorEvent,
    PermissionRequestEvent,
    RetryEvent,
    StatusEvent,
    StopEvent,
    TextEvent,
    ThinkingEvent,
    TitleEvent,
    ToolCallEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
    ToolResultEvent,
)
from zed_agent.core.types import PermissionDecision


class AgentInterface(abc.ABC):
    """Abstract interface that any UI must implement to work with the agent.

    This is the pluggable boundary between the backend and frontend.
    Implement this class to connect the agent to any UI framework:
    - Terminal/CLI
    - Web (Flask, FastAPI, Django)
    - Desktop (Tkinter, Qt, Electron)
    - Mobile
    - Chat platforms (Slack, Discord)
    - IDE extensions
    - Test harnesses

    Event Flow:
        Agent -> Interface (display events)
        Interface -> Agent (user input, permissions)

    Example:
        class MyWebInterface(AgentInterface):
            def __init__(self, websocket):
                self.ws = websocket

            async def on_text_event(self, event):
                await self.ws.send_json({"type": "text", "text": event.text})

            async def get_user_input(self):
                msg = await self.ws.receive_json()
                return msg["text"]

            # ... implement remaining abstract methods ...
    """

    # ──────────────────────────────────────────────────────────
    # DISPLAY METHODS - Called by the agent to update the UI
    # ──────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def on_text_event(self, event: TextEvent) -> None:
        """Agent is streaming text content.

        Called repeatedly as text chunks arrive. Use event.is_complete
        to know when the full message is done.
        """
        ...

    async def on_thinking_event(self, event: ThinkingEvent) -> None:
        """Agent is streaming thinking/reasoning content.

        Optional - defaults to ignoring thinking content.
        Override to display the model's chain-of-thought.
        """
        pass

    @abc.abstractmethod
    async def on_tool_call_event(self, event: ToolCallEvent) -> None:
        """Agent wants to call a tool.

        Display the tool call to the user. The event includes:
        - tool_call: The raw tool call data
        - info: UI-friendly metadata (name, kind, title)
        """
        ...

    async def on_tool_start_event(self, event: ToolCallStartEvent) -> None:
        """A tool has started executing. Optional."""
        pass

    async def on_tool_progress_event(self, event: ToolCallProgressEvent) -> None:
        """Progress update during tool execution. Optional."""
        pass

    @abc.abstractmethod
    async def on_tool_result_event(self, event: ToolResultEvent) -> None:
        """Tool has finished executing.

        Display the result to the user. Check result.is_error
        for error handling.
        """
        ...

    @abc.abstractmethod
    async def on_error_event(self, event: ErrorEvent) -> None:
        """An error occurred. Display to the user."""
        ...

    @abc.abstractmethod
    async def on_stop_event(self, event: StopEvent) -> None:
        """Agent has stopped generating.

        Mark the turn as complete. The event includes the stop reason
        and optional token usage statistics.
        """
        ...

    async def on_retry_event(self, event: RetryEvent) -> None:
        """Agent is retrying after an error. Optional."""
        pass

    async def on_title_event(self, event: TitleEvent) -> None:
        """Thread title was generated. Optional."""
        pass

    async def on_status_event(self, event: StatusEvent) -> None:
        """Agent status changed. Optional."""
        pass

    # ──────────────────────────────────────────────────────────
    # INPUT METHODS - Called by the agent to get user input
    # ──────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def get_user_input(self) -> Optional[str]:
        """Get a message from the user.

        Returns:
            The user's message text, or None to end the conversation.
        """
        ...

    @abc.abstractmethod
    async def get_permission(
        self, event: PermissionRequestEvent
    ) -> PermissionDecision:
        """Ask the user for permission to execute a tool.

        This is called for tools that have requires_permission=True
        (e.g., terminal commands, file writes).

        Args:
            event: Details about the permission request, including
                   tool name, kind, title, and description.

        Returns:
            The user's decision (allow, deny, allow_always, deny_always).
        """
        ...

    # ──────────────────────────────────────────────────────────
    # LIFECYCLE METHODS - Optional hooks
    # ──────────────────────────────────────────────────────────

    async def on_session_start(self) -> None:
        """Called when a new agent session starts. Optional."""
        pass

    async def on_session_end(self) -> None:
        """Called when the agent session ends. Optional."""
        pass
