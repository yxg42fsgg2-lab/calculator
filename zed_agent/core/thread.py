"""
Conversation thread - manages the message history and context.

Ported from Zed's Thread struct. This is the stateful conversation
context that tracks messages, tool calls/results, and token usage.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from zed_agent.core.types import (
    AgentMessage,
    Message,
    Role,
    ToolCall,
    ToolResult,
    TokenUsage,
    UserMessage,
)


class Thread:
    """A conversation thread with message history.

    Manages the ordered sequence of messages between user and agent,
    including tool call/result pairs. Converts to the format needed
    by LLM providers.
    """

    def __init__(self, thread_id: Optional[str] = None):
        self.id = thread_id or str(uuid.uuid4())
        self.title: Optional[str] = None
        self.messages: list[Message] = []
        self.cumulative_token_usage = TokenUsage()
        self._pending_tool_calls: list[ToolCall] = []
        self._pending_tool_results: list[ToolResult] = []

    def add_user_message(self, content: str) -> Message:
        """Add a user message to the thread."""
        msg = Message(role=Role.USER, content=content)
        self.messages.append(msg)
        return msg

    def add_agent_text(self, content: str, thinking: str = "") -> Message:
        """Add or append to the current agent message."""
        # If the last message is an incomplete agent message, append to it
        if self.messages and self.messages[-1].role == Role.ASSISTANT:
            self.messages[-1].content += content
            if thinking:
                self.messages[-1].thinking += thinking
            return self.messages[-1]

        msg = Message(role=Role.ASSISTANT, content=content, thinking=thinking)
        self.messages.append(msg)
        return msg

    def add_agent_tool_calls(self, tool_calls: list[ToolCall]) -> Message:
        """Add tool calls to the agent's message."""
        if self.messages and self.messages[-1].role == Role.ASSISTANT:
            self.messages[-1].tool_calls.extend(tool_calls)
            return self.messages[-1]

        msg = Message(role=Role.ASSISTANT, tool_calls=tool_calls)
        self.messages.append(msg)
        return msg

    def add_tool_results(self, results: list[ToolResult]) -> Message:
        """Add tool results as a user message (per LLM protocol)."""
        msg = Message(role=Role.USER, tool_results=results)
        self.messages.append(msg)
        return msg

    def add_resume_message(self) -> Message:
        """Add a resume message to continue after tool use."""
        msg = Message(role=Role.USER, content="Continue where you left off.")
        self.messages.append(msg)
        return msg

    def update_token_usage(self, usage: TokenUsage) -> None:
        """Update cumulative token usage."""
        self.cumulative_token_usage = self.cumulative_token_usage + usage

    def to_llm_messages(self) -> list[dict[str, Any]]:
        """Convert thread to the format expected by LLM providers."""
        result: list[dict[str, Any]] = []

        for msg in self.messages:
            if msg.tool_results:
                result.append(
                    {
                        "role": "user",
                        "tool_results": [
                            {
                                "tool_call_id": tr.tool_call_id,
                                "tool_name": tr.tool_name,
                                "content": tr.content,
                                "is_error": tr.is_error,
                            }
                            for tr in msg.tool_results
                        ],
                    }
                )
            elif msg.tool_calls:
                result.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.input,
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
            else:
                result.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

        return result

    def is_empty(self) -> bool:
        return len(self.messages) == 0

    @property
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None

    def truncate_to(self, message_index: int) -> None:
        """Truncate the thread to the given message index."""
        self.messages = self.messages[: message_index + 1]
