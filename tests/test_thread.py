"""Tests for the conversation thread."""

import pytest

from zed_agent.core.thread import Thread
from zed_agent.core.types import (
    Message,
    Role,
    ToolCall,
    ToolResult,
    TokenUsage,
)


class TestThread:
    def test_empty_thread(self):
        thread = Thread()
        assert thread.is_empty()
        assert thread.last_message is None

    def test_add_user_message(self):
        thread = Thread()
        msg = thread.add_user_message("Hello")
        assert not thread.is_empty()
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert thread.last_message is msg

    def test_add_agent_text(self):
        thread = Thread()
        thread.add_user_message("Hello")
        msg = thread.add_agent_text("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"

    def test_agent_text_appends(self):
        thread = Thread()
        thread.add_user_message("Hello")
        thread.add_agent_text("Hi ")
        thread.add_agent_text("there!")
        assert len(thread.messages) == 2
        assert thread.messages[1].content == "Hi there!"

    def test_tool_calls_and_results(self):
        thread = Thread()
        thread.add_user_message("Read main.py")
        thread.add_agent_tool_calls([
            ToolCall(id="tc1", name="read_file", input={"path": "main.py"}, is_input_complete=True)
        ])
        thread.add_tool_results([
            ToolResult(tool_call_id="tc1", tool_name="read_file", content="file content")
        ])
        assert len(thread.messages) == 3

    def test_to_llm_messages(self):
        thread = Thread()
        thread.add_user_message("Hello")
        thread.add_agent_text("Hi there!")

        messages = thread.to_llm_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    def test_to_llm_messages_with_tools(self):
        thread = Thread()
        thread.add_user_message("Read main.py")
        thread.add_agent_tool_calls([
            ToolCall(id="tc1", name="read_file", input={"path": "main.py"}, is_input_complete=True)
        ])
        thread.add_tool_results([
            ToolResult(tool_call_id="tc1", tool_name="read_file", content="content")
        ])

        messages = thread.to_llm_messages()
        assert len(messages) == 3
        assert "tool_calls" in messages[1]
        assert "tool_results" in messages[2]

    def test_token_usage_tracking(self):
        thread = Thread()
        thread.update_token_usage(TokenUsage(input_tokens=100, output_tokens=50))
        thread.update_token_usage(TokenUsage(input_tokens=200, output_tokens=100))
        assert thread.cumulative_token_usage.input_tokens == 300
        assert thread.cumulative_token_usage.output_tokens == 150
        assert thread.cumulative_token_usage.total_tokens == 450

    def test_truncate(self):
        thread = Thread()
        thread.add_user_message("First")
        thread.add_agent_text("Response 1")
        thread.add_user_message("Second")
        thread.add_agent_text("Response 2")

        assert len(thread.messages) == 4
        thread.truncate_to(1)
        assert len(thread.messages) == 2
        assert thread.last_message.content == "Response 1"
