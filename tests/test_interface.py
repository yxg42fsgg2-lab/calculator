"""Tests for the interface protocol."""

import pytest
from typing import Optional

from zed_agent.core.events import (
    ErrorEvent,
    PermissionRequestEvent,
    StopEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from zed_agent.core.types import (
    PermissionDecision,
    StopReason,
    ToolCall,
    ToolCallInfo,
    ToolKind,
    ToolPermissionRequest,
    ToolResult,
)
from zed_agent.interface.base import AgentInterface


class MockInterface(AgentInterface):
    """A mock interface for testing that records all events."""

    def __init__(self):
        self.events: list = []
        self.user_inputs: list[Optional[str]] = []
        self.permission_decisions: list[PermissionDecision] = []
        self._input_index = 0
        self._permission_index = 0

    async def on_text_event(self, event: TextEvent) -> None:
        self.events.append(("text", event))

    async def on_tool_call_event(self, event: ToolCallEvent) -> None:
        self.events.append(("tool_call", event))

    async def on_tool_result_event(self, event: ToolResultEvent) -> None:
        self.events.append(("tool_result", event))

    async def on_error_event(self, event: ErrorEvent) -> None:
        self.events.append(("error", event))

    async def on_stop_event(self, event: StopEvent) -> None:
        self.events.append(("stop", event))

    async def get_user_input(self) -> Optional[str]:
        if self._input_index < len(self.user_inputs):
            val = self.user_inputs[self._input_index]
            self._input_index += 1
            return val
        return None

    async def get_permission(
        self, event: PermissionRequestEvent
    ) -> PermissionDecision:
        self.events.append(("permission_request", event))
        if self._permission_index < len(self.permission_decisions):
            val = self.permission_decisions[self._permission_index]
            self._permission_index += 1
            return val
        return PermissionDecision.ALLOW


class TestMockInterface:
    """Tests that demonstrate how to use the MockInterface for testing."""

    @pytest.mark.asyncio
    async def test_text_events_recorded(self):
        interface = MockInterface()
        await interface.on_text_event(TextEvent(text="Hello"))
        await interface.on_text_event(TextEvent(text=" world", is_complete=True))
        assert len(interface.events) == 2
        assert interface.events[0][0] == "text"
        assert interface.events[0][1].text == "Hello"

    @pytest.mark.asyncio
    async def test_tool_events_recorded(self):
        interface = MockInterface()
        tc = ToolCall(id="1", name="read_file", input={"path": "x"}, is_input_complete=True)
        info = ToolCallInfo(id="1", name="read_file", kind=ToolKind.READ, title="Read x")
        await interface.on_tool_call_event(ToolCallEvent(tool_call=tc, info=info))

        result = ToolResult(tool_call_id="1", tool_name="read_file", content="content")
        await interface.on_tool_result_event(ToolResultEvent(result=result))

        assert len(interface.events) == 2
        assert interface.events[0][0] == "tool_call"
        assert interface.events[1][0] == "tool_result"

    @pytest.mark.asyncio
    async def test_user_input_sequence(self):
        interface = MockInterface()
        interface.user_inputs = ["first", "second", None]

        assert await interface.get_user_input() == "first"
        assert await interface.get_user_input() == "second"
        assert await interface.get_user_input() is None

    @pytest.mark.asyncio
    async def test_permission_decisions(self):
        interface = MockInterface()
        interface.permission_decisions = [PermissionDecision.ALLOW, PermissionDecision.DENY]

        request = PermissionRequestEvent(
            request=ToolPermissionRequest(
                tool_call_id="1",
                tool_name="terminal",
                tool_kind=ToolKind.EXECUTE,
                title="run command",
                description="ls -la",
            )
        )

        assert await interface.get_permission(request) == PermissionDecision.ALLOW
        assert await interface.get_permission(request) == PermissionDecision.DENY
