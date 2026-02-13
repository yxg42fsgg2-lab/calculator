#!/usr/bin/env python3
"""
Example: Using the agent programmatically (without an interactive loop).

This shows how to drive the agent from code, useful for:
- Automated tasks
- CI/CD pipelines
- Testing
- Batch processing
"""

import asyncio
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zed_agent import Agent, AgentConfig
from zed_agent.core.events import (
    ErrorEvent,
    PermissionRequestEvent,
    StopEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from zed_agent.core.types import PermissionDecision
from zed_agent.interface.base import AgentInterface


class ProgrammaticInterface(AgentInterface):
    """An interface that collects results for programmatic use.

    Instead of displaying to a terminal, this collects all output
    for later processing.
    """

    def __init__(self, auto_approve: bool = True):
        self.full_response = ""
        self.tool_calls_made: list[dict] = []
        self.errors: list[str] = []
        self.auto_approve = auto_approve
        self._input_provided = False
        self._user_message: Optional[str] = None

    def set_message(self, message: str):
        """Set the message to send."""
        self._user_message = message
        self._input_provided = False

    async def on_text_event(self, event: TextEvent) -> None:
        self.full_response += event.text

    async def on_tool_call_event(self, event: ToolCallEvent) -> None:
        self.tool_calls_made.append({
            "name": event.info.name,
            "title": event.info.title,
            "input": event.tool_call.input,
        })

    async def on_tool_result_event(self, event: ToolResultEvent) -> None:
        pass  # Results are handled internally

    async def on_error_event(self, event: ErrorEvent) -> None:
        self.errors.append(event.error)

    async def on_stop_event(self, event: StopEvent) -> None:
        pass

    async def get_user_input(self) -> Optional[str]:
        if not self._input_provided and self._user_message:
            self._input_provided = True
            return self._user_message
        return None  # End after one message

    async def get_permission(
        self, event: PermissionRequestEvent
    ) -> PermissionDecision:
        if self.auto_approve:
            return PermissionDecision.ALLOW
        return PermissionDecision.DENY


async def main():
    """Demonstrate programmatic agent usage."""
    print("Programmatic Usage Example")
    print("=" * 40)
    print()
    print("This example shows how to use the agent from code.")
    print("In real usage, you'd provide an actual LLM provider.")
    print()
    print("Example code:")
    print()
    print("""
    # Create agent
    agent = Agent(AgentConfig(
        llm_provider=AnthropicProvider(api_key="..."),
        working_directory="/my/project",
    ))

    # Create programmatic interface
    interface = ProgrammaticInterface(auto_approve=True)
    interface.set_message("Read main.py and summarize it")

    # Run one turn
    await agent.run(interface)

    # Access results
    print(f"Response: {interface.full_response}")
    print(f"Tools used: {interface.tool_calls_made}")
    print(f"Errors: {interface.errors}")
    """)

    # Show that the interface works without an LLM
    interface = ProgrammaticInterface()
    interface.set_message("Hello")

    # Without a real LLM, we can still test the interface behavior
    assert await interface.get_user_input() == "Hello"
    assert await interface.get_user_input() is None  # Only one message
    assert await interface.get_permission(None) == PermissionDecision.ALLOW

    print("Interface self-test passed!")


if __name__ == "__main__":
    asyncio.run(main())
