"""
The pluggable UI contract — THIS IS THE FILE UI IMPLEMENTORS CARE ABOUT.

In Zed, the equivalent is AgentConnection (trait in acp_thread) + AcpThread.
Here we define a simpler async protocol:

1. The UI creates a Thread and calls thread.send("user message")
2. thread.send() returns an asyncio.Queue of ThreadEvent objects
3. The UI consumes events from the queue and renders them
4. When the UI sees a ThreadEventToolCallAuthorization, it must resolve the
   attached Future (True=allow, False=deny)
5. When the queue yields None, the turn is done.

AgentInterface is an ABC that wraps this pattern with named handler methods,
making it easier to implement for specific UI frameworks.
"""

from __future__ import annotations

import abc
import asyncio
import sys
from typing import Optional

from zed_agent.thread import (
    Thread,
    ThreadEvent,
    ThreadEventAgentText,
    ThreadEventAgentThinking,
    ThreadEventRetry,
    ThreadEventStop,
    ThreadEventToolCall,
    ThreadEventToolCallAuthorization,
    ThreadEventToolCallUpdate,
    ThreadEventUserMessage,
)


class AgentInterface(abc.ABC):
    """Abstract UI interface.

    Implement this to connect the Thread to any frontend. The interface
    consumes ThreadEvents from the queue and renders them.

    Minimum implementation: on_text, on_tool_call, on_tool_update, on_stop,
    on_error, get_user_input, authorize_tool.
    """

    @abc.abstractmethod
    async def on_text(self, text: str) -> None:
        """Streaming text from the agent."""
        ...

    async def on_thinking(self, text: str) -> None:
        """Streaming thinking/reasoning (optional)."""
        pass

    @abc.abstractmethod
    async def on_tool_call(self, event: ThreadEventToolCall) -> None:
        """Agent invoked a tool."""
        ...

    async def on_tool_update(self, event: ThreadEventToolCallUpdate) -> None:
        """Tool progress/status update (optional)."""
        pass

    @abc.abstractmethod
    async def on_stop(self, reason: str) -> None:
        """Turn finished."""
        ...

    @abc.abstractmethod
    async def on_error(self, error: Exception) -> None:
        """An error occurred."""
        ...

    async def on_retry(self, event: ThreadEventRetry) -> None:
        """Retrying after error (optional)."""
        pass

    @abc.abstractmethod
    async def authorize_tool(self, event: ThreadEventToolCallAuthorization) -> bool:
        """User must decide: allow this tool call? Return True/False."""
        ...

    @abc.abstractmethod
    async def get_user_input(self) -> Optional[str]:
        """Get next user message. Return None to quit."""
        ...

    async def run(self, thread: Thread) -> None:
        """Main loop: get input → send to thread → consume events → repeat.

        Mirrors the NativeAgentConnection event consumption loop.
        """
        while True:
            user_input = await self.get_user_input()
            if user_input is None:
                break

            events_queue = thread.send(user_input)
            await self._consume_events(events_queue)

    async def _consume_events(self, queue: asyncio.Queue) -> None:
        """Consume all events from a turn. Mirrors handle_thread_events()."""
        while True:
            event = await queue.get()
            if event is None:
                break  # turn done

            if isinstance(event, Exception):
                await self.on_error(event)
                continue

            if isinstance(event, ThreadEventAgentText):
                await self.on_text(event.text)
            elif isinstance(event, ThreadEventAgentThinking):
                await self.on_thinking(event.text)
            elif isinstance(event, ThreadEventToolCall):
                await self.on_tool_call(event)
            elif isinstance(event, ThreadEventToolCallUpdate):
                await self.on_tool_update(event)
            elif isinstance(event, ThreadEventToolCallAuthorization):
                allowed = await self.authorize_tool(event)
                if not event.response_future.done():
                    event.response_future.set_result(allowed)
            elif isinstance(event, ThreadEventRetry):
                await self.on_retry(event)
            elif isinstance(event, ThreadEventStop):
                await self.on_stop(event.reason)
            elif isinstance(event, ThreadEventUserMessage):
                pass  # echo, usually ignored


class CLIInterface(AgentInterface):
    """Reference CLI implementation — demonstrates the minimum needed to connect a UI."""

    async def on_text(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    async def on_thinking(self, text: str) -> None:
        sys.stdout.write(f"\033[2m{text}\033[0m")
        sys.stdout.flush()

    async def on_tool_call(self, event: ThreadEventToolCall) -> None:
        print(f"\n\033[36m🔧 [{event.kind.value}] {event.title}\033[0m")

    async def on_tool_update(self, event: ThreadEventToolCallUpdate) -> None:
        f = event.fields
        if f.status:
            print(f"   status: {f.status.value}")

    async def on_stop(self, reason: str) -> None:
        print(f"\n\033[2m[stop: {reason}]\033[0m\n")

    async def on_error(self, error: Exception) -> None:
        print(f"\n\033[31m❌ {error}\033[0m\n")

    async def on_retry(self, event: ThreadEventRetry) -> None:
        print(f"\033[33m⟳ Retry {event.attempt}/{event.max_attempts} in {event.delay_seconds:.0f}s: {event.last_error}\033[0m")

    async def authorize_tool(self, event: ThreadEventToolCallAuthorization) -> bool:
        try:
            resp = input(f"\n\033[33m⚠ Allow '{event.title}'? [y/n] \033[0m").strip().lower()
            return resp in ("y", "yes", "")
        except (EOFError, KeyboardInterrupt):
            return False

    async def get_user_input(self) -> Optional[str]:
        try:
            text = input("\n\033[1m> \033[0m").strip()
            return text if text and text.lower() not in ("quit", "exit", "q") else None
        except (EOFError, KeyboardInterrupt):
            return None
