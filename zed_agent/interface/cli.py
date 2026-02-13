"""
Reference CLI interface implementation.

This is a complete, working terminal interface that demonstrates how to
implement AgentInterface. Use this as a reference for building your own.
"""

from __future__ import annotations

import sys
from typing import Optional

from zed_agent.core.events import (
    ErrorEvent,
    PermissionRequestEvent,
    RetryEvent,
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
from zed_agent.interface.base import AgentInterface


class CLIInterface(AgentInterface):
    """Terminal/CLI interface for the agent.

    This is a reference implementation that shows the minimal work needed
    to connect a UI to the agent backend.
    """

    def __init__(self, show_thinking: bool = False, show_tool_details: bool = True):
        self.show_thinking = show_thinking
        self.show_tool_details = show_tool_details
        self._current_text = ""

    async def on_session_start(self) -> None:
        print("\n" + "=" * 60)
        print("  Zed Agent - AI Coding Assistant")
        print("=" * 60)
        print("Type your message and press Enter. Type 'quit' to exit.\n")

    async def on_session_end(self) -> None:
        print("\nGoodbye!")

    async def on_text_event(self, event: TextEvent) -> None:
        sys.stdout.write(event.text)
        sys.stdout.flush()
        self._current_text += event.text
        if event.is_complete:
            if not self._current_text.endswith("\n"):
                print()
            self._current_text = ""

    async def on_thinking_event(self, event: ThinkingEvent) -> None:
        if self.show_thinking:
            sys.stdout.write(f"\033[2m{event.text}\033[0m")  # dim
            sys.stdout.flush()
            if event.is_complete:
                print()

    async def on_tool_call_event(self, event: ToolCallEvent) -> None:
        info = event.info
        print(f"\n\033[36m🔧 {info.title or info.name}\033[0m")
        if self.show_tool_details and event.tool_call.input:
            for key, value in event.tool_call.input.items():
                val_str = str(value)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                print(f"   {key}: {val_str}")

    async def on_tool_start_event(self, event: ToolCallStartEvent) -> None:
        if event.title:
            print(f"   \033[33m⏳ {event.title}\033[0m")

    async def on_tool_progress_event(self, event: ToolCallProgressEvent) -> None:
        if event.content:
            print(f"   {event.content[:200]}")

    async def on_tool_result_event(self, event: ToolResultEvent) -> None:
        result = event.result
        if result.is_error:
            print(f"   \033[31m✗ Error: {result.content[:300]}\033[0m")
        else:
            # Show a truncated result
            content = result.content
            if len(content) > 500:
                content = content[:250] + "\n   ... (truncated) ...\n" + content[-250:]
            lines = content.split("\n")
            if len(lines) > 15:
                lines = lines[:7] + ["   ... (truncated) ..."] + lines[-7:]
            print(f"   \033[32m✓ {lines[0]}\033[0m")
            for line in lines[1:]:
                print(f"   {line}")
        print()

    async def on_error_event(self, event: ErrorEvent) -> None:
        print(f"\n\033[31m❌ Error: {event.error}\033[0m")
        if event.is_retryable:
            print("   (will retry)")

    async def on_stop_event(self, event: StopEvent) -> None:
        if event.token_usage:
            usage = event.token_usage
            print(
                f"\n\033[2m[tokens: {usage.input_tokens}in / "
                f"{usage.output_tokens}out"
                f"{' / ' + str(usage.cache_read_input_tokens) + ' cache' if usage.cache_read_input_tokens else ''}"
                f"]\033[0m"
            )

    async def on_retry_event(self, event: RetryEvent) -> None:
        print(
            f"\n\033[33m⟳ Retrying ({event.attempt}/{event.max_attempts}) "
            f"in {event.delay_seconds:.1f}s"
            f"{': ' + event.reason if event.reason else ''}\033[0m"
        )

    async def on_title_event(self, event: TitleEvent) -> None:
        print(f"\033[2m[Thread: {event.title}]\033[0m")

    async def get_user_input(self) -> Optional[str]:
        try:
            text = input("\n\033[1m> \033[0m").strip()
            if text.lower() in ("quit", "exit", "q"):
                return None
            return text if text else None
        except (EOFError, KeyboardInterrupt):
            return None

    async def get_permission(
        self, event: PermissionRequestEvent
    ) -> PermissionDecision:
        print(f"\n\033[33m⚠ Permission required: {event.request.title}\033[0m")
        print(f"  Tool: {event.request.tool_name} ({event.request.tool_kind.value})")
        if event.request.description:
            print(f"  {event.request.description}")

        while True:
            try:
                response = input("  Allow? [y/n/a(lways)/N(ever)] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return PermissionDecision.DENY

            if response in ("y", "yes"):
                return PermissionDecision.ALLOW
            elif response in ("n", "no"):
                return PermissionDecision.DENY
            elif response in ("a", "always"):
                return PermissionDecision.ALLOW_ALWAYS
            elif response in ("never",):
                return PermissionDecision.DENY_ALWAYS
            else:
                print("  Please enter y, n, a(lways), or N(ever)")
