"""
Core Agent - the main orchestrator.

This is the heart of the system. It manages the conversation loop:
1. Receive user input
2. Build the LLM request (system prompt + messages + tools)
3. Stream the completion
4. Execute tool calls
5. Feed results back to the LLM
6. Repeat until the agent stops

Ported from Zed's Thread.send() / Thread.run_turn() / NativeAgentConnection.

The Agent is completely UI-agnostic. It communicates with the frontend
exclusively through the AgentInterface protocol.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from zed_agent.core.config import AgentConfig
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
    ToolCallStartEvent,
    ToolResultEvent,
)
from zed_agent.core.thread import Thread
from zed_agent.core.types import (
    PermissionDecision,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolCallInfo,
    ToolKind,
    ToolPermissionRequest,
    ToolResult,
)
from zed_agent.interface.base import AgentInterface
from zed_agent.llm.base import (
    CompletionRequest,
    LLMProvider,
    StopEvent as LLMStopEvent,
    TextChunk,
    ThinkingChunk,
    ToolUseEvent,
    UsageEvent,
)
from zed_agent.prompts.system import SystemPromptBuilder
from zed_agent.tools.base import AgentTool
from zed_agent.tools.registry import ToolRegistry
from zed_agent.tools.read_file import ReadFileTool
from zed_agent.tools.write_file import WriteFileTool
from zed_agent.tools.edit_file import EditFileTool
from zed_agent.tools.terminal import TerminalTool
from zed_agent.tools.grep import GrepTool
from zed_agent.tools.find_path import FindPathTool
from zed_agent.tools.list_directory import ListDirectoryTool

logger = logging.getLogger(__name__)


class Agent:
    """The core AI coding agent.

    Orchestrates the conversation loop between user, LLM, and tools.
    Completely UI-agnostic - communicates through the AgentInterface protocol.

    Usage:
        config = AgentConfig(
            llm_provider=AnthropicProvider(api_key="..."),
            working_directory="/path/to/project",
        )
        agent = Agent(config)

        # Run with a UI
        await agent.run(CLIInterface())

        # Or drive programmatically
        async for event in agent.send("Fix the bug in main.py"):
            print(event)
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.thread = Thread()
        self.tools = ToolRegistry()
        self._always_allowed_tools: set[str] = set()
        self._always_denied_tools: set[str] = set()
        self._cancelled = False
        self._turn_token_usage = TokenUsage()

        if config.llm_provider is None:
            raise ValueError("AgentConfig.llm_provider is required")

        self._llm: LLMProvider = config.llm_provider

        # Register default tools
        if config.enable_default_tools:
            self._register_default_tools()

        # Register custom tools
        for tool in config.custom_tools:
            self.tools.register(tool)

        # Remove disabled tools
        for name in config.disabled_tools:
            self.tools.unregister(name)

    def _register_default_tools(self) -> None:
        """Register the default built-in tools (mirrors Zed's tool set)."""
        wd = self.config.working_directory
        self.tools.register(ReadFileTool(wd))
        self.tools.register(WriteFileTool(wd))
        self.tools.register(EditFileTool(wd))
        self.tools.register(TerminalTool(wd))
        self.tools.register(GrepTool(wd))
        self.tools.register(FindPathTool(wd))
        self.tools.register(ListDirectoryTool(wd))

    def register_tool(self, tool: AgentTool) -> None:
        """Register a custom tool at runtime."""
        self.tools.register(tool)

    def unregister_tool(self, name: str) -> None:
        """Remove a tool at runtime."""
        self.tools.unregister(name)

    async def run(self, interface: AgentInterface) -> None:
        """Run the agent with the given interface.

        This is the main entry point for interactive use. It runs the
        conversation loop until the user quits.
        """
        await interface.on_session_start()

        try:
            while True:
                user_input = await interface.get_user_input()
                if user_input is None:
                    break

                await self._run_turn(user_input, interface)
        except Exception as e:
            await interface.on_error_event(ErrorEvent(error=str(e)))
        finally:
            await interface.on_session_end()

    async def send(self, message: str, interface: AgentInterface) -> StopReason:
        """Send a single message and process the response.

        Returns the stop reason when the agent finishes its turn.
        """
        return await self._run_turn(message, interface)

    async def _run_turn(
        self, user_message: str, interface: AgentInterface
    ) -> StopReason:
        """Execute a full agent turn (may include multiple LLM calls for tool use).

        This mirrors Zed's Thread.send() -> run_turn() flow:
        1. Add user message
        2. Build system prompt
        3. Stream LLM completion
        4. If tool calls: execute tools, add results, goto 3
        5. If end_turn: return
        """
        self._cancelled = False
        self._turn_token_usage = TokenUsage()

        # Add user message to thread
        self.thread.add_user_message(user_message)

        tool_call_count = 0
        turn_count = 0

        while not self._cancelled:
            turn_count += 1
            if turn_count > self.config.max_turns:
                await interface.on_error_event(
                    ErrorEvent(error=f"Max turns ({self.config.max_turns}) exceeded")
                )
                return StopReason.ERROR

            # Build the completion request
            request = self._build_request()

            # Stream the completion with retries
            stop_reason, tool_calls = await self._stream_with_retries(
                request, interface
            )

            if self._cancelled:
                return StopReason.CANCELLED

            if stop_reason == StopReason.TOOL_USE and tool_calls:
                # Execute tool calls
                tool_call_count += len(tool_calls)
                if tool_call_count > self.config.max_tool_calls_per_turn:
                    await interface.on_error_event(
                        ErrorEvent(
                            error=f"Max tool calls ({self.config.max_tool_calls_per_turn}) exceeded"
                        )
                    )
                    return StopReason.ERROR

                results = await self._execute_tools(tool_calls, interface)
                self.thread.add_tool_results(results)
                # Continue the loop - LLM will process tool results
                continue
            else:
                # Agent finished its turn
                await interface.on_stop_event(
                    StopEvent(
                        reason=stop_reason,
                        token_usage=self._turn_token_usage,
                    )
                )
                return stop_reason

        return StopReason.CANCELLED

    def _build_request(self) -> CompletionRequest:
        """Build the LLM completion request."""
        # Build system prompt
        prompt_builder = SystemPromptBuilder(
            working_directory=self.config.working_directory,
            available_tools=self.tools.list_names(),
            model_name=self._llm.info.model_name,
            custom_rules=self.config.custom_rules,
            extra_context=self.config.extra_system_context,
        )

        return CompletionRequest(
            system_prompt=prompt_builder.build(),
            messages=self.thread.to_llm_messages(),
            tools=self.tools.get_schemas(),
            thinking_enabled=self.config.enable_thinking,
            thinking_effort=self.config.thinking_effort,
            max_tokens=self._llm.info.max_output_tokens,
        )

    async def _stream_with_retries(
        self,
        request: CompletionRequest,
        interface: AgentInterface,
    ) -> tuple[StopReason, list[ToolCall]]:
        """Stream a completion with automatic retries on transient errors.

        Mirrors Zed's retry logic with exponential backoff.
        """
        last_error: Optional[str] = None

        for attempt in range(self.config.max_retries + 1):
            if self._cancelled:
                return StopReason.CANCELLED, []

            try:
                return await self._stream_completion(request, interface)
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.max_retries:
                    delay = self.config.base_retry_delay * (2**attempt)
                    await interface.on_retry_event(
                        RetryEvent(
                            attempt=attempt + 1,
                            max_attempts=self.config.max_retries,
                            delay_seconds=delay,
                            reason=last_error,
                        )
                    )
                    await asyncio.sleep(delay)
                else:
                    await interface.on_error_event(
                        ErrorEvent(error=f"Failed after {self.config.max_retries} retries: {last_error}")
                    )

        return StopReason.ERROR, []

    async def _stream_completion(
        self,
        request: CompletionRequest,
        interface: AgentInterface,
    ) -> tuple[StopReason, list[ToolCall]]:
        """Stream a single completion from the LLM.

        Mirrors Zed's handle_thread_events() - processes the stream of
        CompletionEvents and dispatches them to the interface.
        """
        tool_calls: list[ToolCall] = []
        agent_text = ""
        agent_thinking = ""
        stop_reason = StopReason.END_TURN

        await interface.on_status_event(StatusEvent(status="thinking"))

        async for event in self._llm.stream_completion(request):
            if self._cancelled:
                return StopReason.CANCELLED, []

            if isinstance(event, TextChunk):
                agent_text += event.text
                await interface.on_text_event(TextEvent(text=event.text))

            elif isinstance(event, ThinkingChunk):
                agent_thinking += event.text
                await interface.on_thinking_event(ThinkingEvent(text=event.text))

            elif isinstance(event, ToolUseEvent):
                tc = event.tool_call
                tool_calls.append(tc)

                # Get tool metadata for the UI
                tool = self.tools.get(tc.name)
                kind = tool.kind if tool else ToolKind.OTHER
                title = tool.initial_title(tc.input) if tool else tc.name

                await interface.on_tool_call_event(
                    ToolCallEvent(
                        tool_call=tc,
                        info=ToolCallInfo(
                            id=tc.id,
                            name=tc.name,
                            kind=kind,
                            title=title,
                        ),
                    )
                )

            elif isinstance(event, UsageEvent):
                self._turn_token_usage = self._turn_token_usage + event.usage
                self.thread.update_token_usage(event.usage)

            elif isinstance(event, LLMStopEvent):
                stop_reason = event.reason

        # Record the agent's message in the thread
        if agent_text or agent_thinking:
            self.thread.add_agent_text(agent_text, agent_thinking)
        if tool_calls:
            self.thread.add_agent_tool_calls(tool_calls)

        return stop_reason, tool_calls

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        interface: AgentInterface,
    ) -> list[ToolResult]:
        """Execute a batch of tool calls.

        Mirrors Zed's tool execution flow:
        1. Check permissions
        2. Execute the tool
        3. Collect results
        """
        results: list[ToolResult] = []

        for tc in tool_calls:
            if self._cancelled:
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content="Cancelled by user",
                        is_error=True,
                    )
                )
                continue

            tool = self.tools.get(tc.name)
            if tool is None:
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=f"Tool '{tc.name}' not found",
                        is_error=True,
                    )
                )
                continue

            # Check permissions
            if tool.requires_permission and not self._is_auto_approved(tool, tc):
                decision = await interface.get_permission(
                    PermissionRequestEvent(
                        request=ToolPermissionRequest(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            tool_kind=tool.kind,
                            title=tool.initial_title(tc.input),
                            description=str(tc.input),
                        )
                    )
                )

                if decision == PermissionDecision.DENY:
                    results.append(
                        ToolResult(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            content="Tool call denied by user",
                            is_error=True,
                        )
                    )
                    continue
                elif decision == PermissionDecision.DENY_ALWAYS:
                    self._always_denied_tools.add(tc.name)
                    results.append(
                        ToolResult(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            content="Tool call denied by user",
                            is_error=True,
                        )
                    )
                    continue
                elif decision == PermissionDecision.ALLOW_ALWAYS:
                    self._always_allowed_tools.add(tc.name)

            # Validate input
            validation_error = tool.validate_input(tc.input)
            if validation_error:
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=f"Invalid input: {validation_error}",
                        is_error=True,
                    )
                )
                continue

            # Execute the tool
            await interface.on_tool_start_event(
                ToolCallStartEvent(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    title=tool.initial_title(tc.input),
                )
            )

            try:
                output = await tool.run(tc.input)
                result = ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=output,
                    is_error=False,
                )
            except Exception as e:
                logger.error(f"Tool {tc.name} failed: {e}")
                result = ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=str(e),
                    is_error=True,
                )

            results.append(result)
            await interface.on_tool_result_event(ToolResultEvent(result=result))

        return results

    def _is_auto_approved(self, tool: AgentTool, tc: ToolCall) -> bool:
        """Check if a tool call is auto-approved based on settings."""
        if tc.name in self._always_denied_tools:
            return False
        if tc.name in self._always_allowed_tools:
            return True
        if tool.kind == ToolKind.READ and self.config.auto_approve_reads:
            return True
        if tool.kind == ToolKind.WRITE and self.config.auto_approve_writes:
            return True
        if tool.kind == ToolKind.EXECUTE and self.config.auto_approve_terminal:
            return True
        return False

    def cancel(self) -> None:
        """Cancel the current operation."""
        self._cancelled = True

    def reset(self) -> None:
        """Reset the agent state (new conversation)."""
        self.thread = Thread()
        self._cancelled = False
        self._turn_token_usage = TokenUsage()
