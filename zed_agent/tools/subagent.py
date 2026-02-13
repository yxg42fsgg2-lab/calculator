"""
Mirrors crates/agent/src/tools/subagent_tool.rs.

The SubagentTool spawns a child Thread with its own context window to perform
a delegated task. Key behaviors:
- Max 8 subagents in parallel (MAX_PARALLEL_SUBAGENTS)
- Max 4 levels deep (MAX_SUBAGENT_DEPTH)
- Subagent inherits parent's model and a subset of tools
- Parent receives a summary when subagent completes
- Timeout support
- Cancellation propagation
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from zed_agent.thread import (
    AgentTool,
    ToolCallEventStream,
    ToolKind,
    ThreadEnvironment,
    MAX_SUBAGENT_DEPTH,
    MAX_PARALLEL_SUBAGENTS,
)


class SubagentTool(AgentTool):
    """Spawns a subagent with its own context window to perform a delegated task.

    Use this tool when you want to:
    - Perform an investigation where all you need is the outcome
    - Complete a self-contained task where you need success/failure
    - Run multiple tasks in parallel that would be slow sequentially
    """

    NAME = "subagent"

    def __init__(self, parent_thread: Any, environment: ThreadEnvironment):
        self._parent_thread = parent_thread
        self._environment = environment

    def description(self) -> str:
        return (
            "Spawns a subagent with its own context window to perform a delegated task.\n\n"
            "Use this tool when you want to:\n"
            "- Perform an investigation where all you need is the outcome\n"
            "- Complete a self-contained task where you need success/failure\n"
            "- Run multiple tasks in parallel that would be slow sequentially\n\n"
            "Each subagent has access to the same tools you do. You can optionally "
            "restrict which tools each subagent can use.\n\n"
            "Note:\n"
            f"- Maximum {MAX_PARALLEL_SUBAGENTS} subagents can run in parallel\n"
            "- Subagents cannot use tools you don't have access to\n"
            "- If spawning multiple subagents that might write to the filesystem, "
            "provide guidance on how to avoid conflicts"
        )

    def kind(self) -> ToolKind:
        return ToolKind.Other

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Short label displayed in UI while subagent runs.",
                },
                "task_prompt": {
                    "type": "string",
                    "description": "The initial prompt telling the subagent what to do.",
                },
                "summary_prompt": {
                    "type": "string",
                    "description": (
                        "Prompt sent when subagent completes, asking it to summarize. "
                        "This summary becomes your tool result."
                    ),
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional max runtime in ms. If exceeded, subagent summarizes and returns.",
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of tool names the subagent may use (subset of parent's tools).",
                },
            },
            "required": ["label", "task_prompt", "summary_prompt"],
        }

    def initial_title(self, input: dict[str, Any]) -> str:
        return input.get("label", "Subagent")

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        parent = self._parent_thread

        # Validate depth
        if parent.depth() >= MAX_SUBAGENT_DEPTH:
            raise RuntimeError(
                f"Maximum subagent depth ({MAX_SUBAGENT_DEPTH}) reached"
            )

        # Validate parallel count
        if parent.running_subagent_count() >= MAX_PARALLEL_SUBAGENTS:
            raise RuntimeError(
                f"Maximum parallel subagents ({MAX_PARALLEL_SUBAGENTS}) reached. "
                "Wait for existing subagents to complete."
            )

        # Validate allowed_tools
        allowed_tools = input.get("allowed_tools")
        if allowed_tools:
            invalid = [t for t in allowed_tools if t not in parent.tools]
            if invalid:
                raise ValueError(
                    f"The following tools do not exist: {', '.join(repr(t) for t in invalid)}"
                )

        # Spawn subagent via ThreadEnvironment
        subagent_handle = self._environment.create_subagent(
            parent_thread=parent,
            label=input.get("label", "Subagent"),
            initial_prompt=input["task_prompt"],
            timeout_ms=input.get("timeout_ms"),
            allowed_tools=allowed_tools,
        )

        # Emit subagent spawned event
        event_stream._stream._queue.put_nowait(
            type("ThreadEventSubagentSpawned", (), {"session_id": subagent_handle.id()})()
        )

        # Wait for summary, racing with cancellation
        summary_task = asyncio.create_task(
            subagent_handle.wait_for_summary(input["summary_prompt"])
        )

        # Race: summary vs user cancellation
        cancel_task = asyncio.create_task(event_stream.cancelled_by_user())
        done, pending = await asyncio.wait(
            [summary_task, cancel_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for p in pending:
            p.cancel()

        if summary_task in done:
            return summary_task.result()
        else:
            raise RuntimeError("Subagent was cancelled by user")
