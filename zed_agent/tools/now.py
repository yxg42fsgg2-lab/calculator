"""Mirrors crates/agent/src/tools/now_tool.rs."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

class NowTool(AgentTool):
    NAME = "now"
    def description(self) -> str:
        return "Returns the current date and time."
    def kind(self) -> ToolKind:
        return ToolKind.Other
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}
    def initial_title(self, input: dict[str, Any]) -> str:
        return "Get current time"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        return datetime.now(timezone.utc).isoformat()
