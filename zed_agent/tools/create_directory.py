"""Mirrors crates/agent/src/tools/create_directory_tool.rs."""
from __future__ import annotations
import os
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

class CreateDirectoryTool(AgentTool):
    NAME = "create_directory"
    def __init__(self, wd: str):
        self._wd = wd
    def description(self) -> str:
        return "Creates a new directory (and any parent directories)."
    def kind(self) -> ToolKind:
        return ToolKind.Write
    def input_schema(self) -> dict[str, Any]:
        return {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
    def initial_title(self, input: dict[str, Any]) -> str:
        return f"Create directory `{input.get('path','')}`"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        path = input.get("path","")
        ap = os.path.normpath(os.path.join(self._wd, path))
        if not ap.startswith(os.path.normpath(self._wd)):
            raise ValueError("Path outside project")
        os.makedirs(ap, exist_ok=True)
        return f"Created directory `{path}`"
