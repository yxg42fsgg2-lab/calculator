"""Mirrors crates/agent/src/tools/list_directory_tool.rs."""
from __future__ import annotations
import os
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

class ListDirectoryTool(AgentTool):
    NAME = "list_directory"
    def __init__(self, wd: str):
        self._wd = wd
    def description(self) -> str:
        return "Lists files and directories in the specified directory."
    def kind(self) -> ToolKind:
        return ToolKind.Read
    def input_schema(self) -> dict[str, Any]:
        return {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
    def initial_title(self, input: dict[str, Any]) -> str:
        return f"List `{input.get('path','')}`"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        path = input.get("path","")
        ap = os.path.normpath(os.path.join(self._wd, path))
        if not ap.startswith(os.path.normpath(self._wd)):
            raise ValueError("Path outside project")
        if not os.path.isdir(ap):
            raise FileNotFoundError(f"Directory not found: {path}")
        entries = []
        for e in sorted(os.listdir(ap)):
            fp = os.path.join(ap, e)
            entries.append(f"  {e}/" if os.path.isdir(fp) else f"  {e}")
        return "\n".join(entries) if entries else "(empty)"
