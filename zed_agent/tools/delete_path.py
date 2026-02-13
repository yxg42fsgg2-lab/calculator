"""Mirrors crates/agent/src/tools/delete_path_tool.rs."""
from __future__ import annotations
import os, shutil
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

class DeletePathTool(AgentTool):
    NAME = "delete_path"
    def __init__(self, wd: str):
        self._wd = wd
    def description(self) -> str:
        return "Deletes a file or directory at the given path."
    def kind(self) -> ToolKind:
        return ToolKind.Write
    def input_schema(self) -> dict[str, Any]:
        return {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
    def initial_title(self, input: dict[str, Any]) -> str:
        return f"Delete `{input.get('path','')}`"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        path = input.get("path","")
        ap = os.path.normpath(os.path.join(self._wd, path))
        if not ap.startswith(os.path.normpath(self._wd)):
            raise ValueError("Path outside project")
        if os.path.isdir(ap):
            shutil.rmtree(ap)
        elif os.path.exists(ap):
            os.remove(ap)
        else:
            raise FileNotFoundError(f"{path} not found")
        return f"Deleted `{path}`"
