"""Mirrors crates/agent/src/tools/move_path_tool.rs."""
from __future__ import annotations
import os, shutil
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

class MovePathTool(AgentTool):
    NAME = "move_path"
    def __init__(self, wd: str):
        self._wd = wd
    def description(self) -> str:
        return "Moves/renames a file or directory."
    def kind(self) -> ToolKind:
        return ToolKind.Write
    def input_schema(self) -> dict[str, Any]:
        return {"type":"object","properties":{"src":{"type":"string"},"dst":{"type":"string"}},"required":["src","dst"]}
    def initial_title(self, input: dict[str, Any]) -> str:
        return f"Move `{input.get('src','')}` → `{input.get('dst','')}`"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        src = os.path.normpath(os.path.join(self._wd, input.get("src","")))
        dst = os.path.normpath(os.path.join(self._wd, input.get("dst","")))
        nwd = os.path.normpath(self._wd)
        if not src.startswith(nwd) or not dst.startswith(nwd):
            raise ValueError("Path outside project")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        return f"Moved"
