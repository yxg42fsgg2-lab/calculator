"""Mirrors the file creation part of Zed's tools (create_file_parser.rs)."""
from __future__ import annotations
import os
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

class CreateFileTool(AgentTool):
    NAME = "create_file"
    def __init__(self, wd: str):
        self._wd = wd
    def description(self) -> str:
        return "Creates a new file with the specified content. Parent directories are created automatically."
    def kind(self) -> ToolKind:
        return ToolKind.Write
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path of the file to create."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        }
    def initial_title(self, input: dict[str, Any]) -> str:
        return f"Create file `{input.get('path', '')}`"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        path, content = input.get("path",""), input.get("content","")
        abs_path = os.path.normpath(os.path.join(self._wd, path))
        if not abs_path.startswith(os.path.normpath(self._wd)):
            raise ValueError(f"Invalid path: {path}")
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w") as f:
            f.write(content)
        return f"Created `{path}`"
