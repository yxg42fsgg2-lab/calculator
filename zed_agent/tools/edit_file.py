"""Mirrors crates/agent/src/tools/edit_file_tool.rs."""
from __future__ import annotations
import os
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

class EditFileTool(AgentTool):
    NAME = "edit_file"
    def __init__(self, wd: str):
        self._wd = wd
    def description(self) -> str:
        return (
            "Makes targeted edits to an existing file using search and replace.\n"
            "- old_string must match exactly (including whitespace).\n"
            "- old_string must be unique in the file."
        )
    def kind(self) -> ToolKind:
        return ToolKind.Write
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path of the file."},
                "old_string": {"type": "string", "description": "Exact text to find."},
                "new_string": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_string", "new_string"],
        }
    def initial_title(self, input: dict[str, Any]) -> str:
        return f"Edit file `{input.get('path', '')}`"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        path, old, new = input.get("path",""), input.get("old_string",""), input.get("new_string","")
        abs_path = os.path.normpath(os.path.join(self._wd, path))
        if not abs_path.startswith(os.path.normpath(self._wd)):
            raise ValueError(f"Invalid path: {path}")
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(abs_path) as f:
            content = f.read()
        count = content.count(old)
        if count == 0:
            raise ValueError(f"old_string not found in {path}")
        if count > 1:
            raise ValueError(f"old_string found {count} times in {path}. Include more context.")
        with open(abs_path, "w") as f:
            f.write(content.replace(old, new, 1))
        return f"Edited `{path}`"
