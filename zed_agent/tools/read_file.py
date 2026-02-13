"""Mirrors crates/agent/src/tools/read_file_tool.rs."""
from __future__ import annotations
import os
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind, ToolCallUpdateFields

MAX_FULL_READ_LINES = 500

class ReadFileTool(AgentTool):
    NAME = "read_file"

    def __init__(self, working_directory: str):
        self._wd = working_directory

    def description(self) -> str:
        return (
            "Reads the content of the given file in the project.\n"
            "- Never attempt to read a path that hasn't been previously mentioned.\n"
            "- For large files, this tool returns a file outline instead of the full content.\n"
            "- Use start_line/end_line to read specific sections."
        )

    def kind(self) -> ToolKind:
        return ToolKind.Read

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path of the file to read."},
                "start_line": {"type": "integer", "description": "Optional 1-based start line."},
                "end_line": {"type": "integer", "description": "Optional 1-based end line (inclusive)."},
            },
            "required": ["path"],
        }

    def initial_title(self, input: dict[str, Any]) -> str:
        p = input.get("path", "")
        s, e = input.get("start_line"), input.get("end_line")
        if s and e:
            return f"Read file `{p}` (lines {s}-{e})"
        return f"Read file `{p}`"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        path = input.get("path", "")
        start_line = input.get("start_line")
        end_line = input.get("end_line")
        abs_path = self._resolve(path)
        if not abs_path or not os.path.isfile(abs_path):
            raise FileNotFoundError(f"{path} not found")

        event_stream.update_fields(ToolCallUpdateFields(locations=[abs_path]))

        with open(abs_path, "r", errors="replace") as f:
            lines = f.readlines()

        if start_line is not None or end_line is not None:
            s = max((start_line or 1) - 1, 0)
            e = end_line if end_line else len(lines)
            if e <= s:
                e = s + 1
            return "".join(lines[s:e])

        if len(lines) > MAX_FULL_READ_LINES:
            return (
                f"File has {len(lines)} lines (too large). "
                f"Use start_line/end_line to read sections.\n\n"
                f"First 50 lines:\n{''.join(lines[:50])}\n"
                f"... ({len(lines)-100} more lines) ...\n"
                f"Last 50 lines:\n{''.join(lines[-50:])}"
            )
        return "".join(lines)

    def _resolve(self, path: str) -> str | None:
        if ".." in path.split(os.sep):
            return None
        p = os.path.normpath(os.path.join(self._wd, path))
        if not p.startswith(os.path.normpath(self._wd)):
            return None
        return p
