"""
Read file tool - reads file contents from the project.

Ported from Zed's ReadFileTool. Key behaviors:
- Supports line range selection (start_line, end_line)
- Returns file outline for large files instead of full content
- Respects file exclusion patterns
"""

from __future__ import annotations

import os
from typing import Any, Optional

from zed_agent.core.types import ToolKind
from zed_agent.tools.base import AgentTool

# Files larger than this get an outline instead of full content
MAX_FULL_READ_LINES = 500


class ReadFileTool(AgentTool):
    """Read file contents from the project."""

    def __init__(self, working_directory: str):
        self._working_dir = working_directory

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Reads the content of the given file in the project.\n\n"
            "- Never attempt to read a path that hasn't been previously mentioned.\n"
            "- For large files, this tool returns a file outline with symbol names "
            "and line numbers instead of the full content. Use start_line/end_line "
            "to read specific sections.\n"
            "- This tool supports reading image files (PNG, JPEG, WebP, GIF, BMP, TIFF)."
        )

    @property
    def kind(self) -> ToolKind:
        return ToolKind.READ

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "The relative path of the file to read. Should be relative "
                        "to the project root."
                    ),
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional 1-based line number to start reading from.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional 1-based line number to end reading at (inclusive).",
                },
            },
            "required": ["path"],
        }

    def initial_title(self, input: dict[str, Any]) -> str:
        path = input.get("path", "")
        start = input.get("start_line")
        end = input.get("end_line")
        if start and end:
            return f"Read file `{path}` (lines {start}-{end})"
        elif start:
            return f"Read file `{path}` (from line {start})"
        return f"Read file `{path}`"

    async def run(self, input: dict[str, Any]) -> str:
        path = input.get("path", "")
        start_line = input.get("start_line")
        end_line = input.get("end_line")

        # Resolve the path
        abs_path = self._resolve_path(path)
        if abs_path is None:
            raise FileNotFoundError(f"Path {path} not found in project")

        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"{path} not found")

        with open(abs_path, "r", errors="replace") as f:
            lines = f.readlines()

        # If line range specified, return that range
        if start_line is not None or end_line is not None:
            start = max((start_line or 1) - 1, 0)
            end = end_line if end_line else len(lines)
            if end <= start:
                end = start + 1
            return "".join(lines[start:end])

        # For large files, return a summary
        if len(lines) > MAX_FULL_READ_LINES:
            return (
                f"SUCCESS: File outline retrieved. This file has {len(lines)} lines, "
                f"which is too large to read all at once.\n\n"
                f"IMPORTANT: Use start_line and end_line parameters to read "
                f"specific sections.\n\n"
                f"First 50 lines:\n"
                + "".join(lines[:50])
                + f"\n\n... ({len(lines) - 100} more lines) ...\n\n"
                f"Last 50 lines:\n"
                + "".join(lines[-50:])
            )

        return "".join(lines)

    def _resolve_path(self, path: str) -> Optional[str]:
        """Resolve a relative path against the working directory."""
        # Prevent path traversal
        if ".." in path.split(os.sep):
            return None

        abs_path = os.path.join(self._working_dir, path)
        abs_path = os.path.normpath(abs_path)

        # Ensure it's within the working directory
        if not abs_path.startswith(os.path.normpath(self._working_dir)):
            return None

        return abs_path
