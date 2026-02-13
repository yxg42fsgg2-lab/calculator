"""
List directory tool - lists directory contents.

Ported from Zed's ListDirectoryTool.
"""

from __future__ import annotations

import os
from typing import Any

from zed_agent.core.types import ToolKind
from zed_agent.tools.base import AgentTool


class ListDirectoryTool(AgentTool):
    """List the contents of a directory."""

    def __init__(self, working_directory: str):
        self._working_dir = working_directory

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return (
            "Lists files and directories in the specified directory.\n\n"
            "- Returns the directory contents with file types indicated.\n"
            "- Use this to explore the project structure."
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
                    "description": "The relative path of the directory to list.",
                },
            },
            "required": ["path"],
        }

    def initial_title(self, input: dict[str, Any]) -> str:
        path = input.get("path", "")
        return f"List directory `{path}`"

    async def run(self, input: dict[str, Any]) -> str:
        path = input.get("path", "")

        abs_path = os.path.normpath(os.path.join(self._working_dir, path))
        if not abs_path.startswith(os.path.normpath(self._working_dir)):
            raise ValueError(f"Path is outside project: {path}")

        if not os.path.isdir(abs_path):
            raise FileNotFoundError(f"Directory not found: {path}")

        entries: list[str] = []
        try:
            for entry in sorted(os.listdir(abs_path)):
                full_path = os.path.join(abs_path, entry)
                if os.path.isdir(full_path):
                    entries.append(f"  {entry}/")
                else:
                    size = os.path.getsize(full_path)
                    entries.append(f"  {entry} ({_format_size(size)})")
        except PermissionError:
            raise PermissionError(f"Permission denied: {path}")

        if not entries:
            return f"Directory `{path}` is empty."

        return f"Contents of `{path}`:\n" + "\n".join(entries)


def _format_size(size: int) -> str:
    """Format a file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"
