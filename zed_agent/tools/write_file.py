"""
Write file tool - creates or overwrites files.

This combines aspects of Zed's CreateFileTool and SaveFileTool
into a simpler interface for the pluggable backend.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from zed_agent.core.types import ToolKind
from zed_agent.tools.base import AgentTool


class WriteFileTool(AgentTool):
    """Create or overwrite a file with the given content."""

    def __init__(self, working_directory: str):
        self._working_dir = working_directory

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Creates a new file or overwrites an existing file with the specified content.\n\n"
            "- Use this to create new files.\n"
            "- For making targeted edits to existing files, prefer the edit_file tool.\n"
            "- The file will be created along with any necessary parent directories."
        )

    @property
    def kind(self) -> ToolKind:
        return ToolKind.WRITE

    @property
    def requires_permission(self) -> bool:
        return True

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The relative path of the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write to the file.",
                },
            },
            "required": ["path", "content"],
        }

    def initial_title(self, input: dict[str, Any]) -> str:
        path = input.get("path", "")
        return f"Write file `{path}`"

    async def run(self, input: dict[str, Any]) -> str:
        path = input.get("path", "")
        content = input.get("content", "")

        abs_path = self._resolve_path(path)
        if abs_path is None:
            raise ValueError(f"Invalid path: {path}")

        # Create parent directories
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        is_new = not os.path.exists(abs_path)
        with open(abs_path, "w") as f:
            f.write(content)

        action = "Created" if is_new else "Updated"
        lines = content.count("\n") + 1
        return f"{action} file `{path}` ({lines} lines)"

    def _resolve_path(self, path: str) -> Optional[str]:
        if ".." in path.split(os.sep):
            return None
        abs_path = os.path.normpath(os.path.join(self._working_dir, path))
        if not abs_path.startswith(os.path.normpath(self._working_dir)):
            return None
        return abs_path
