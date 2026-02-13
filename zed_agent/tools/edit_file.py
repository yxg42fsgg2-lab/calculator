"""
Edit file tool - makes targeted edits to existing files.

Ported from Zed's EditFileTool. Supports search-and-replace style edits,
which is the most reliable way for LLMs to make precise changes.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from zed_agent.core.types import ToolKind
from zed_agent.tools.base import AgentTool


class EditFileTool(AgentTool):
    """Make targeted edits to an existing file using search-and-replace."""

    def __init__(self, working_directory: str):
        self._working_dir = working_directory

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Makes targeted edits to an existing file using search and replace.\n\n"
            "- Provide the old_string to find and the new_string to replace it with.\n"
            "- The old_string must match exactly (including whitespace and indentation).\n"
            "- The old_string must be unique within the file to avoid ambiguous replacements.\n"
            "- To delete text, provide the old_string with an empty new_string.\n"
            "- To insert text, include surrounding context in old_string to pinpoint the location."
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
                    "description": "The relative path of the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": (
                        "The exact text to find in the file. Must be unique within the file."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace old_string with.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    def initial_title(self, input: dict[str, Any]) -> str:
        path = input.get("path", "")
        return f"Edit file `{path}`"

    async def run(self, input: dict[str, Any]) -> str:
        path = input.get("path", "")
        old_string = input.get("old_string", "")
        new_string = input.get("new_string", "")

        abs_path = self._resolve_path(path)
        if abs_path is None:
            raise ValueError(f"Invalid path: {path}")

        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(abs_path, "r") as f:
            content = f.read()

        # Verify uniqueness
        count = content.count(old_string)
        if count == 0:
            raise ValueError(
                f"old_string not found in {path}. Make sure it matches exactly "
                f"(including whitespace and indentation)."
            )
        if count > 1:
            raise ValueError(
                f"old_string found {count} times in {path}. "
                f"Include more surrounding context to make it unique."
            )

        # Apply the edit
        new_content = content.replace(old_string, new_string, 1)

        with open(abs_path, "w") as f:
            f.write(new_content)

        # Describe the change
        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1
        if not new_string:
            return f"Deleted {old_lines} lines from `{path}`"
        elif not old_string:
            return f"Inserted {new_lines} lines in `{path}`"
        else:
            return f"Replaced {old_lines} lines with {new_lines} lines in `{path}`"

    def _resolve_path(self, path: str) -> Optional[str]:
        if ".." in path.split(os.sep):
            return None
        abs_path = os.path.normpath(os.path.join(self._working_dir, path))
        if not abs_path.startswith(os.path.normpath(self._working_dir)):
            return None
        return abs_path
