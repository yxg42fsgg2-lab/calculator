"""
Find path tool - finds files by name or pattern.

Ported from Zed's FindPathTool. Uses fuzzy and glob matching
to locate files in the project.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Any

from zed_agent.core.types import ToolKind
from zed_agent.tools.base import AgentTool

MAX_RESULTS = 20


class FindPathTool(AgentTool):
    """Find files and directories by name or glob pattern."""

    def __init__(self, working_directory: str):
        self._working_dir = working_directory

    @property
    def name(self) -> str:
        return "find_path"

    @property
    def description(self) -> str:
        return (
            "Searches for files and directories by name or glob pattern.\n\n"
            "- Use this to find the full path of a file when you only know "
            "the filename or a partial path.\n"
            "- Returns matching paths relative to the project root.\n"
            "- Use glob patterns like '*.rs' or '**/test_*.py'."
        )

    @property
    def kind(self) -> ToolKind:
        return ToolKind.SEARCH

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "A filename, partial path, or glob pattern to search for. "
                        "Examples: 'main.py', '*.rs', '**/test_*.py'"
                    ),
                },
            },
            "required": ["pattern"],
        }

    def initial_title(self, input: dict[str, Any]) -> str:
        pattern = input.get("pattern", "")
        return f"Find path `{pattern}`"

    async def run(self, input: dict[str, Any]) -> str:
        pattern = input.get("pattern", "")
        matches: list[str] = []

        for dirpath, dirnames, filenames in os.walk(self._working_dir):
            # Skip hidden and common non-code dirs
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".")
                and d not in ("node_modules", "__pycache__", "target", ".git")
            ]

            for name in filenames + dirnames:
                filepath = os.path.join(dirpath, name)
                rel_path = os.path.relpath(filepath, self._working_dir)

                # Try glob match
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(
                    rel_path, pattern
                ):
                    matches.append(rel_path)
                # Try substring match
                elif pattern.lower() in name.lower():
                    matches.append(rel_path)

                if len(matches) >= MAX_RESULTS * 5:
                    break

        # Sort by relevance (shorter paths first, exact matches first)
        matches.sort(
            key=lambda p: (
                0 if os.path.basename(p).lower() == pattern.lower() else 1,
                len(p),
            )
        )
        matches = matches[:MAX_RESULTS]

        if not matches:
            return f"No files found matching `{pattern}`"

        result = f"Found {len(matches)} matches:\n"
        for m in matches:
            is_dir = os.path.isdir(os.path.join(self._working_dir, m))
            suffix = "/" if is_dir else ""
            result += f"  {m}{suffix}\n"

        return result
