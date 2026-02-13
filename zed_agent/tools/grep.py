"""
Grep tool - searches file contents with regex.

Ported from Zed's GrepTool. Provides regex-based content search
across the project with pagination.
"""

from __future__ import annotations

import fnmatch
import os
import re
from typing import Any

from zed_agent.core.types import ToolKind
from zed_agent.tools.base import AgentTool

RESULTS_PER_PAGE = 20
CONTEXT_LINES = 2


class GrepTool(AgentTool):
    """Search file contents with regex patterns."""

    def __init__(self, working_directory: str):
        self._working_dir = working_directory

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Searches the contents of files in the project with a regular expression.\n\n"
            "- Prefer this tool over path search when looking for symbols.\n"
            "- Supports full regex syntax.\n"
            "- Pass `include_pattern` to narrow search to specific file types.\n"
            "- Results are paginated with 20 matches per page. Use `offset` for more.\n"
            "- Never use this tool to search for paths. Only search file contents."
        )

    @property
    def kind(self) -> ToolKind:
        return ToolKind.SEARCH

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "regex": {
                    "type": "string",
                    "description": "A regex pattern to search for in file contents.",
                },
                "include_pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern for files to include (e.g., '**/*.rs'). "
                        "If omitted, all files are searched."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting position for paginated results (0-based).",
                    "default": 0,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the regex is case-sensitive. Default: false.",
                    "default": False,
                },
            },
            "required": ["regex"],
        }

    def initial_title(self, input: dict[str, Any]) -> str:
        regex = input.get("regex", "")
        return f"Search for `{regex}`"

    async def run(self, input: dict[str, Any]) -> str:
        regex_str = input.get("regex", "")
        include_pattern = input.get("include_pattern")
        offset = input.get("offset", 0)
        case_sensitive = input.get("case_sensitive", False)

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(regex_str, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex: {e}")

        matches: list[dict[str, Any]] = []
        files_searched = 0

        for dirpath, _dirnames, filenames in os.walk(self._working_dir):
            # Skip hidden directories and common non-code dirs
            rel_dir = os.path.relpath(dirpath, self._working_dir)
            if any(
                part.startswith(".")
                for part in rel_dir.split(os.sep)
                if part != "."
            ):
                continue
            if any(
                skip in rel_dir.split(os.sep)
                for skip in ["node_modules", "__pycache__", "target", ".git"]
            ):
                continue

            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, self._working_dir)

                # Apply include pattern
                if include_pattern and not fnmatch.fnmatch(
                    rel_path, include_pattern
                ):
                    continue

                # Skip binary files
                try:
                    with open(filepath, "r", errors="strict") as f:
                        try:
                            lines = f.readlines()
                        except (UnicodeDecodeError, ValueError):
                            continue
                except (OSError, PermissionError):
                    continue

                files_searched += 1

                for line_num, line in enumerate(lines, 1):
                    if pattern.search(line):
                        # Get context
                        start = max(0, line_num - 1 - CONTEXT_LINES)
                        end = min(len(lines), line_num + CONTEXT_LINES)
                        context = "".join(lines[start:end])

                        matches.append(
                            {
                                "path": rel_path,
                                "line": line_num,
                                "text": line.rstrip(),
                                "context": context,
                            }
                        )

        total = len(matches)
        page_matches = matches[offset : offset + RESULTS_PER_PAGE]

        if not page_matches:
            if total == 0:
                return f"No matches found for `{regex_str}` in {files_searched} files."
            else:
                return f"No more results. Total matches: {total}."

        result_lines = [
            f"Found {total} matches in {files_searched} files"
            f" (showing {offset + 1}-{offset + len(page_matches)}):\n"
        ]

        for m in page_matches:
            result_lines.append(f"\n{m['path']}:{m['line']}: {m['text']}")

        if offset + RESULTS_PER_PAGE < total:
            remaining = total - offset - RESULTS_PER_PAGE
            result_lines.append(
                f"\n\n... {remaining} more matches. "
                f"Use offset={offset + RESULTS_PER_PAGE} to see next page."
            )

        return "\n".join(result_lines)
