"""Mirrors crates/agent/src/tools/grep_tool.rs."""
from __future__ import annotations
import fnmatch, os, re
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

RESULTS_PER_PAGE = 20

class GrepTool(AgentTool):
    NAME = "grep"
    def __init__(self, wd: str):
        self._wd = wd
    def description(self) -> str:
        return (
            "Searches file contents with a regular expression.\n"
            "- Prefer this over path search for symbols.\n"
            "- Supports full regex syntax.\n"
            "- Results paginated at 20/page; use `offset` for more."
        )
    def kind(self) -> ToolKind:
        return ToolKind.Search
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "regex": {"type": "string", "description": "Regex pattern to search file contents."},
                "include_pattern": {"type": "string", "description": "Glob for files to include."},
                "offset": {"type": "integer", "default": 0},
                "case_sensitive": {"type": "boolean", "default": False},
            },
            "required": ["regex"],
        }
    def initial_title(self, input: dict[str, Any]) -> str:
        return f"Search for `{input.get('regex', '')}`"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        regex_str = input.get("regex", "")
        include = input.get("include_pattern")
        offset = input.get("offset", 0)
        flags = 0 if input.get("case_sensitive") else re.IGNORECASE
        try:
            pattern = re.compile(regex_str, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex: {e}")
        SKIP = {".git", "node_modules", "__pycache__", "target", ".venv"}
        matches = []
        for dp, dns, fns in os.walk(self._wd):
            dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
            for fn in fns:
                fp = os.path.join(dp, fn)
                rp = os.path.relpath(fp, self._wd)
                if include and not fnmatch.fnmatch(rp, include):
                    continue
                try:
                    with open(fp, "r", errors="strict") as f:
                        for i, line in enumerate(f, 1):
                            if pattern.search(line):
                                matches.append((rp, i, line.rstrip()))
                except (OSError, UnicodeDecodeError, ValueError):
                    continue
        total = len(matches)
        page = matches[offset:offset + RESULTS_PER_PAGE]
        if not page:
            return f"No matches found for `{regex_str}`." if total == 0 else f"No more results (total: {total})."
        lines = [f"Found {total} matches (showing {offset+1}-{offset+len(page)}):\n"]
        for rp, ln, txt in page:
            lines.append(f"{rp}:{ln}: {txt}")
        if offset + RESULTS_PER_PAGE < total:
            lines.append(f"\n...{total - offset - RESULTS_PER_PAGE} more. Use offset={offset+RESULTS_PER_PAGE}.")
        return "\n".join(lines)
