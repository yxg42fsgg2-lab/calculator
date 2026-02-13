"""Mirrors crates/agent/src/tools/find_path_tool.rs."""
from __future__ import annotations
import fnmatch, os
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

class FindPathTool(AgentTool):
    NAME = "find_path"
    def __init__(self, wd: str):
        self._wd = wd
    def description(self) -> str:
        return "Searches for files and directories by name or glob pattern."
    def kind(self) -> ToolKind:
        return ToolKind.Search
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}
    def initial_title(self, input: dict[str, Any]) -> str:
        return f"Find path `{input.get('pattern','')}`"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        pat = input.get("pattern", "")
        SKIP = {".git","node_modules","__pycache__","target"}
        matches = []
        for dp, dns, fns in os.walk(self._wd):
            dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
            for n in fns + dns:
                rp = os.path.relpath(os.path.join(dp, n), self._wd)
                if fnmatch.fnmatch(n, pat) or fnmatch.fnmatch(rp, pat) or pat.lower() in n.lower():
                    matches.append(rp)
                if len(matches) >= 100:
                    break
        matches.sort(key=lambda p: (0 if os.path.basename(p).lower()==pat.lower() else 1, len(p)))
        matches = matches[:20]
        if not matches:
            return f"No files found matching `{pat}`"
        return "Found:\n" + "\n".join(f"  {m}" for m in matches)
