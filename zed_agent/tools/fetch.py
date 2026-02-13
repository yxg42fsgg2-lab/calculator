"""Mirrors crates/agent/src/tools/fetch_tool.rs."""
from __future__ import annotations
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

class FetchTool(AgentTool):
    NAME = "fetch"
    def description(self) -> str:
        return "Fetches content from a URL and returns the response body as text."
    def kind(self) -> ToolKind:
        return ToolKind.Read
    def input_schema(self) -> dict[str, Any]:
        return {"type":"object","properties":{"url":{"type":"string","description":"The URL to fetch."}},"required":["url"]}
    def initial_title(self, input: dict[str, Any]) -> str:
        return f"Fetch `{input.get('url','')}`"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        import urllib.request, urllib.error
        url = input.get("url", "")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "zed-agent/0.2"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read(64 * 1024).decode("utf-8", errors="replace")
            return body
        except Exception as e:
            raise RuntimeError(f"Fetch failed: {e}") from e
