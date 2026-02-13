"""
Tool registry - manages available tools.

Mirrors Zed's tool registration system where tools are registered
at thread creation time and can be filtered per-session.
"""

from __future__ import annotations

import logging
from typing import Optional

from zed_agent.core.types import ToolSchema
from zed_agent.tools.base import AgentTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of available tools.

    Tools can be added and removed dynamically. The registry provides
    tool schemas for the LLM and dispatches tool calls to the right tool.
    """

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        """Register a tool. Replaces any existing tool with the same name."""
        if tool.name in self._tools:
            logger.warning(f"Replacing existing tool: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> Optional[AgentTool]:
        """Remove a tool by name. Returns the removed tool or None."""
        return self._tools.pop(name, None)

    def get(self, name: str) -> Optional[AgentTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[AgentTool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> list[ToolSchema]:
        """Get tool schemas for all registered tools (for LLM)."""
        return [tool.to_schema() for tool in self._tools.values()]

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
