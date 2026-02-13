"""
Abstract tool interface.

This mirrors Zed's AgentTool trait - the contract that every tool must
implement to be usable by the agent.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional

from zed_agent.core.types import ToolKind, ToolSchema


class AgentTool(abc.ABC):
    """Abstract base class for all agent tools.

    Mirrors Zed's AgentTool trait. Each tool must provide:
    - name: A unique identifier used by the LLM to call it
    - description: Human/LLM-readable description (used as the tool's
      docstring for the LLM)
    - kind: Category for UI rendering (read, write, execute, search)
    - input_schema: JSON Schema describing the tool's parameters
    - run(): Async method that executes the tool

    Optional:
    - initial_title(): Human-readable title for the UI
    - validate_input(): Pre-execution validation
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique tool name. Must match [a-zA-Z0-9_-]+."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Description for the LLM. Include usage guidelines."""
        ...

    @property
    def kind(self) -> ToolKind:
        """Tool category for UI rendering."""
        return ToolKind.OTHER

    @property
    @abc.abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's input parameters.

        Must be a valid JSON Schema object with 'type': 'object' and
        'properties' describing each parameter.
        """
        ...

    def initial_title(self, input: dict[str, Any]) -> str:
        """Generate a human-readable title for this tool invocation.

        Used by the UI to display what the tool is doing.
        Override this for more descriptive titles.
        """
        return self.name

    def validate_input(self, input: dict[str, Any]) -> Optional[str]:
        """Validate tool input before execution.

        Returns None if valid, or an error message string if invalid.
        """
        return None

    @abc.abstractmethod
    async def run(self, input: dict[str, Any]) -> str:
        """Execute the tool with the given input.

        Args:
            input: Dictionary matching the input_schema.

        Returns:
            String result to feed back to the LLM.

        Raises:
            Exception: If the tool fails. The error message will be
                      sent to the LLM as an error result.
        """
        ...

    @property
    def requires_permission(self) -> bool:
        """Whether this tool requires user permission before executing.

        Override to return True for tools that modify state (writes,
        deletes, shell commands, etc.)
        """
        return False

    def to_schema(self) -> ToolSchema:
        """Convert to a ToolSchema for the LLM."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )
