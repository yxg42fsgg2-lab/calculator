#!/usr/bin/env python3
"""
Example: Adding a custom tool to the agent.

This shows how to create and register a custom tool that the LLM
can use during conversations. In this example, we add a simple
"timestamp" tool that returns the current date and time.
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zed_agent import Agent, AgentConfig
from zed_agent.tools.base import AgentTool
from zed_agent.core.types import ToolKind
from zed_agent.interface import CLIInterface


class TimestampTool(AgentTool):
    """A custom tool that returns the current timestamp."""

    @property
    def name(self) -> str:
        return "get_timestamp"

    @property
    def description(self) -> str:
        return "Returns the current date and time in various formats."

    @property
    def kind(self) -> ToolKind:
        return ToolKind.READ

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Time format: 'iso', 'unix', 'human'",
                    "enum": ["iso", "unix", "human"],
                    "default": "iso",
                },
            },
        }

    async def run(self, input: dict) -> str:
        fmt = input.get("format", "iso")
        now = datetime.now()

        if fmt == "iso":
            return now.isoformat()
        elif fmt == "unix":
            return str(int(now.timestamp()))
        elif fmt == "human":
            return now.strftime("%A, %B %d, %Y at %I:%M %p")
        else:
            return now.isoformat()


class GitStatusTool(AgentTool):
    """A custom tool that shows git status."""

    def __init__(self, working_dir: str):
        self._working_dir = working_dir

    @property
    def name(self) -> str:
        return "git_status"

    @property
    def description(self) -> str:
        return "Shows the current git status of the project (modified files, branch, etc.)"

    @property
    def kind(self) -> ToolKind:
        return ToolKind.READ

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
        }

    async def run(self, input: dict) -> str:
        import subprocess

        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "-b"],
                cwd=self._working_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return f"Git error: {result.stderr}"
            return result.stdout or "Clean working tree"
        except FileNotFoundError:
            return "Git is not installed"
        except subprocess.TimeoutExpired:
            return "Git command timed out"


def main():
    # For this example, we use a mock provider that just echoes.
    # Replace with a real provider for actual use.
    print("Custom Tool Example")
    print("=" * 40)
    print()
    print("This example shows how to create custom tools.")
    print("The following custom tools would be registered:")
    print()

    timestamp_tool = TimestampTool()
    git_tool = GitStatusTool(".")

    print(f"  1. {timestamp_tool.name}: {timestamp_tool.description}")
    print(f"     Schema: {timestamp_tool.input_schema}")
    print()
    print(f"  2. {git_tool.name}: {git_tool.description}")
    print(f"     Schema: {git_tool.input_schema}")
    print()

    # Demonstrate running the tools directly
    print("Running tools directly:")
    print()

    result = asyncio.run(timestamp_tool.run({"format": "human"}))
    print(f"  Timestamp (human): {result}")

    result = asyncio.run(timestamp_tool.run({"format": "iso"}))
    print(f"  Timestamp (ISO): {result}")

    result = asyncio.run(git_tool.run({}))
    print(f"  Git status: {result[:200]}")

    print()
    print("To use with the agent:")
    print("  config = AgentConfig(")
    print('      llm_provider=...,')
    print('      custom_tools=[TimestampTool(), GitStatusTool(".")],')
    print("  )")


if __name__ == "__main__":
    main()
