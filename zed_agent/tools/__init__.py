"""
Tool system - pluggable tools that the agent can invoke.

Each tool follows the AgentTool protocol, which mirrors Zed's AgentTool trait:
- name: unique identifier
- description: for the LLM's understanding
- input_schema: JSON schema for parameters
- run(): async execution
- kind: category (read, write, execute, search)

Built-in tools mirror Zed's tool set:
- read_file: Read file contents
- write_file: Create or overwrite files
- edit_file: Make targeted edits to files
- terminal: Execute shell commands
- grep: Search file contents with regex
- find_path: Find files by name/pattern
- list_directory: List directory contents
- web_search: Search the web

To add a custom tool:
    from zed_agent.tools.base import AgentTool, ToolKind

    class MyTool(AgentTool):
        @property
        def name(self) -> str: return "my_tool"

        @property
        def description(self) -> str: return "Does something useful"

        @property
        def input_schema(self) -> dict: return {...}

        async def run(self, input: dict) -> str: ...

    agent.register_tool(MyTool())
"""

from zed_agent.tools.base import AgentTool
from zed_agent.tools.registry import ToolRegistry
from zed_agent.tools.read_file import ReadFileTool
from zed_agent.tools.write_file import WriteFileTool
from zed_agent.tools.edit_file import EditFileTool
from zed_agent.tools.terminal import TerminalTool
from zed_agent.tools.grep import GrepTool
from zed_agent.tools.find_path import FindPathTool
from zed_agent.tools.list_directory import ListDirectoryTool

__all__ = [
    "AgentTool",
    "ToolRegistry",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "TerminalTool",
    "GrepTool",
    "FindPathTool",
    "ListDirectoryTool",
]
