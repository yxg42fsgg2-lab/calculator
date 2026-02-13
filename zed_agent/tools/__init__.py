"""
Mirrors crates/agent/src/tools.rs — the tools! macro and built-in tool registry.

In Zed, tools! macro generates ALL_TOOL_NAMES and built_in_tools().
Here we do the same with a plain list.
"""

from zed_agent.tools.read_file import ReadFileTool
from zed_agent.tools.edit_file import EditFileTool
from zed_agent.tools.create_file import CreateFileTool
from zed_agent.tools.terminal import TerminalTool
from zed_agent.tools.grep import GrepTool
from zed_agent.tools.find_path import FindPathTool
from zed_agent.tools.list_directory import ListDirectoryTool
from zed_agent.tools.delete_path import DeletePathTool
from zed_agent.tools.copy_path import CopyPathTool
from zed_agent.tools.move_path import MovePathTool
from zed_agent.tools.create_directory import CreateDirectoryTool
from zed_agent.tools.now import NowTool
from zed_agent.tools.fetch import FetchTool
from zed_agent.thread import AgentTool

ALL_TOOL_CLASSES = [
    ReadFileTool,
    EditFileTool,
    CreateFileTool,
    TerminalTool,
    GrepTool,
    FindPathTool,
    ListDirectoryTool,
    DeletePathTool,
    CopyPathTool,
    MovePathTool,
    CreateDirectoryTool,
    NowTool,
    FetchTool,
]

ALL_TOOL_NAMES = [cls.NAME for cls in ALL_TOOL_CLASSES]  # type: ignore[attr-defined]


def create_default_tools(working_directory: str) -> list[AgentTool]:
    """Mirrors Thread::add_default_tools(). Returns instances of all built-in tools."""
    return [
        ReadFileTool(working_directory),
        EditFileTool(working_directory),
        CreateFileTool(working_directory),
        TerminalTool(working_directory),
        GrepTool(working_directory),
        FindPathTool(working_directory),
        ListDirectoryTool(working_directory),
        DeletePathTool(working_directory),
        CopyPathTool(working_directory),
        MovePathTool(working_directory),
        CreateDirectoryTool(working_directory),
        NowTool(),
        FetchTool(),
    ]
