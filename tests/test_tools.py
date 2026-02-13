"""Tests for the built-in tools."""

import os
import tempfile
import pytest

from zed_agent.tools.read_file import ReadFileTool
from zed_agent.tools.write_file import WriteFileTool
from zed_agent.tools.edit_file import EditFileTool
from zed_agent.tools.list_directory import ListDirectoryTool
from zed_agent.tools.find_path import FindPathTool
from zed_agent.tools.grep import GrepTool
from zed_agent.tools.terminal import TerminalTool
from zed_agent.tools.registry import ToolRegistry
from zed_agent.tools.base import AgentTool
from zed_agent.core.types import ToolKind


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace with test files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()\n"
    )
    (tmp_path / "src" / "utils.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n"
    )
    (tmp_path / "README.md").write_text("# Test Project\n\nA test project.\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text(
        "def test_main():\n    assert True\n"
    )
    return tmp_path


class TestReadFileTool:
    @pytest.mark.asyncio
    async def test_read_small_file(self, workspace):
        tool = ReadFileTool(str(workspace))
        result = await tool.run({"path": "README.md"})
        assert "# Test Project" in result

    @pytest.mark.asyncio
    async def test_read_with_line_range(self, workspace):
        tool = ReadFileTool(str(workspace))
        result = await tool.run({"path": "src/main.py", "start_line": 1, "end_line": 2})
        assert "def main():" in result
        assert "print('hello')" in result
        assert "__name__" not in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, workspace):
        tool = ReadFileTool(str(workspace))
        with pytest.raises(FileNotFoundError):
            await tool.run({"path": "nonexistent.txt"})

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, workspace):
        tool = ReadFileTool(str(workspace))
        with pytest.raises(FileNotFoundError):
            await tool.run({"path": "../../../etc/passwd"})

    def test_initial_title(self, workspace):
        tool = ReadFileTool(str(workspace))
        assert tool.initial_title({"path": "main.py"}) == "Read file `main.py`"
        assert "lines 1-10" in tool.initial_title(
            {"path": "main.py", "start_line": 1, "end_line": 10}
        )


class TestWriteFileTool:
    @pytest.mark.asyncio
    async def test_create_new_file(self, workspace):
        tool = WriteFileTool(str(workspace))
        result = await tool.run({"path": "new_file.txt", "content": "hello world"})
        assert "Created" in result
        assert (workspace / "new_file.txt").read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_create_with_directories(self, workspace):
        tool = WriteFileTool(str(workspace))
        await tool.run({"path": "deep/nested/dir/file.txt", "content": "nested"})
        assert (workspace / "deep" / "nested" / "dir" / "file.txt").read_text() == "nested"

    @pytest.mark.asyncio
    async def test_overwrite_file(self, workspace):
        tool = WriteFileTool(str(workspace))
        await tool.run({"path": "README.md", "content": "new content"})
        assert (workspace / "README.md").read_text() == "new content"

    def test_requires_permission(self, workspace):
        tool = WriteFileTool(str(workspace))
        assert tool.requires_permission is True


class TestEditFileTool:
    @pytest.mark.asyncio
    async def test_simple_edit(self, workspace):
        tool = EditFileTool(str(workspace))
        result = await tool.run({
            "path": "src/main.py",
            "old_string": "print('hello')",
            "new_string": "print('world')",
        })
        assert "Replaced" in result
        content = (workspace / "src" / "main.py").read_text()
        assert "print('world')" in content
        assert "print('hello')" not in content

    @pytest.mark.asyncio
    async def test_old_string_not_found(self, workspace):
        tool = EditFileTool(str(workspace))
        with pytest.raises(ValueError, match="not found"):
            await tool.run({
                "path": "src/main.py",
                "old_string": "nonexistent text",
                "new_string": "replacement",
            })

    @pytest.mark.asyncio
    async def test_delete_text(self, workspace):
        tool = EditFileTool(str(workspace))
        result = await tool.run({
            "path": "src/main.py",
            "old_string": "if __name__ == '__main__':\n    main()\n",
            "new_string": "",
        })
        assert "Deleted" in result


class TestGrepTool:
    @pytest.mark.asyncio
    async def test_simple_search(self, workspace):
        tool = GrepTool(str(workspace))
        result = await tool.run({"regex": "def main"})
        assert "main.py" in result

    @pytest.mark.asyncio
    async def test_case_insensitive(self, workspace):
        tool = GrepTool(str(workspace))
        result = await tool.run({"regex": "DEF MAIN", "case_sensitive": False})
        assert "main.py" in result

    @pytest.mark.asyncio
    async def test_no_matches(self, workspace):
        tool = GrepTool(str(workspace))
        result = await tool.run({"regex": "zzz_nonexistent_zzz"})
        assert "No matches" in result

    @pytest.mark.asyncio
    async def test_include_pattern(self, workspace):
        tool = GrepTool(str(workspace))
        result = await tool.run({"regex": "def", "include_pattern": "*.py"})
        assert ".py" in result


class TestFindPathTool:
    @pytest.mark.asyncio
    async def test_find_by_name(self, workspace):
        tool = FindPathTool(str(workspace))
        result = await tool.run({"pattern": "main.py"})
        assert "src/main.py" in result

    @pytest.mark.asyncio
    async def test_find_by_glob(self, workspace):
        tool = FindPathTool(str(workspace))
        result = await tool.run({"pattern": "*.md"})
        assert "README.md" in result

    @pytest.mark.asyncio
    async def test_no_results(self, workspace):
        tool = FindPathTool(str(workspace))
        result = await tool.run({"pattern": "nonexistent_xyz"})
        assert "No files found" in result


class TestListDirectoryTool:
    @pytest.mark.asyncio
    async def test_list_root(self, workspace):
        tool = ListDirectoryTool(str(workspace))
        result = await tool.run({"path": "."})
        assert "src/" in result
        assert "README.md" in result

    @pytest.mark.asyncio
    async def test_list_subdirectory(self, workspace):
        tool = ListDirectoryTool(str(workspace))
        result = await tool.run({"path": "src"})
        assert "main.py" in result
        assert "utils.py" in result

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self, workspace):
        tool = ListDirectoryTool(str(workspace))
        with pytest.raises(FileNotFoundError):
            await tool.run({"path": "nonexistent"})


class TestTerminalTool:
    @pytest.mark.asyncio
    async def test_simple_command(self, workspace):
        tool = TerminalTool(str(workspace))
        result = await tool.run({"command": "echo hello", "cd": "."})
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_command_with_timeout(self, workspace):
        tool = TerminalTool(str(workspace))
        result = await tool.run({
            "command": "sleep 10",
            "cd": ".",
            "timeout_ms": 500,
        })
        assert "timed out" in result.lower()

    def test_requires_permission(self, workspace):
        tool = TerminalTool(str(workspace))
        assert tool.requires_permission is True


class TestToolRegistry:
    def test_register_and_get(self, workspace):
        registry = ToolRegistry()
        tool = ReadFileTool(str(workspace))
        registry.register(tool)
        assert registry.get("read_file") is tool
        assert registry.has("read_file")
        assert len(registry) == 1

    def test_unregister(self, workspace):
        registry = ToolRegistry()
        tool = ReadFileTool(str(workspace))
        registry.register(tool)
        registry.unregister("read_file")
        assert not registry.has("read_file")

    def test_get_schemas(self, workspace):
        registry = ToolRegistry()
        registry.register(ReadFileTool(str(workspace)))
        registry.register(GrepTool(str(workspace)))
        schemas = registry.get_schemas()
        assert len(schemas) == 2
        names = [s.name for s in schemas]
        assert "read_file" in names
        assert "grep" in names

    def test_custom_tool(self, workspace):
        class MyCustomTool(AgentTool):
            @property
            def name(self) -> str:
                return "my_custom"

            @property
            def description(self) -> str:
                return "A custom tool"

            @property
            def kind(self):
                return ToolKind.OTHER

            @property
            def input_schema(self):
                return {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                }

            async def run(self, input):
                return f"Custom result for: {input['query']}"

        registry = ToolRegistry()
        registry.register(MyCustomTool())
        assert registry.has("my_custom")
