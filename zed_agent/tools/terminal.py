"""
Terminal tool - executes shell commands.

Ported from Zed's TerminalTool. Key behaviors:
- Executes commands in the user's shell
- Supports working directory and timeout
- Returns combined stdout/stderr output
- Requires user permission by default
"""

from __future__ import annotations

import asyncio
import os
import logging
from typing import Any, Optional

from zed_agent.core.types import ToolKind
from zed_agent.tools.base import AgentTool

logger = logging.getLogger(__name__)

# Default output limit in bytes
COMMAND_OUTPUT_LIMIT = 16 * 1024


class TerminalTool(AgentTool):
    """Execute shell commands and return the output."""

    def __init__(self, working_directory: str):
        self._working_dir = working_directory

    @property
    def name(self) -> str:
        return "terminal"

    @property
    def description(self) -> str:
        return (
            "Executes a shell command and returns the combined output.\n\n"
            "- The command runs in the user's default shell.\n"
            "- Use the `cd` parameter to set the working directory.\n"
            "- For long-running commands, specify `timeout_ms` to bound runtime.\n"
            "- Do not use for commands that run indefinitely (servers, watchers, etc.).\n"
            "- Each invocation spawns a new shell process (no state from previous calls).\n"
            "- The output results will be shown to the user already, only list "
            "them again if necessary."
        )

    @property
    def kind(self) -> ToolKind:
        return ToolKind.EXECUTE

    @property
    def requires_permission(self) -> bool:
        return True

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "cd": {
                    "type": "string",
                    "description": (
                        "Working directory for the command. Must be a project "
                        "root directory or subdirectory thereof."
                    ),
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": (
                        "Optional maximum runtime in milliseconds. "
                        "The command will be killed if it exceeds this."
                    ),
                },
            },
            "required": ["command", "cd"],
        }

    def initial_title(self, input: dict[str, Any]) -> str:
        return input.get("command", "")

    async def run(self, input: dict[str, Any]) -> str:
        command = input.get("command", "")
        cd = input.get("cd", "")
        timeout_ms = input.get("timeout_ms")

        # Resolve working directory
        cwd = self._resolve_path(cd)
        if cwd is None or not os.path.isdir(cwd):
            raise ValueError(
                f"Working directory not found: {cd}. "
                f"Must be a project root directory or subdirectory."
            )

        timeout = timeout_ms / 1000.0 if timeout_ms else 120.0  # 2 min default

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "TERM": "dumb"},
            )

            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return (
                    f"Command timed out after {timeout_ms}ms.\n"
                    f"Exit code: -1 (killed)\n"
                    f"Consider increasing timeout_ms or breaking the command into smaller steps."
                )

            output = stdout.decode("utf-8", errors="replace") if stdout else ""

            # Truncate if too long
            if len(output) > COMMAND_OUTPUT_LIMIT:
                half = COMMAND_OUTPUT_LIMIT // 2
                output = (
                    output[:half]
                    + f"\n\n... ({len(output) - COMMAND_OUTPUT_LIMIT} bytes truncated) ...\n\n"
                    + output[-half:]
                )

            exit_code = process.returncode
            result = output
            if exit_code != 0:
                result += f"\n\nExit code: {exit_code}"

            return result

        except Exception as e:
            logger.error(f"Terminal tool error: {e}")
            raise

    def _resolve_path(self, path: str) -> Optional[str]:
        if ".." in path.split(os.sep):
            return None
        abs_path = os.path.normpath(os.path.join(self._working_dir, path))
        if not abs_path.startswith(os.path.normpath(self._working_dir)):
            return None
        return abs_path
