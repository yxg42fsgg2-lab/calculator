"""Mirrors crates/agent/src/tools/terminal_tool.rs."""
from __future__ import annotations
import asyncio, os
from typing import Any
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

COMMAND_OUTPUT_LIMIT = 16 * 1024

class TerminalTool(AgentTool):
    NAME = "terminal"
    def __init__(self, wd: str):
        self._wd = wd
    def description(self) -> str:
        return (
            "Executes a shell one-liner and returns the combined output.\n"
            "- Use `cd` to set working directory.\n"
            "- Specify `timeout_ms` for long-running commands.\n"
            "- Do not use for commands that run indefinitely."
        )
    def kind(self) -> ToolKind:
        return ToolKind.Execute
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to execute."},
                "cd": {"type": "string", "description": "Working directory (project root or subdir)."},
                "timeout_ms": {"type": "integer", "description": "Optional max runtime in ms."},
            },
            "required": ["command", "cd"],
        }
    def initial_title(self, input: dict[str, Any]) -> str:
        return input.get("command", "")

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        command = input.get("command", "")
        cd = input.get("cd", ".")
        timeout_ms = input.get("timeout_ms")
        cwd = os.path.normpath(os.path.join(self._wd, cd))
        if not cwd.startswith(os.path.normpath(self._wd)) or not os.path.isdir(cwd):
            raise ValueError(f"Working directory not found: {cd}")
        timeout = timeout_ms / 1000.0 if timeout_ms else 120.0
        proc = await asyncio.create_subprocess_shell(
            command, cwd=cwd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env={**os.environ, "TERM": "dumb"},
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill(); await proc.wait()
            return f"Command timed out after {timeout_ms}ms.\nExit code: -1 (killed)"
        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        if len(output) > COMMAND_OUTPUT_LIMIT:
            h = COMMAND_OUTPUT_LIMIT // 2
            output = output[:h] + f"\n...({len(output)-COMMAND_OUTPUT_LIMIT} bytes truncated)...\n" + output[-h:]
        if proc.returncode != 0:
            output += f"\n\nExit code: {proc.returncode}"
        return output
