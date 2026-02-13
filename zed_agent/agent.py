"""
Faithful port of crates/agent/src/agent.rs — NativeAgent + session management.

In Zed, NativeAgent manages multiple Thread sessions, a model registry,
project context, and template rendering. Here we port the essential structure:

- Agent creates a Thread with model + tools + system prompt
- Agent is the entry point users interact with
"""

from __future__ import annotations

from typing import Optional

from zed_agent.language_model.model import LanguageModel
from zed_agent.templates import load_project_rules, render_system_prompt
from zed_agent.thread import AgentTool, Thread
from zed_agent.tools import create_default_tools


class Agent:
    """Mirrors NativeAgent from agent.rs.

    Creates and configures a Thread with:
    - A LanguageModel
    - Default + custom tools
    - System prompt rendered from templates
    """

    def __init__(
        self,
        model: LanguageModel,
        working_directory: str = ".",
        custom_tools: Optional[list[AgentTool]] = None,
        disabled_tools: Optional[list[str]] = None,
        custom_rules: Optional[str] = None,
        enable_thinking: bool = False,
        thinking_effort: Optional[str] = None,
    ):
        self.model = model
        self.working_directory = working_directory
        self.custom_rules = custom_rules

        # Create default tools (mirrors Thread::add_default_tools)
        tools = create_default_tools(working_directory)
        if custom_tools:
            tools.extend(custom_tools)

        disabled = set(disabled_tools or [])

        # Build system prompt (mirrors build_request_messages)
        tool_names = [t.NAME for t in tools if t.NAME not in disabled]
        rules_text = load_project_rules(working_directory)
        system_prompt = render_system_prompt(
            working_directories=[working_directory],
            available_tools=tool_names,
            model_name=model.name(),
            rules_text=rules_text,
            user_rules=custom_rules,
        )

        # Create thread
        self.thread = Thread(
            model=model,
            system_prompt=system_prompt,
            working_directory=working_directory,
        )
        self.thread.thinking_enabled = enable_thinking
        self.thread.thinking_effort = thinking_effort

        # Register tools
        for tool in tools:
            if tool.NAME not in disabled:
                self.thread.add_tool(tool)

    def new_thread(self) -> Thread:
        """Create a fresh thread with the same config. Mirrors NativeAgent::new_session()."""
        tool_names = list(self.thread.tools.keys())
        rules_text = load_project_rules(self.working_directory)
        system_prompt = render_system_prompt(
            working_directories=[self.working_directory],
            available_tools=tool_names,
            model_name=self.model.name(),
            rules_text=rules_text,
            user_rules=self.custom_rules,
        )
        thread = Thread(
            model=self.model,
            system_prompt=system_prompt,
            working_directory=self.working_directory,
        )
        thread.thinking_enabled = self.thread.thinking_enabled
        thread.thinking_effort = self.thread.thinking_effort
        # Copy tool registrations
        for name, tool in self.thread.tools.items():
            thread.tools[name] = tool
        return thread
