"""
System prompt builder.

Ported from Zed's system_prompt.hbs template. Generates the system prompt
that instructs the LLM on how to behave as a coding agent.

The prompt includes:
- Communication guidelines
- Tool usage instructions (dynamically based on available tools)
- Project context (working directories, custom rules)
- Code formatting guidelines
- Debugging and diagnostic guidelines
- System information
"""

from __future__ import annotations

import os
import platform
from typing import Optional


class SystemPromptBuilder:
    """Builds the system prompt for the coding agent.

    This mirrors Zed's SystemPromptTemplate with Handlebars, but uses
    Python string formatting for simplicity and portability.
    """

    def __init__(
        self,
        working_directory: str,
        available_tools: list[str] | None = None,
        model_name: Optional[str] = None,
        custom_rules: Optional[str] = None,
        extra_context: Optional[str] = None,
    ):
        self.working_directory = working_directory
        self.available_tools = available_tools or []
        self.model_name = model_name
        self.custom_rules = custom_rules
        self.extra_context = extra_context

    def build(self) -> str:
        """Build the complete system prompt."""
        sections = [
            self._identity(),
            self._communication(),
        ]

        if self.available_tools:
            sections.append(self._tool_use())
            sections.append(self._searching_and_reading())
        else:
            sections.append(self._no_tools())

        sections.append(self._code_formatting())

        if self.available_tools:
            sections.append(self._fixing_diagnostics())
            sections.append(self._debugging())

        sections.append(self._external_apis())
        sections.append(self._system_info())

        if self.model_name:
            sections.append(self._model_info())

        if self.custom_rules:
            sections.append(self._custom_rules())

        if self.extra_context:
            sections.append(self.extra_context)

        # Load project rules files (.rules, .cursorrules, etc.)
        rules = self._load_project_rules()
        if rules:
            sections.append(rules)

        return "\n\n".join(s for s in sections if s)

    def _identity(self) -> str:
        return (
            "You are a highly skilled software engineer with extensive knowledge "
            "in many programming languages, frameworks, design patterns, and best practices."
        )

    def _communication(self) -> str:
        return (
            "## Communication\n\n"
            "- Be conversational but professional.\n"
            "- Refer to the user in the second person and yourself in the first person.\n"
            "- Format your responses in markdown. Use backticks to format file, "
            "directory, function, and class names.\n"
            "- NEVER lie or make things up.\n"
            "- Refrain from apologizing all the time when results are unexpected. "
            "Instead, just try your best to proceed or explain the circumstances."
        )

    def _tool_use(self) -> str:
        tools_list = ", ".join(f"`{t}`" for t in self.available_tools)
        return (
            "## Tool Use\n\n"
            f"Available tools: {tools_list}\n\n"
            "- Make sure to adhere to the tools schema.\n"
            "- Provide every required argument.\n"
            "- DO NOT use tools to access items already in the context.\n"
            "- Use only the tools that are currently available.\n"
            "- You can call multiple tools in a single response. If there are no "
            "dependencies between them, make all independent tool calls in parallel.\n"
            "- When running commands that may run for a long time, specify `timeout_ms`.\n"
            "- Avoid HTML entity escaping - use plain characters instead."
        )

    def _no_tools(self) -> str:
        return (
            "You have no ability to use tools or to read or write any aspect of "
            "the user's system (other than context the user might have provided).\n\n"
            "If you need the user to perform any actions, request them explicitly. "
            "Bias towards giving a response to the best of your ability."
        )

    def _searching_and_reading(self) -> str:
        lines = [
            "## Searching and Reading\n",
            "If you are unsure how to fulfill the user's request, gather more "
            "information with tool calls and/or clarifying questions.\n",
            "The project contains the following root directories:\n",
            f"- `{self.working_directory}`\n",
            "- Bias towards not asking the user for help if you can find the answer yourself.",
            "- Before you read or edit a file, you must first find the full path. "
            "DO NOT ever guess a file path!",
        ]

        if "grep" in self.available_tools:
            lines.extend(
                [
                    "- When looking for symbols in the project, prefer the `grep` tool.",
                    "- Use `find_path` (not `grep`) to find a file by name or partial path.",
                ]
            )

        return "\n".join(lines)

    def _code_formatting(self) -> str:
        return (
            "## Code Block Formatting\n\n"
            "When showing code, use markdown code blocks with the appropriate "
            "language identifier. Include file paths when referencing specific files.\n\n"
            "Example:\n"
            "```python\n"
            "# path/to/file.py\n"
            "def example():\n"
            '    return "hello"\n'
            "```"
        )

    def _fixing_diagnostics(self) -> str:
        return (
            "## Fixing Diagnostics\n\n"
            "1. Make 1-2 attempts at fixing diagnostics, then defer to the user.\n"
            "2. Never simplify code you've written just to solve diagnostics. "
            "Complete, mostly correct code is more valuable than perfect code "
            "that doesn't solve the problem."
        )

    def _debugging(self) -> str:
        return (
            "## Debugging\n\n"
            "When debugging, only make code changes if you are certain you can "
            "solve the problem.\n"
            "Otherwise, follow debugging best practices:\n"
            "1. Address the root cause instead of the symptoms.\n"
            "2. Add descriptive logging statements and error messages.\n"
            "3. Add test functions and statements to isolate the problem."
        )

    def _external_apis(self) -> str:
        return (
            "## Calling External APIs\n\n"
            "1. Unless explicitly requested, use the best suited external APIs "
            "and packages to solve the task.\n"
            "2. Choose API/package versions compatible with existing dependencies.\n"
            "3. If an external API requires an API Key, point this out to the user. "
            "DO NOT hardcode API keys."
        )

    def _system_info(self) -> str:
        return (
            "## System Information\n\n"
            f"Operating System: {platform.system()} {platform.release()}\n"
            f"Default Shell: {os.environ.get('SHELL', 'unknown')}"
        )

    def _model_info(self) -> str:
        return (
            "## Model Information\n\n"
            f"You are powered by the model named {self.model_name}."
        )

    def _custom_rules(self) -> str:
        return (
            "## User's Custom Instructions\n\n"
            "The following additional instructions are provided by the user:\n\n"
            f"{self.custom_rules}"
        )

    def _load_project_rules(self) -> Optional[str]:
        """Load project rules from standard rules files."""
        rules_file_names = [
            ".rules",
            ".cursorrules",
            ".clinerules",
            ".github/copilot-instructions.md",
        ]

        for name in rules_file_names:
            path = os.path.join(self.working_directory, name)
            if os.path.isfile(path):
                try:
                    with open(path, "r") as f:
                        content = f.read().strip()
                    if content:
                        return (
                            f"## Project Rules (from `{name}`)\n\n"
                            f"{content}"
                        )
                except OSError:
                    pass

        return None
