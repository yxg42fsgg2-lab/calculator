"""Tests for the system prompt builder."""

import os
import pytest

from zed_agent.prompts.system import SystemPromptBuilder


class TestSystemPromptBuilder:
    def test_basic_prompt(self, tmp_path):
        builder = SystemPromptBuilder(
            working_directory=str(tmp_path),
            available_tools=["read_file", "grep", "terminal"],
        )
        prompt = builder.build()
        assert "software engineer" in prompt
        assert "read_file" in prompt
        assert "grep" in prompt
        assert "Communication" in prompt

    def test_no_tools(self, tmp_path):
        builder = SystemPromptBuilder(
            working_directory=str(tmp_path),
            available_tools=[],
        )
        prompt = builder.build()
        assert "no ability to use tools" in prompt

    def test_with_model_name(self, tmp_path):
        builder = SystemPromptBuilder(
            working_directory=str(tmp_path),
            available_tools=["read_file"],
            model_name="claude-sonnet-4-20250514",
        )
        prompt = builder.build()
        assert "claude-sonnet-4-20250514" in prompt

    def test_with_custom_rules(self, tmp_path):
        builder = SystemPromptBuilder(
            working_directory=str(tmp_path),
            available_tools=["read_file"],
            custom_rules="Always use TypeScript. Never use var.",
        )
        prompt = builder.build()
        assert "Always use TypeScript" in prompt

    def test_loads_project_rules_file(self, tmp_path):
        rules_file = tmp_path / ".rules"
        rules_file.write_text("Use tabs for indentation.")

        builder = SystemPromptBuilder(
            working_directory=str(tmp_path),
            available_tools=["read_file"],
        )
        prompt = builder.build()
        assert "Use tabs for indentation" in prompt

    def test_system_info_included(self, tmp_path):
        builder = SystemPromptBuilder(
            working_directory=str(tmp_path),
            available_tools=["read_file"],
        )
        prompt = builder.build()
        assert "Operating System" in prompt

    def test_grep_specific_instructions(self, tmp_path):
        builder = SystemPromptBuilder(
            working_directory=str(tmp_path),
            available_tools=["read_file", "grep", "find_path"],
        )
        prompt = builder.build()
        assert "grep" in prompt.lower()
        assert "find_path" in prompt
