"""
Agent configuration.

Central configuration object that controls agent behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from zed_agent.llm.base import LLMProvider


@dataclass
class AgentConfig:
    """Configuration for the Agent.

    This is the primary entry point for configuring the agent backend.
    At minimum, you need to provide an LLM provider and working directory.

    Example:
        config = AgentConfig(
            llm_provider=AnthropicProvider(api_key="sk-..."),
            working_directory="/path/to/project",
        )
    """

    # Required
    llm_provider: Optional[LLMProvider] = None
    working_directory: str = "."

    # Tool configuration
    enable_default_tools: bool = True
    custom_tools: list[Any] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)

    # System prompt customization
    custom_rules: Optional[str] = None
    extra_system_context: Optional[str] = None

    # Permission settings
    auto_approve_reads: bool = True
    auto_approve_writes: bool = False
    auto_approve_terminal: bool = False
    always_approved_commands: list[str] = field(default_factory=list)

    # Retry settings
    max_retries: int = 4
    base_retry_delay: float = 5.0

    # Turn limits
    max_tool_calls_per_turn: int = 50
    max_turns: int = 100

    # Thinking/reasoning
    enable_thinking: bool = False
    thinking_effort: Optional[str] = None
