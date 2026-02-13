"""
zed_agent - A pluggable AI coding agent backend extracted from Zed's architecture.

This package provides a complete AI coding agent backend that can be connected
to any user interface. The architecture is designed around clear interface
contracts that frontends must implement.

Quick Start:
    from zed_agent import Agent, AgentConfig
    from zed_agent.llm import AnthropicProvider
    from zed_agent.interface import CLIInterface

    config = AgentConfig(
        llm_provider=AnthropicProvider(api_key="..."),
        working_directory="/path/to/project",
    )
    agent = Agent(config)
    interface = CLIInterface(agent)
    interface.run()
"""

from zed_agent.core.agent import Agent
from zed_agent.core.config import AgentConfig
from zed_agent.core.types import (
    Message,
    UserMessage,
    AgentMessage,
    ToolCall,
    ToolResult,
    TokenUsage,
    StopReason,
    Role,
)
from zed_agent.core.events import (
    AgentEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    ErrorEvent,
    StopEvent,
    RetryEvent,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "Message",
    "UserMessage",
    "AgentMessage",
    "ToolCall",
    "ToolResult",
    "TokenUsage",
    "StopReason",
    "Role",
    "AgentEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "ErrorEvent",
    "StopEvent",
    "RetryEvent",
]
