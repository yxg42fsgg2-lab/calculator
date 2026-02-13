"""
LLM provider abstraction layer.

This module defines the contract that any LLM provider must implement,
and provides concrete implementations for popular providers.

To add a new provider:
    1. Subclass LLMProvider
    2. Implement stream_completion() and count_tokens()
    3. Register it with the agent via AgentConfig

The LiteLLMProvider is provided as a "universal" adapter that works
with any provider supported by the litellm library.
"""

from zed_agent.llm.base import LLMProvider, LLMProviderInfo, CompletionRequest
from zed_agent.llm.anthropic_provider import AnthropicProvider
from zed_agent.llm.openai_provider import OpenAIProvider
from zed_agent.llm.litellm_provider import LiteLLMProvider

__all__ = [
    "LLMProvider",
    "LLMProviderInfo",
    "CompletionRequest",
    "AnthropicProvider",
    "OpenAIProvider",
    "LiteLLMProvider",
]
