#!/usr/bin/env python3
"""
Example: Run the agent with the CLI interface.

Usage:
    # With Anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/cli_agent.py /path/to/project

    # With OpenAI
    export OPENAI_API_KEY="sk-..."
    python examples/cli_agent.py /path/to/project --provider openai

    # With any LiteLLM-supported provider
    python examples/cli_agent.py /path/to/project --provider litellm --model ollama/llama3
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zed_agent import Agent, AgentConfig
from zed_agent.interface import CLIInterface


def create_provider(provider_name: str, model: str | None = None):
    """Create an LLM provider based on the name."""
    if provider_name == "anthropic":
        from zed_agent.llm import AnthropicProvider

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable is required")
            sys.exit(1)
        return AnthropicProvider(
            api_key=api_key, model=model or "claude-sonnet-4-20250514"
        )

    elif provider_name == "openai":
        from zed_agent.llm import OpenAIProvider

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable is required")
            sys.exit(1)
        return OpenAIProvider(api_key=api_key, model=model or "gpt-4o")

    elif provider_name == "litellm":
        from zed_agent.llm import LiteLLMProvider

        return LiteLLMProvider(model=model or "claude-sonnet-4-20250514")

    else:
        print(f"Unknown provider: {provider_name}")
        print("Available: anthropic, openai, litellm")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the AI coding agent")
    parser.add_argument(
        "project_dir",
        nargs="?",
        default=".",
        help="Path to the project directory",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        help="LLM provider (anthropic, openai, litellm)",
    )
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument(
        "--show-thinking", action="store_true", help="Show model thinking"
    )
    parser.add_argument(
        "--auto-approve", action="store_true", help="Auto-approve all tool calls"
    )
    parser.add_argument(
        "--rules", default=None, help="Custom rules for the agent"
    )

    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    if not os.path.isdir(project_dir):
        print(f"Error: {project_dir} is not a directory")
        sys.exit(1)

    provider = create_provider(args.provider, args.model)

    config = AgentConfig(
        llm_provider=provider,
        working_directory=project_dir,
        auto_approve_writes=args.auto_approve,
        auto_approve_terminal=args.auto_approve,
        custom_rules=args.rules,
    )

    agent = Agent(config)
    interface = CLIInterface(
        show_thinking=args.show_thinking, show_tool_details=True
    )

    asyncio.run(agent.run(interface))


if __name__ == "__main__":
    main()
