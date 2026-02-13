"""
Interface protocol - the contract between agent backend and UI frontends.

To connect a new UI to the agent backend, implement the AgentInterface
abstract class. This is the ONLY thing you need to implement.

The interface receives events from the agent and provides user input back.
"""

from zed_agent.interface.base import AgentInterface
from zed_agent.interface.cli import CLIInterface

__all__ = ["AgentInterface", "CLIInterface"]
