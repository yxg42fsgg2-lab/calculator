# zed-agent: Pluggable AI Coding Agent Backend

A **pluggable, UI-agnostic AI coding agent backend** extracted and refactored from [Zed's](https://github.com/zed-industries/zed) open-source coder architecture. Designed so you can prototype and test many interfaces and UI/UX concepts with minimal effort.

## Architecture Overview

This package ports the core concepts from Zed's Rust-based agent (`crates/agent/`) into a clean Python package with explicit interface contracts:

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR UI / INTERFACE                    │
│  (CLI, Web, Desktop, Mobile, Chat, IDE Extension, etc.) │
│                                                          │
│  Implement: AgentInterface (1 abstract class, ~7 methods)│
└─────────────────┬───────────────────────────────────────┘
                  │  Events flow down (TextEvent, ToolCallEvent, etc.)
                  │  User input flows up (messages, permissions)
                  │
┌─────────────────▼───────────────────────────────────────┐
│                     AGENT CORE                           │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │    Thread     │  │  System      │  │   Tool        │  │
│  │  (messages,   │  │  Prompt      │  │   Registry    │  │
│  │   history)    │  │  Builder     │  │   (pluggable) │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                          │
│  Orchestrates: user msg → LLM → tool calls → results    │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                   LLM PROVIDER                           │
│                                                          │
│  Implement: LLMProvider (1 abstract class, 2 methods)    │
│                                                          │
│  Built-in: Anthropic, OpenAI, LiteLLM (100+ providers)  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Install

```bash
# Core (no LLM providers)
pip install -e .

# With specific providers
pip install -e ".[anthropic]"    # Anthropic Claude
pip install -e ".[openai]"       # OpenAI GPT
pip install -e ".[litellm]"      # 100+ providers via LiteLLM
pip install -e ".[all]"          # Everything
```

### Minimal Example (5 lines)

```python
import asyncio
from zed_agent import Agent, AgentConfig
from zed_agent.llm import AnthropicProvider
from zed_agent.interface import CLIInterface

agent = Agent(AgentConfig(
    llm_provider=AnthropicProvider(api_key="sk-ant-..."),
    working_directory="/path/to/your/project",
))
asyncio.run(agent.run(CLIInterface()))
```

## How to Plug In Your Own UI

Implementing a custom interface requires implementing **one abstract class** with **7 required methods**. That's it.

### The Interface Contract

```python
from zed_agent.interface.base import AgentInterface

class MyInterface(AgentInterface):
    # === DISPLAY (agent → UI) ===

    async def on_text_event(self, event: TextEvent) -> None:
        """Agent is streaming text. Show it to the user."""

    async def on_tool_call_event(self, event: ToolCallEvent) -> None:
        """Agent wants to call a tool. Show what it's doing."""

    async def on_tool_result_event(self, event: ToolResultEvent) -> None:
        """Tool finished. Show the result."""

    async def on_error_event(self, event: ErrorEvent) -> None:
        """Something went wrong. Show the error."""

    async def on_stop_event(self, event: StopEvent) -> None:
        """Agent finished its turn. Mark as complete."""

    # === INPUT (UI → agent) ===

    async def get_user_input(self) -> Optional[str]:
        """Get the next message from the user. None = quit."""

    async def get_permission(self, event) -> PermissionDecision:
        """Ask user: should this tool run? (for writes, terminal, etc.)"""
```

### Example: Web Interface (FastAPI + WebSocket)

```python
from fastapi import FastAPI, WebSocket
from zed_agent import Agent, AgentConfig
from zed_agent.llm import AnthropicProvider
from zed_agent.interface.base import AgentInterface

app = FastAPI()

class WebSocketInterface(AgentInterface):
    def __init__(self, ws: WebSocket):
        self.ws = ws

    async def on_text_event(self, event):
        await self.ws.send_json({"type": "text", "text": event.text})

    async def on_tool_call_event(self, event):
        await self.ws.send_json({
            "type": "tool_call",
            "name": event.info.name,
            "title": event.info.title,
            "kind": event.info.kind.value,
        })

    async def on_tool_result_event(self, event):
        await self.ws.send_json({
            "type": "tool_result",
            "content": event.result.content[:1000],
            "is_error": event.result.is_error,
        })

    async def on_error_event(self, event):
        await self.ws.send_json({"type": "error", "error": event.error})

    async def on_stop_event(self, event):
        await self.ws.send_json({
            "type": "stop",
            "reason": event.reason.value,
        })

    async def get_user_input(self):
        data = await self.ws.receive_json()
        return data.get("message")

    async def get_permission(self, event):
        await self.ws.send_json({
            "type": "permission_request",
            "tool": event.request.tool_name,
            "title": event.request.title,
        })
        response = await self.ws.receive_json()
        return PermissionDecision(response.get("decision", "allow"))

@app.websocket("/agent")
async def agent_endpoint(ws: WebSocket):
    await ws.accept()
    agent = Agent(AgentConfig(
        llm_provider=AnthropicProvider(api_key="..."),
        working_directory="/project",
    ))
    await agent.run(WebSocketInterface(ws))
```

### Example: Streamlit Interface

```python
import streamlit as st
import asyncio
from zed_agent import Agent, AgentConfig
from zed_agent.llm import AnthropicProvider
from zed_agent.interface.base import AgentInterface

class StreamlitInterface(AgentInterface):
    def __init__(self):
        self.response_placeholder = st.empty()
        self.current_text = ""

    async def on_text_event(self, event):
        self.current_text += event.text
        self.response_placeholder.markdown(self.current_text)

    async def on_tool_call_event(self, event):
        st.info(f"🔧 {event.info.title}")

    async def on_tool_result_event(self, event):
        with st.expander(f"Tool: {event.result.tool_name}"):
            st.code(event.result.content[:2000])

    async def on_error_event(self, event):
        st.error(event.error)

    async def on_stop_event(self, event):
        if event.token_usage:
            st.caption(f"Tokens: {event.token_usage.total_tokens}")

    async def get_user_input(self):
        return st.chat_input("Message the agent...")

    async def get_permission(self, event):
        if st.button(f"Allow {event.request.tool_name}?"):
            return PermissionDecision.ALLOW
        return PermissionDecision.DENY
```

## How to Add Custom Tools

Tools follow the `AgentTool` protocol. Implement **4 properties and 1 method**:

```python
from zed_agent.tools.base import AgentTool
from zed_agent.core.types import ToolKind

class DatabaseQueryTool(AgentTool):
    @property
    def name(self) -> str:
        return "query_database"

    @property
    def description(self) -> str:
        return "Execute a read-only SQL query against the project database."

    @property
    def kind(self) -> ToolKind:
        return ToolKind.READ

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute (SELECT only).",
                },
            },
            "required": ["query"],
        }

    async def run(self, input: dict) -> str:
        query = input["query"]
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        # Execute your query here...
        return "query results..."

# Register it
agent.register_tool(DatabaseQueryTool())
```

## How to Add a New LLM Provider

Implement **2 methods**: `info` (property) and `stream_completion` (async generator):

```python
from zed_agent.llm.base import LLMProvider, LLMProviderInfo, CompletionRequest, CompletionEvent

class MyCustomProvider(LLMProvider):
    @property
    def info(self) -> LLMProviderInfo:
        return LLMProviderInfo(
            id="my_provider",
            name="My Provider",
            model_id="my-model-v1",
            model_name="My Model v1",
            supports_tools=True,
        )

    async def stream_completion(self, request: CompletionRequest):
        # Call your LLM API here
        # Yield CompletionEvent objects:
        #   TextChunk(text="...")       - streaming text
        #   ThinkingChunk(text="...")   - chain of thought
        #   ToolUseEvent(tool_call=...) - tool call request
        #   UsageEvent(usage=...)       - token usage
        #   StopEvent(reason=...)       - generation complete
        ...
```

## Configuration

```python
config = AgentConfig(
    # Required
    llm_provider=AnthropicProvider(api_key="..."),
    working_directory="/path/to/project",

    # Tool configuration
    enable_default_tools=True,           # Include built-in tools
    custom_tools=[MyTool()],             # Add custom tools
    disabled_tools=["web_search"],       # Disable specific tools

    # System prompt
    custom_rules="Always use TypeScript.",
    extra_system_context="This is a React project.",

    # Permissions
    auto_approve_reads=True,             # Auto-approve read tools
    auto_approve_writes=False,           # Ask for write permission
    auto_approve_terminal=False,         # Ask for terminal permission

    # Limits
    max_retries=4,
    max_tool_calls_per_turn=50,
    max_turns=100,

    # Thinking mode (for models that support it)
    enable_thinking=False,
    thinking_effort="medium",
)
```

## Built-in Tools (Ported from Zed)

| Tool | Name | Kind | Description |
|------|------|------|-------------|
| Read File | `read_file` | read | Read file contents with optional line ranges |
| Write File | `write_file` | write | Create or overwrite files |
| Edit File | `edit_file` | write | Search-and-replace edits |
| Terminal | `terminal` | execute | Execute shell commands |
| Grep | `grep` | search | Regex search across files |
| Find Path | `find_path` | search | Find files by name/pattern |
| List Directory | `list_directory` | read | List directory contents |

## Event Types

Events flow from the agent to your interface:

| Event | When | Key Fields |
|-------|------|------------|
| `TextEvent` | Agent is streaming text | `text`, `is_complete` |
| `ThinkingEvent` | Agent is reasoning | `text`, `is_complete` |
| `ToolCallEvent` | Agent wants a tool | `tool_call`, `info` |
| `ToolCallStartEvent` | Tool execution started | `tool_call_id`, `title` |
| `ToolCallProgressEvent` | Tool progress update | `content`, `locations` |
| `ToolResultEvent` | Tool finished | `result` (with `is_error`) |
| `PermissionRequestEvent` | Needs user OK | `request` |
| `RetryEvent` | Retrying after error | `attempt`, `delay_seconds` |
| `ErrorEvent` | Something went wrong | `error`, `is_retryable` |
| `StopEvent` | Turn complete | `reason`, `token_usage` |
| `TitleEvent` | Title generated | `title` |
| `StatusEvent` | Status change | `status`, `detail` |

## Project Structure

```
zed_agent/
├── __init__.py              # Public API
├── core/
│   ├── agent.py             # Main Agent orchestrator
│   ├── config.py            # AgentConfig
│   ├── events.py            # Event types (agent → UI)
│   ├── thread.py            # Conversation history
│   └── types.py             # Core data types
├── llm/
│   ├── base.py              # LLMProvider abstract class
│   ├── anthropic_provider.py
│   ├── openai_provider.py
│   └── litellm_provider.py  # Universal (100+ providers)
├── tools/
│   ├── base.py              # AgentTool abstract class
│   ├── registry.py          # Tool registry
│   ├── read_file.py         # Built-in tools...
│   ├── write_file.py
│   ├── edit_file.py
│   ├── terminal.py
│   ├── grep.py
│   ├── find_path.py
│   └── list_directory.py
├── prompts/
│   └── system.py            # System prompt builder
├── interface/
│   ├── base.py              # AgentInterface (IMPLEMENT THIS)
│   └── cli.py               # Reference CLI implementation
tests/
├── test_tools.py
├── test_thread.py
├── test_interface.py
└── test_system_prompt.py
```

## Relationship to Zed's Codebase

This package ports the following Zed components:

| Zed (Rust) | This Package (Python) | Description |
|---|---|---|
| `crates/agent/src/thread.rs` → `Thread` | `core/thread.py` → `Thread` | Conversation state management |
| `crates/agent/src/thread.rs` → `AgentTool` trait | `tools/base.py` → `AgentTool` | Tool interface contract |
| `crates/agent/src/agent.rs` → `NativeAgent` | `core/agent.py` → `Agent` | Main orchestrator |
| `crates/agent/src/agent.rs` → `NativeAgentConnection` | `interface/base.py` → `AgentInterface` | UI bridge protocol |
| `crates/language_model/` → `LanguageModel` trait | `llm/base.py` → `LLMProvider` | LLM abstraction |
| `crates/agent/src/templates/system_prompt.hbs` | `prompts/system.py` | System prompt generation |
| `crates/agent/src/tools/*.rs` | `tools/*.py` | Built-in tool implementations |
| `crates/agent/src/thread.rs` → `ThreadEvent` enum | `core/events.py` | Event types |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
