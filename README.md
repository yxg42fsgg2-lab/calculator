# zed-agent — Faithful Python Port of Zed's AI Coding Agent

A 1:1 Python port of the agent backend from [zed-industries/zed](https://github.com/zed-industries/zed) (`crates/agent/`). Same architecture, same control flow, same naming — translated from Rust to Python with idiomatic async equivalents.

## Source Mapping

Every file and type in this package maps directly to its Zed counterpart:

| This Package | Zed Source (Rust) | What It Contains |
|---|---|---|
| `thread.py` | `crates/agent/src/thread.rs` | **The core.** Thread, Message types, AgentTool trait, ThreadEventStream, ToolCallEventStream, RunningTurn, the agentic loop (`run_turn_internal`), retry logic, pending message/flush pattern |
| `agent.py` | `crates/agent/src/agent.rs` | Agent (≈NativeAgent) — session/thread factory with model + tools + system prompt |
| `templates.py` | `crates/agent/src/templates.rs` + `templates/system_prompt.hbs` | System prompt rendering |
| `interface.py` | `crates/acp_thread/` (AgentConnection trait) | **NEW** — the pluggable UI contract |
| `language_model/model.py` | `crates/language_model/src/language_model.rs` | LanguageModel trait, CompletionEvent variants, StopReason, TokenUsage |
| `language_model/request.py` | `crates/language_model/src/request.rs` | LanguageModelRequest, MessageContent, RequestMessage |
| `language_model/providers/anthropic.py` | `crates/anthropic/` | Anthropic Claude provider |
| `tools/*.py` | `crates/agent/src/tools/*.rs` | All built-in tools |

## Rust → Python Idiom Map

| Rust (Zed) | Python (this port) |
|---|---|
| `Entity<T>` / `WeakEntity<T>` | plain object reference |
| `Task<T>` | `asyncio.Task[T]` |
| `mpsc::unbounded()` | `asyncio.Queue()` |
| `watch::channel(T)` | `WatchChannel` class |
| `FuturesUnordered` | `asyncio.gather()` |
| `Arc<dyn LanguageModel>` | `LanguageModel` (ABC instance) |
| `BTreeMap<SharedString, Arc<dyn AnyAgentTool>>` | `dict[str, AnyAgentTool]` |
| `Context<Self>` / `App` / `cx.notify()` | not needed (no GPUI) |
| `cx.spawn(async move \|this, cx\| ...)` | `asyncio.create_task(...)` |
| `futures::select!` | `asyncio` event loop |
| `Serialize` / `Deserialize` | `dataclass` |
| `#[derive(JsonSchema)]` | `input_schema()` returning dict |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              YOUR UI (implement AgentInterface)       │
│  CLIInterface is the reference implementation        │
└──────────────┬──────────────────────────────────────┘
               │ consumes asyncio.Queue[ThreadEvent]
               │ (same pattern as Zed's mpsc channel)
┌──────────────▼──────────────────────────────────────┐
│                     Thread                           │
│  (faithful port of thread.rs)                        │
│                                                      │
│  send("msg") → run_turn_internal():                  │
│    loop {                                            │
│      build_completion_request()                      │
│      model.stream_completion() →                     │
│        handle_completion_event() for each event:     │
│          Text → pending_message.content += AMCText   │
│          ToolUse → spawn tool task                   │
│          Stop → break or raise                       │
│      asyncio.gather(*tool_tasks)  ← PARALLEL        │
│      flush_pending_message()                         │
│      if error → retry_strategy_for() → sleep → loop  │
│      if no tools → end_turn                          │
│      if tools → loop (model processes results)       │
│    }                                                 │
│                                                      │
│  ThreadEventStream → pushes to asyncio.Queue         │
│  ToolCallEventStream → per-tool with cancellation    │
│  RunningTurn → holds task + cancellation channel     │
│  pending_message / flush_pending_message() pattern   │
└──────────────┬──────────────────────────────────────┘
               │ calls
┌──────────────▼──────────────────────────────────────┐
│              LanguageModel (ABC)                      │
│  stream_completion() → AsyncIterator[CompletionEvent]│
│                                                      │
│  Built-in: AnthropicModel                            │
│  (implement for any provider)                        │
└─────────────────────────────────────────────────────┘
```

## Quick Start

```python
import asyncio
from zed_agent.agent import Agent
from zed_agent.interface import CLIInterface
from zed_agent.language_model.providers.anthropic import AnthropicModel

model = AnthropicModel(api_key="sk-ant-...")
agent = Agent(model=model, working_directory="/path/to/project")
asyncio.run(CLIInterface().run(agent.thread))
```

## Plugging In Your Own UI

Implement `AgentInterface` — 6 required methods:

```python
from zed_agent.interface import AgentInterface

class MyUI(AgentInterface):
    async def on_text(self, text: str) -> None:
        """Agent is streaming text."""

    async def on_tool_call(self, event) -> None:
        """Agent invoked a tool (event.tool_name, event.title, event.kind)."""

    async def on_stop(self, reason: str) -> None:
        """Turn finished ("end_turn", "max_tokens", "cancelled", etc.)."""

    async def on_error(self, error: Exception) -> None:
        """Something went wrong."""

    async def authorize_tool(self, event) -> bool:
        """Allow this tool call? Return True/False."""

    async def get_user_input(self) -> str | None:
        """Get user message. None = quit."""
```

Then: `await MyUI().run(agent.thread)`

Or consume the event queue directly (lower level):

```python
queue = thread.send("Fix the bug in main.py")
while True:
    event = await queue.get()
    if event is None:
        break
    if isinstance(event, ThreadEventAgentText):
        print(event.text, end="")
    elif isinstance(event, ThreadEventToolCall):
        print(f"[tool: {event.tool_name}]")
    elif isinstance(event, ThreadEventStop):
        print(f"[done: {event.reason}]")
```

## Adding Custom Tools

Same pattern as Zed's `AgentTool` trait:

```python
from zed_agent.thread import AgentTool, ToolCallEventStream, ToolKind

class MyTool(AgentTool):
    NAME = "my_tool"

    def description(self) -> str:
        return "Does something useful."

    def kind(self) -> ToolKind:
        return ToolKind.Read

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    async def run(self, input: dict, event_stream: ToolCallEventStream) -> str:
        # event_stream.update_fields(...) for progress
        # event_stream.is_cancelled() to check cancellation
        return f"Result for {input['query']}"

agent.thread.add_tool(MyTool())
```

## Adding a New LLM Provider

Implement `LanguageModel` (same shape as Zed's trait):

```python
from zed_agent.language_model import LanguageModel, TextEvent, StopEvent, StopReason

class MyProvider(LanguageModel):
    def id(self): return LanguageModelId("my-model")
    def name(self): return "My Model"
    def provider_id(self): return LanguageModelProviderId("my-provider")
    def supports_images(self): return False
    def supports_tools(self): return True

    async def stream_completion(self, request):
        # Yield LanguageModelCompletionEvent variants:
        yield TextEvent(text="Hello!")
        yield StopEvent(reason=StopReason.EndTurn)
```

## Built-in Tools

| Tool | Zed Source | Kind |
|---|---|---|
| `read_file` | `read_file_tool.rs` | Read |
| `edit_file` | `edit_file_tool.rs` | Write |
| `create_file` | `create_file_parser.rs` | Write |
| `terminal` | `terminal_tool.rs` | Execute |
| `grep` | `grep_tool.rs` | Search |
| `find_path` | `find_path_tool.rs` | Search |
| `list_directory` | `list_directory_tool.rs` | Read |
| `delete_path` | `delete_path_tool.rs` | Write |
| `copy_path` | `copy_path_tool.rs` | Write |
| `move_path` | `move_path_tool.rs` | Write |
| `create_directory` | `create_directory_tool.rs` | Write |
| `now` | `now_tool.rs` | Other |
| `fetch` | `fetch_tool.rs` | Read |

## Key Architectural Patterns Preserved

1. **Pending message / flush** — Agent text and tool calls accumulate in `pending_message: Optional[AgentMessage]`. When a turn boundary is reached, `flush_pending_message()` moves it to `messages[]`, auto-filling "Tool canceled" for any tool_use without a result. (thread.rs L2322-2354)

2. **Parallel tool execution** — All tool calls from a single model response run concurrently via `asyncio.gather()`, matching Zed's `FuturesUnordered`. (thread.rs L1730-1803)

3. **Channel-based events** — `ThreadEventStream` wraps `asyncio.Queue`, matching Zed's `mpsc::UnboundedSender<Result<ThreadEvent>>`. The UI consumes the queue. (thread.rs L2914-2999)

4. **Per-tool event stream** — Each running tool gets a `ToolCallEventStream` that provides cancellation detection and progress updates, matching Zed's `ToolCallEventStream`. (thread.rs L3002-3055)

5. **Retry strategies** — `retry_strategy_for()` returns `ExponentialBackoff` or `FixedDelay` per error type, matching Zed's `Thread::retry_strategy_for()`. (thread.rs L2622-2721)

6. **Cancellation propagation** — `WatchChannel(bool)` mirrors Rust's `watch::channel`. `RunningTurn.cancel()` sends `True`, which propagates to the agentic loop and all running tools. (thread.rs L2725-2746)

7. **Type-erased tools** — `AnyAgentTool` wraps typed `AgentTool` instances, matching Zed's `Erased<Arc<T>>` pattern. (thread.rs L2820-2911)

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
