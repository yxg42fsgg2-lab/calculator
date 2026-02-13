"""
Faithful port of crates/agent/src/thread.rs.

This is the core of the Zed agent. It contains:

1. Message types (Message, UserMessage, AgentMessage, AgentMessageContent)
2. Thread — the stateful conversation manager and agentic loop
3. ThreadEventStream — channel-based event emitter (wraps asyncio.Queue)
4. ToolCallEventStream — per-tool-call event sub-stream with cancellation
5. RunningTurn — holds the running turn's task + cancellation handle
6. AgentTool — the tool trait (ABC)
7. AnyAgentTool — type-erased tool wrapper (like Erased<Arc<T>>)

Rust → Python idiom key:
  Entity<T>                  → plain object (Python GC handles lifetimes)
  WeakEntity<T>              → weakref.ref or just hold object
  Task<T>                    → asyncio.Task[T]
  mpsc::unbounded()          → asyncio.Queue()
  watch::channel(bool)       → WatchChannel (see below)
  FuturesUnordered           → asyncio.gather()
  Arc<dyn Trait>             → Trait ABC instance
  BTreeMap<K,V>              → dict[K,V] (Python 3.7+ dicts are ordered)
  SharedString               → str
  Context<Self> / App        → not needed (no GPUI reactive system)
  cx.notify()                → not needed
  cx.spawn(async move |…|…)  → asyncio.create_task(…)
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from zed_agent.language_model.model import (
    LanguageModel,
    LanguageModelCompletionError,
    LanguageModelCompletionEvent,
    LanguageModelToolResult,
    LanguageModelToolResultContent,
    LanguageModelToolUse,
    LanguageModelToolUseId,
    PromptTooLargeError,
    RateLimitExceededError,
    Role,
    ServerOverloadedError,
    ApiInternalServerError,
    AuthenticationError,
    StopReason,
    TextEvent as LMTextEvent,
    ThinkingEvent as LMThinkingEvent,
    RedactedThinkingEvent as LMRedactedThinkingEvent,
    ToolUseEvent as LMToolUseEvent,
    ToolUseJsonParseErrorEvent as LMToolUseJsonParseErrorEvent,
    UsageUpdateEvent as LMUsageUpdateEvent,
    StopEvent as LMStopEvent,
    StartMessageEvent as LMStartMessageEvent,
    StartedEvent as LMStartedEvent,
    QueuedEvent as LMQueuedEvent,
    ReasoningDetailsEvent as LMReasoningDetailsEvent,
    TokenUsage,
)
from zed_agent.language_model.request import (
    LanguageModelRequest,
    LanguageModelRequestMessage,
    LanguageModelRequestTool,
    MessageContent,
    TextContent,
    ThinkingContent,
    RedactedThinkingContent,
    ToolUseContent,
    ToolResultContent,
)

logger = logging.getLogger(__name__)

# ── Constants (from thread.rs) ─────────────────────────────

TOOL_CANCELED_MESSAGE = "Tool canceled by user"
MAX_TOOL_NAME_LENGTH = 64
MAX_SUBAGENT_DEPTH = 4
MAX_PARALLEL_SUBAGENTS = 8
MAX_RETRY_ATTEMPTS = 4
BASE_RETRY_DELAY = 5.0  # seconds


# ╔══════════════════════════════════════════════════════════╗
# ║  1. Message types — mirrors thread.rs lines 109-553     ║
# ╚══════════════════════════════════════════════════════════╝

# ── UserMessageContent ─────────────────────────────────────

@dataclass
class UserMessageText:
    text: str

@dataclass
class UserMessageImage:
    source: str  # base64

@dataclass
class UserMessageMention:
    uri: str
    content: str

UserMessageContent = UserMessageText | UserMessageImage | UserMessageMention


# ── UserMessage ────────────────────────────────────────────

@dataclass
class UserMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: list[UserMessageContent] = field(default_factory=list)

    def to_request(self) -> LanguageModelRequestMessage:
        """Mirrors UserMessage.to_request() in thread.rs."""
        parts: list[MessageContent] = []
        for chunk in self.content:
            if isinstance(chunk, UserMessageText):
                parts.append(TextContent(text=chunk.text))
            elif isinstance(chunk, UserMessageImage):
                parts.append(TextContent(text="[image]"))
            elif isinstance(chunk, UserMessageMention):
                parts.append(TextContent(text=chunk.content))
        return LanguageModelRequestMessage(role=Role.User, content=parts)

    def text(self) -> str:
        return "".join(
            c.text if isinstance(c, UserMessageText) else ""
            for c in self.content
        )


# ── AgentMessageContent (enum) ─────────────────────────────

@dataclass
class AMCText:
    """AgentMessageContent::Text"""
    text: str

@dataclass
class AMCThinking:
    """AgentMessageContent::Thinking"""
    text: str
    signature: Optional[str] = None

@dataclass
class AMCRedactedThinking:
    """AgentMessageContent::RedactedThinking"""
    data: str

@dataclass
class AMCToolUse:
    """AgentMessageContent::ToolUse"""
    tool_use: LanguageModelToolUse

AgentMessageContent = AMCText | AMCThinking | AMCRedactedThinking | AMCToolUse


# ── AgentMessage ───────────────────────────────────────────

@dataclass
class AgentMessage:
    """Mirrors thread.rs AgentMessage struct.

    content: Vec<AgentMessageContent>
    tool_results: IndexMap<LanguageModelToolUseId, LanguageModelToolResult>
    reasoning_details: Option<serde_json::Value>
    """
    content: list[AgentMessageContent] = field(default_factory=list)
    tool_results: OrderedDict[LanguageModelToolUseId, LanguageModelToolResult] = field(
        default_factory=OrderedDict
    )
    reasoning_details: Optional[Any] = None

    def to_request(self) -> list[LanguageModelRequestMessage]:
        """Mirrors AgentMessage.to_request() — produces assistant msg + tool results msg."""
        assistant_content: list[MessageContent] = []
        for chunk in self.content:
            if isinstance(chunk, AMCText):
                assistant_content.append(TextContent(text=chunk.text))
            elif isinstance(chunk, AMCThinking):
                assistant_content.append(
                    ThinkingContent(text=chunk.text, signature=chunk.signature)
                )
            elif isinstance(chunk, AMCRedactedThinking):
                assistant_content.append(RedactedThinkingContent(data=chunk.data))
            elif isinstance(chunk, AMCToolUse):
                # Only include tool_use if we have the result
                if chunk.tool_use.id in self.tool_results:
                    assistant_content.append(ToolUseContent(tool_use=chunk.tool_use))

        user_content: list[MessageContent] = []
        for tool_result in self.tool_results.values():
            tr = tool_result
            # API fails on empty tool result content
            if tr.content.is_empty():
                tr = LanguageModelToolResult(
                    tool_use_id=tr.tool_use_id,
                    tool_name=tr.tool_name,
                    is_error=tr.is_error,
                    content=LanguageModelToolResultContent.from_str(
                        "<Tool returned an empty string>"
                    ),
                    output=tr.output,
                )
            user_content.append(ToolResultContent(tool_result=tr))

        messages: list[LanguageModelRequestMessage] = []
        if assistant_content:
            messages.append(LanguageModelRequestMessage(
                role=Role.Assistant,
                content=assistant_content,
                reasoning_details=self.reasoning_details,
            ))
        if user_content:
            messages.append(LanguageModelRequestMessage(
                role=Role.User,
                content=user_content,
            ))
        return messages

    def to_markdown(self) -> str:
        parts: list[str] = []
        for item in self.content:
            if isinstance(item, AMCText):
                parts.append(item.text + "\n")
            elif isinstance(item, AMCThinking):
                parts.append(f"<think>{item.text}</think>\n")
            elif isinstance(item, AMCRedactedThinking):
                parts.append("<redacted_thinking />\n")
            elif isinstance(item, AMCToolUse):
                parts.append(
                    f"**Tool Use**: {item.tool_use.name} (ID: {item.tool_use.id})\n"
                )
                parts.append(f"```json\n{json.dumps(item.tool_use.input, indent=2)}\n```\n")
        for tr in self.tool_results.values():
            parts.append(f"**Tool Result**: {tr.tool_name} (ID: {tr.tool_use_id})\n\n")
            if tr.is_error:
                parts.append("**ERROR:**\n")
            parts.append(tr.content.text + "\n")
        return "".join(parts)


# ── Message enum ───────────────────────────────────────────

class MessageKind(str, Enum):
    User = "user"
    Agent = "agent"
    Resume = "resume"

@dataclass
class Message:
    """Mirrors thread.rs Message enum (User | Agent | Resume)."""
    kind: MessageKind
    user_message: Optional[UserMessage] = None
    agent_message: Optional[AgentMessage] = None

    @staticmethod
    def user(msg: UserMessage) -> Message:
        return Message(kind=MessageKind.User, user_message=msg)

    @staticmethod
    def agent(msg: AgentMessage) -> Message:
        return Message(kind=MessageKind.Agent, agent_message=msg)

    @staticmethod
    def resume() -> Message:
        return Message(kind=MessageKind.Resume)

    def to_request(self) -> list[LanguageModelRequestMessage]:
        if self.kind == MessageKind.User and self.user_message:
            msg = self.user_message.to_request()
            return [msg] if msg.content else []
        elif self.kind == MessageKind.Agent and self.agent_message:
            return self.agent_message.to_request()
        elif self.kind == MessageKind.Resume:
            return [LanguageModelRequestMessage(
                role=Role.User,
                content=[TextContent(text="Continue where you left off")],
            )]
        return []

    def to_markdown(self) -> str:
        if self.kind == MessageKind.User and self.user_message:
            return self.user_message.text() + "\n"
        elif self.kind == MessageKind.Agent and self.agent_message:
            return self.agent_message.to_markdown()
        elif self.kind == MessageKind.Resume:
            return "[resume]\n"
        return ""


# ╔══════════════════════════════════════════════════════════╗
# ║  2. WatchChannel — equivalent of tokio::sync::watch     ║
# ╚══════════════════════════════════════════════════════════╝

class WatchChannel:
    """Mirrors Rust's watch::channel. Holds a value; waiters are notified on change."""

    def __init__(self, initial: Any = None):
        self._value = initial
        self._event = asyncio.Event()

    def send(self, value: Any) -> None:
        self._value = value
        self._event.set()
        self._event = asyncio.Event()  # reset for next wait

    def borrow(self) -> Any:
        return self._value

    async def changed(self) -> bool:
        """Block until the value changes. Returns True."""
        await self._event.wait()
        return True


# ╔══════════════════════════════════════════════════════════╗
# ║  3. ThreadEvent — mirrors thread.rs ThreadEvent enum     ║
# ╚══════════════════════════════════════════════════════════╝

class ToolKind(str, Enum):
    """Mirrors acp::ToolKind."""
    Read = "read"
    Write = "write"
    Execute = "execute"
    Search = "search"
    Other = "other"


class ToolCallStatus(str, Enum):
    Pending = "pending"
    InProgress = "in_progress"
    Completed = "completed"
    Failed = "failed"


@dataclass
class ToolCallUpdateFields:
    title: Optional[str] = None
    status: Optional[ToolCallStatus] = None
    kind: Optional[ToolKind] = None
    raw_input: Optional[Any] = None
    raw_output: Optional[Any] = None
    content: Optional[list[Any]] = None
    locations: Optional[list[str]] = None


@dataclass
class ThreadEventUserMessage:
    message: UserMessage

@dataclass
class ThreadEventAgentText:
    text: str

@dataclass
class ThreadEventAgentThinking:
    text: str

@dataclass
class ThreadEventToolCall:
    """Initial tool call. Carries id, tool_name, title, kind, raw_input."""
    id: str
    tool_name: str
    title: str
    kind: ToolKind
    raw_input: Any = None

@dataclass
class ThreadEventToolCallUpdate:
    """Incremental update to a tool call (status, output, content, etc.)."""
    tool_call_id: str
    fields: ToolCallUpdateFields

@dataclass
class ThreadEventToolCallAuthorization:
    """Tool needs user permission. Carries a Future the tool awaits on."""
    tool_call_id: str
    tool_name: str
    title: str
    # The authorization result is communicated through the response_future
    response_future: asyncio.Future  # resolves to bool (True=allow)

@dataclass
class ThreadEventRetry:
    last_error: str
    attempt: int
    max_attempts: int
    delay_seconds: float

@dataclass
class ThreadEventStop:
    reason: str  # "end_turn", "max_tokens", "cancelled", "refusal"

ThreadEvent = (
    ThreadEventUserMessage
    | ThreadEventAgentText
    | ThreadEventAgentThinking
    | ThreadEventToolCall
    | ThreadEventToolCallUpdate
    | ThreadEventToolCallAuthorization
    | ThreadEventRetry
    | ThreadEventStop
)


# ╔══════════════════════════════════════════════════════════╗
# ║  4. ThreadEventStream — mirrors the Rust struct          ║
# ╚══════════════════════════════════════════════════════════╝

class ThreadEventStream:
    """Mirrors thread.rs ThreadEventStream(mpsc::UnboundedSender<Result<ThreadEvent>>).

    Wraps an asyncio.Queue to push events to consumers (the UI layer).
    """

    def __init__(self, queue: asyncio.Queue[ThreadEvent | Exception | None]):
        self._queue = queue

    def send_user_message(self, message: UserMessage) -> None:
        self._queue.put_nowait(ThreadEventUserMessage(message=message))

    def send_text(self, text: str) -> None:
        self._queue.put_nowait(ThreadEventAgentText(text=text))

    def send_thinking(self, text: str) -> None:
        self._queue.put_nowait(ThreadEventAgentThinking(text=text))

    def send_tool_call(
        self,
        id: LanguageModelToolUseId,
        tool_name: str,
        title: str,
        kind: ToolKind,
        raw_input: Any,
    ) -> None:
        self._queue.put_nowait(ThreadEventToolCall(
            id=str(id), tool_name=tool_name, title=title, kind=kind, raw_input=raw_input,
        ))

    def update_tool_call_fields(
        self,
        tool_use_id: LanguageModelToolUseId,
        fields: ToolCallUpdateFields,
    ) -> None:
        self._queue.put_nowait(ThreadEventToolCallUpdate(
            tool_call_id=str(tool_use_id), fields=fields,
        ))

    def send_retry(self, last_error: str, attempt: int, max_attempts: int, delay: float) -> None:
        self._queue.put_nowait(ThreadEventRetry(
            last_error=last_error, attempt=attempt, max_attempts=max_attempts,
            delay_seconds=delay,
        ))

    def send_stop(self, reason: str) -> None:
        self._queue.put_nowait(ThreadEventStop(reason=reason))

    def send_canceled(self) -> None:
        self.send_stop("cancelled")

    def send_error(self, error: Exception) -> None:
        self._queue.put_nowait(error)


# ╔══════════════════════════════════════════════════════════╗
# ║  5. ToolCallEventStream — per-tool event sub-stream      ║
# ╚══════════════════════════════════════════════════════════╝

class ToolCallEventStream:
    """Mirrors thread.rs ToolCallEventStream.

    Each running tool gets one of these. It tags events with the tool_use_id
    and provides cancellation detection + authorization request.
    """

    def __init__(
        self,
        tool_use_id: LanguageModelToolUseId,
        stream: ThreadEventStream,
        cancellation: WatchChannel,  # borrow() -> bool
    ):
        self.tool_use_id = tool_use_id
        self._stream = stream
        self._cancellation = cancellation

    def update_fields(self, fields: ToolCallUpdateFields) -> None:
        """Send a partial update for this tool call."""
        self._stream.update_tool_call_fields(self.tool_use_id, fields)

    async def cancelled_by_user(self) -> None:
        """Awaitable — resolves when the user cancels. Mirrors Zed's cancelled_by_user()."""
        while not self._cancellation.borrow():
            await self._cancellation.changed()

    def is_cancelled(self) -> bool:
        return bool(self._cancellation.borrow())

    async def authorize(self, title: str) -> bool:
        """Request user authorization. Returns True if allowed.

        Mirrors the ToolCallEventStream.authorize() in Zed which pushes a
        ToolCallAuthorization event and waits on a oneshot::Sender response.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._stream._queue.put_nowait(ThreadEventToolCallAuthorization(
            tool_call_id=str(self.tool_use_id),
            tool_name="",  # filled by caller
            title=title,
            response_future=future,
        ))
        return await future


# ╔══════════════════════════════════════════════════════════╗
# ║  6. AgentTool trait + AnyAgentTool type erasure          ║
# ╚══════════════════════════════════════════════════════════╝

class AgentTool(abc.ABC):
    """Mirrors thread.rs trait AgentTool.

    In Zed, this is a generic trait with associated types:
        type Input: Deserialize + Serialize + JsonSchema
        type Output: Deserialize + Serialize + Into<LanguageModelToolResultContent>

    In Python we use dict for Input and str for Output (matching the JSON/string
    nature of the protocol). The tool_call_event_stream is passed to run()
    just like in Zed.
    """

    @property
    @abc.abstractmethod
    def NAME(self) -> str:
        ...

    @abc.abstractmethod
    def description(self) -> str:
        """Return tool description. In Zed this comes from JsonSchema derive."""
        ...

    @abc.abstractmethod
    def kind(self) -> ToolKind:
        ...

    @abc.abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """Return JSON Schema for the input. Mirrors AgentTool::input_schema()."""
        ...

    def initial_title(self, input: dict[str, Any]) -> str:
        """Mirrors AgentTool::initial_title(). Override for descriptive titles."""
        return self.NAME

    @abc.abstractmethod
    async def run(
        self,
        input: dict[str, Any],
        event_stream: ToolCallEventStream,
    ) -> str:
        """Execute the tool. Mirrors AgentTool::run(self: Arc<Self>, input, event_stream, cx)."""
        ...

    def replay(
        self,
        input: dict[str, Any],
        output: Any,
        event_stream: ToolCallEventStream,
    ) -> None:
        """Emit events for a previous execution. Mirrors AgentTool::replay()."""
        pass


class AnyAgentTool:
    """Mirrors thread.rs AnyAgentTool trait + Erased<Arc<T>> wrapper.

    Type-erases AgentTool so Thread can store them in a dict[str, AnyAgentTool].
    """

    def __init__(self, tool: AgentTool):
        self._tool = tool

    @property
    def name(self) -> str:
        return self._tool.NAME

    def description(self) -> str:
        return self._tool.description()

    def kind(self) -> ToolKind:
        return self._tool.kind()

    def initial_title(self, raw_input: Any) -> str:
        if isinstance(raw_input, dict):
            return self._tool.initial_title(raw_input)
        try:
            return self._tool.initial_title(json.loads(raw_input) if isinstance(raw_input, str) else {})
        except Exception:
            return self._tool.NAME

    def input_schema(self) -> dict[str, Any]:
        return self._tool.input_schema()

    async def run(
        self,
        raw_input: Any,
        event_stream: ToolCallEventStream,
    ) -> LanguageModelToolResult:
        """Run the tool, returning a LanguageModelToolResult.

        Mirrors Erased<Arc<T>>::run() which deserializes input, calls
        T::run(), serializes output.
        """
        parsed_input = raw_input if isinstance(raw_input, dict) else {}
        try:
            output_str = await self._tool.run(parsed_input, event_stream)
            return LanguageModelToolResult(
                tool_use_id=event_stream.tool_use_id,
                tool_name=self.name,
                is_error=False,
                content=LanguageModelToolResultContent(text=output_str),
                output=output_str,
            )
        except Exception as e:
            return LanguageModelToolResult(
                tool_use_id=event_stream.tool_use_id,
                tool_name=self.name,
                is_error=True,
                content=LanguageModelToolResultContent(text=str(e)),
                output=str(e),
            )


# ╔══════════════════════════════════════════════════════════╗
# ║  7. Retry logic — mirrors thread.rs retry_strategy_for   ║
# ╚══════════════════════════════════════════════════════════╝

class RetryStrategy:
    pass

@dataclass
class ExponentialBackoff(RetryStrategy):
    initial_delay: float  # seconds
    max_attempts: int

@dataclass
class FixedDelay(RetryStrategy):
    delay: float
    max_attempts: int


def retry_strategy_for(error: Exception) -> Optional[RetryStrategy]:
    """Mirrors Thread::retry_strategy_for(). Returns None if error is not retryable."""
    if isinstance(error, PromptTooLargeError):
        return None
    if isinstance(error, AuthenticationError):
        return None
    if isinstance(error, RateLimitExceededError):
        delay = error.retry_after if error.retry_after else BASE_RETRY_DELAY
        return FixedDelay(delay=delay, max_attempts=MAX_RETRY_ATTEMPTS)
    if isinstance(error, ServerOverloadedError):
        delay = error.retry_after if error.retry_after else BASE_RETRY_DELAY
        return FixedDelay(delay=delay, max_attempts=MAX_RETRY_ATTEMPTS)
    if isinstance(error, ApiInternalServerError):
        return FixedDelay(delay=BASE_RETRY_DELAY, max_attempts=3)
    # Generic errors — retry a couple times
    if isinstance(error, LanguageModelCompletionError):
        return FixedDelay(delay=BASE_RETRY_DELAY, max_attempts=2)
    return FixedDelay(delay=BASE_RETRY_DELAY, max_attempts=2)


# ╔══════════════════════════════════════════════════════════╗
# ║  8. CompletionError — mirrors thread.rs CompletionError  ║
# ╚══════════════════════════════════════════════════════════╝

class CompletionError(Exception):
    pass

class MaxTokensError(CompletionError):
    pass

class RefusalError(CompletionError):
    pass


# ╔══════════════════════════════════════════════════════════╗
# ║  9. RunningTurn — mirrors thread.rs RunningTurn          ║
# ╚══════════════════════════════════════════════════════════╝

@dataclass
class RunningTurn:
    """Mirrors thread.rs RunningTurn.

    Holds the asyncio.Task for the turn, the event stream, the tool set,
    and the cancellation channel.
    """
    task: asyncio.Task
    event_stream: ThreadEventStream
    tools: dict[str, AnyAgentTool]
    cancellation_tx: WatchChannel  # send(True) to cancel

    def cancel(self) -> asyncio.Task:
        """Cancel the turn. Mirrors RunningTurn::cancel()."""
        logger.debug("Cancelling in-progress turn")
        self.cancellation_tx.send(True)
        self.event_stream.send_canceled()
        return self.task


# ╔══════════════════════════════════════════════════════════╗
# ║  10. Thread — the main class, mirrors thread.rs Thread   ║
# ╚══════════════════════════════════════════════════════════╝

class Thread:
    """Faithful port of thread.rs Thread struct.

    This is the core state machine. It manages:
    - messages: the conversation history
    - pending_message: the in-progress agent message being streamed
    - tools: registered tools (BTreeMap<SharedString, Arc<dyn AnyAgentTool>>)
    - running_turn: the current RunningTurn (if any)
    - model: the active LanguageModel

    The agentic loop lives in run_turn_internal().
    """

    def __init__(
        self,
        model: Optional[LanguageModel] = None,
        system_prompt: str = "",
        working_directory: str = ".",
    ):
        self.id = str(uuid.uuid4())
        self.prompt_id = str(uuid.uuid4())
        self.title: Optional[str] = None
        self.messages: list[Message] = []
        self.model = model
        self.system_prompt = system_prompt
        self.working_directory = working_directory

        # Mirrors thread.rs fields
        self.running_turn: Optional[RunningTurn] = None
        self.has_queued_message: bool = False
        self.pending_message: Optional[AgentMessage] = None
        self.tools: dict[str, AnyAgentTool] = {}
        self.request_token_usage: dict[str, TokenUsage] = {}
        self.cumulative_token_usage = TokenUsage()
        self.thinking_enabled: bool = False
        self.thinking_effort: Optional[str] = None
        self.file_read_times: dict[str, float] = {}

    # ── Tool registration (mirrors add_tool / remove_tool) ──

    def add_tool(self, tool: AgentTool) -> None:
        """Register a tool. Mirrors Thread::add_tool()."""
        wrapped = AnyAgentTool(tool)
        assert wrapped.name not in self.tools, f"Duplicate tool name: {wrapped.name}"
        self.tools[wrapped.name] = wrapped

    def remove_tool(self, name: str) -> bool:
        return self.tools.pop(name, None) is not None

    # ── Model management ────────────────────────────────────

    def set_model(self, model: LanguageModel) -> None:
        self.model = model

    # ── Sending messages (mirrors Thread::send / resume) ────

    def send(
        self,
        content: str,
        user_message_id: Optional[str] = None,
    ) -> asyncio.Queue[ThreadEvent | Exception | None]:
        """Send a user message and start the agentic loop.

        Returns the event queue that the UI should consume.
        Mirrors Thread::send().
        """
        msg_id = user_message_id or str(uuid.uuid4())
        user_msg = UserMessage(
            id=msg_id,
            content=[UserMessageText(text=content)],
        )
        self.messages.append(Message.user(user_msg))
        self._advance_prompt_id()
        return self._run_turn()

    def resume(self) -> asyncio.Queue[ThreadEvent | Exception | None]:
        """Resume after tool calls. Mirrors Thread::resume()."""
        self.messages.append(Message.resume())
        return self._run_turn()

    # ── Cancel ──────────────────────────────────────────────

    def cancel(self) -> Optional[asyncio.Task]:
        """Cancel the running turn. Mirrors Thread::cancel()."""
        if self.running_turn is None:
            self._flush_pending_message()
            return None
        turn = self.running_turn
        self.running_turn = None
        task = turn.cancel()
        self._flush_pending_message()
        return task

    # ── Accessors ───────────────────────────────────────────

    def is_empty(self) -> bool:
        return not self.messages and self.title is None

    def last_message(self) -> Optional[Message]:
        if self.pending_message is not None:
            return Message.agent(self.pending_message)
        return self.messages[-1] if self.messages else None

    def is_turn_complete(self) -> bool:
        return self.running_turn is None

    # ── Internal: the agentic loop ──────────────────────────

    def _run_turn(self) -> asyncio.Queue[ThreadEvent | Exception | None]:
        """Start a new turn. Mirrors Thread::run_turn().

        Creates the event queue, spawns the async turn task, stores RunningTurn.
        Returns the queue for consumers (UI).
        """
        self._flush_pending_message()
        if self.running_turn is not None:
            self.running_turn.cancel()
            self.running_turn = None

        assert self.model is not None, "No language model configured"

        events_queue: asyncio.Queue[ThreadEvent | Exception | None] = asyncio.Queue()
        event_stream = ThreadEventStream(events_queue)
        cancellation_tx = WatchChannel(False)
        tools = dict(self.tools)  # snapshot of tools for this turn

        task = asyncio.create_task(
            self._run_turn_internal(event_stream, cancellation_tx, tools)
        )

        self.running_turn = RunningTurn(
            task=task,
            event_stream=event_stream,
            tools=tools,
            cancellation_tx=cancellation_tx,
        )

        return events_queue

    async def _run_turn_internal(
        self,
        event_stream: ThreadEventStream,
        cancellation: WatchChannel,
        tools: dict[str, AnyAgentTool],
    ) -> None:
        """The core agentic loop. Faithful port of Thread::run_turn_internal().

        Loop:
          1. Build completion request (system prompt + messages + tools)
          2. Stream completion from model
          3. Handle events: text→pending_message, tool_use→spawn tool task
          4. Wait for ALL tool tasks to complete (parallel, like FuturesUnordered)
          5. Flush pending message
          6. If error → retry with strategy
          7. If no tool results (end_turn) → break
          8. If tool results → continue loop (model processes results)
        """
        assert self.model is not None
        model = self.model
        attempt = 0

        try:
            while True:
                # 1. Build request
                request = self._build_completion_request(tools)

                logger.debug("Calling model.stream_completion, attempt %d", attempt)

                # 2. Stream completion
                tool_result_tasks: list[asyncio.Task[LanguageModelToolResult]] = []
                error: Optional[Exception] = None

                try:
                    stream = model.stream_completion(request)
                    async for completion_event in stream:
                        if cancellation.borrow():
                            break

                        try:
                            maybe_task = self._handle_completion_event(
                                completion_event, event_stream, cancellation, tools,
                            )
                            if maybe_task is not None:
                                tool_result_tasks.append(maybe_task)
                        except CompletionError as e:
                            error = e
                            break
                except LanguageModelCompletionError as e:
                    error = e
                except Exception as e:
                    error = e

                if cancellation.borrow():
                    logger.debug("Turn cancelled by user, exiting")
                    return

                # 4. Wait for ALL tool tasks (parallel — like FuturesUnordered)
                end_turn = len(tool_result_tasks) == 0
                if tool_result_tasks:
                    results = await asyncio.gather(
                        *tool_result_tasks, return_exceptions=True
                    )
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error("Tool task raised: %s", result)
                            continue
                        assert isinstance(result, LanguageModelToolResult)
                        logger.debug("Tool finished: %s", result.tool_name)
                        # Update tool call status in the event stream
                        event_stream.update_tool_call_fields(
                            result.tool_use_id,
                            ToolCallUpdateFields(
                                status=(
                                    ToolCallStatus.Failed if result.is_error
                                    else ToolCallStatus.Completed
                                ),
                                raw_output=result.output,
                            ),
                        )
                        # Store result in pending message
                        pm = self._get_pending_message()
                        pm.tool_results[result.tool_use_id] = result

                # 5. Flush pending message
                self._flush_pending_message()

                # 6. Handle errors with retry
                if error is not None:
                    if isinstance(error, RefusalError):
                        event_stream.send_stop("refusal")
                        return
                    if isinstance(error, MaxTokensError):
                        event_stream.send_stop("max_tokens")
                        return

                    attempt += 1
                    strategy = retry_strategy_for(error)
                    if strategy is None:
                        event_stream.send_error(error)
                        return

                    max_a = (
                        strategy.max_attempts
                        if isinstance(strategy, FixedDelay)
                        else strategy.max_attempts
                    )
                    if attempt > max_a:
                        event_stream.send_error(error)
                        return

                    if isinstance(strategy, ExponentialBackoff):
                        delay = strategy.initial_delay * (2 ** (attempt - 1))
                    else:
                        delay = strategy.delay

                    logger.debug("Retry attempt %d with delay %.1fs", attempt, delay)
                    event_stream.send_retry(str(error), attempt, max_a, delay)
                    await asyncio.sleep(delay)

                    # If the last message is an agent message with no tool results,
                    # add a Resume to re-prompt
                    if self.messages and self.messages[-1].kind == MessageKind.Agent:
                        am = self.messages[-1].agent_message
                        if am and not am.tool_results:
                            self.messages.append(Message.resume())
                    continue

                # 7. If end_turn (no tool calls) → done
                if end_turn:
                    event_stream.send_stop("end_turn")
                    return

                # 8. Check for queued message (Zed's has_queued_message)
                if self.has_queued_message:
                    logger.debug("Queued message found, ending turn")
                    event_stream.send_stop("end_turn")
                    return

                # Reset attempt counter after successful tool execution
                attempt = 0

        except Exception as e:
            logger.error("Turn execution failed: %s", e, exc_info=True)
            event_stream.send_error(e)
        finally:
            self.running_turn = None
            # Signal end of stream
            event_stream._queue.put_nowait(None)

    # ── Handle individual completion events ─────────────────

    def _handle_completion_event(
        self,
        event: LanguageModelCompletionEvent,
        event_stream: ThreadEventStream,
        cancellation: WatchChannel,
        tools: dict[str, AnyAgentTool],
    ) -> Optional[asyncio.Task[LanguageModelToolResult]]:
        """Handle a single streamed completion event.

        Mirrors Thread::handle_completion_event(). Returns an optional
        tool result task (the tool execution future).
        """
        if isinstance(event, LMStartMessageEvent):
            self._flush_pending_message()
            self.pending_message = AgentMessage()
            return None

        if isinstance(event, LMTextEvent):
            event_stream.send_text(event.text)
            pm = self._get_pending_message()
            if pm.content and isinstance(pm.content[-1], AMCText):
                pm.content[-1].text += event.text
            else:
                pm.content.append(AMCText(text=event.text))
            return None

        if isinstance(event, LMThinkingEvent):
            event_stream.send_thinking(event.text)
            pm = self._get_pending_message()
            if pm.content and isinstance(pm.content[-1], AMCThinking):
                pm.content[-1].text += event.text
                if event.signature:
                    pm.content[-1].signature = event.signature
            else:
                pm.content.append(AMCThinking(text=event.text, signature=event.signature))
            return None

        if isinstance(event, LMRedactedThinkingEvent):
            pm = self._get_pending_message()
            pm.content.append(AMCRedactedThinking(data=event.data))
            return None

        if isinstance(event, LMReasoningDetailsEvent):
            pm = self._get_pending_message()
            pm.reasoning_details = event.details
            return None

        if isinstance(event, LMToolUseEvent):
            return self._handle_tool_use_event(
                event.tool_use, event_stream, cancellation, tools,
            )

        if isinstance(event, LMToolUseJsonParseErrorEvent):
            result = LanguageModelToolResult(
                tool_use_id=event.id,
                tool_name=event.tool_name,
                is_error=True,
                content=LanguageModelToolResultContent(
                    text=f"Error parsing input JSON: {event.json_parse_error}"
                ),
                output=event.raw_input,
            )
            return asyncio.create_task(_ready(result))

        if isinstance(event, LMUsageUpdateEvent):
            self._update_token_usage(event.usage)
            return None

        if isinstance(event, LMStopEvent):
            if event.reason == StopReason.Refusal:
                raise RefusalError()
            if event.reason == StopReason.MaxTokens:
                raise MaxTokensError()
            # ToolUse and EndTurn are handled by the loop
            return None

        # QueuedEvent, StartedEvent — no-ops
        return None

    def _handle_tool_use_event(
        self,
        tool_use: LanguageModelToolUse,
        event_stream: ThreadEventStream,
        cancellation: WatchChannel,
        tools: dict[str, AnyAgentTool],
    ) -> Optional[asyncio.Task[LanguageModelToolResult]]:
        """Handle a tool use event. Mirrors Thread::handle_tool_use_event().

        Adds the tool use to pending message, sends the tool call event,
        and if input is complete, spawns the tool execution task.
        """
        tool = tools.get(tool_use.name)
        title = tool_use.name
        kind = ToolKind.Other
        if tool is not None:
            title = tool.initial_title(tool_use.input)
            kind = tool.kind()

        # Add to pending message
        pm = self._get_pending_message()
        # Check if last content is same tool_use (streaming input)
        push_new = True
        if pm.content and isinstance(pm.content[-1], AMCToolUse):
            if pm.content[-1].tool_use.id == tool_use.id:
                pm.content[-1].tool_use = tool_use
                push_new = False

        if push_new:
            event_stream.send_tool_call(tool_use.id, tool_use.name, title, kind, tool_use.input)
            pm.content.append(AMCToolUse(tool_use=tool_use))
        else:
            event_stream.update_tool_call_fields(
                tool_use.id,
                ToolCallUpdateFields(title=title, kind=kind, raw_input=tool_use.input),
            )

        if not tool_use.is_input_complete:
            return None  # Still streaming input

        if tool is None:
            content = f"No tool named {tool_use.name} exists"
            result = LanguageModelToolResult(
                tool_use_id=tool_use.id,
                tool_name=tool_use.name,
                is_error=True,
                content=LanguageModelToolResultContent(text=content),
            )
            return asyncio.create_task(_ready(result))

        # Create per-tool event stream
        tool_event_stream = ToolCallEventStream(
            tool_use_id=tool_use.id,
            stream=event_stream,
            cancellation=cancellation,
        )
        tool_event_stream.update_fields(
            ToolCallUpdateFields(status=ToolCallStatus.InProgress)
        )

        # Spawn tool execution (like cx.foreground_executor().spawn())
        return asyncio.create_task(tool.run(tool_use.input, tool_event_stream))

    # ── Pending message management ──────────────────────────

    def _get_pending_message(self) -> AgentMessage:
        """Mirrors Thread::pending_message() — get or create."""
        if self.pending_message is None:
            self.pending_message = AgentMessage()
        return self.pending_message

    def _flush_pending_message(self) -> None:
        """Mirrors Thread::flush_pending_message().

        Moves pending_message into self.messages. Any tool_use blocks
        without results get a "Tool canceled" result.
        """
        if self.pending_message is None:
            return
        message = self.pending_message
        self.pending_message = None

        if not message.content:
            return

        # Fill in missing tool results with cancellation message
        for item in message.content:
            if isinstance(item, AMCToolUse):
                if item.tool_use.id not in message.tool_results:
                    message.tool_results[item.tool_use.id] = LanguageModelToolResult(
                        tool_use_id=item.tool_use.id,
                        tool_name=item.tool_use.name,
                        is_error=True,
                        content=LanguageModelToolResultContent(text=TOOL_CANCELED_MESSAGE),
                    )

        self.messages.append(Message.agent(message))

    # ── Build completion request ────────────────────────────

    def _build_completion_request(
        self,
        tools: dict[str, AnyAgentTool],
    ) -> LanguageModelRequest:
        """Mirrors Thread::build_completion_request()."""
        # System prompt message
        lm_tools = [
            LanguageModelRequestTool(
                name=name,
                description=tool.description(),
                input_schema=tool.input_schema(),
            )
            for name, tool in tools.items()
        ]

        messages: list[LanguageModelRequestMessage] = []
        if self.system_prompt:
            messages.append(LanguageModelRequestMessage(
                role=Role.System,
                content=[TextContent(text=self.system_prompt)],
            ))

        for message in self.messages:
            messages.extend(message.to_request())

        # Mark last message for caching
        if messages:
            messages[-1].cache = True

        # Include pending message content if any
        if self.pending_message is not None:
            messages.extend(self.pending_message.to_request())

        return LanguageModelRequest(
            thread_id=self.id,
            prompt_id=self.prompt_id,
            messages=messages,
            tools=lm_tools,
            temperature=None,
            thinking_allowed=self.thinking_enabled,
            thinking_effort=self.thinking_effort,
        )

    # ── Token usage tracking ────────────────────────────────

    def _update_token_usage(self, usage: TokenUsage) -> None:
        """Mirrors Thread::update_token_usage()."""
        last_user = self._last_user_message()
        if last_user is not None:
            self.request_token_usage[last_user.id] = usage

    def _last_user_message(self) -> Optional[UserMessage]:
        for msg in reversed(self.messages):
            if msg.kind == MessageKind.User and msg.user_message:
                return msg.user_message
        return None

    # ── Helpers ─────────────────────────────────────────────

    def _advance_prompt_id(self) -> None:
        self.prompt_id = str(uuid.uuid4())

    def to_markdown(self) -> str:
        """Mirrors Thread::to_markdown()."""
        parts: list[str] = []
        for i, msg in enumerate(self.messages):
            if i > 0:
                parts.append("\n")
            if msg.kind == MessageKind.User:
                parts.append("## User\n\n")
            elif msg.kind == MessageKind.Agent:
                parts.append("## Assistant\n\n")
            parts.append(msg.to_markdown())
        if self.pending_message is not None:
            parts.append("\n## Assistant\n\n")
            parts.append(self.pending_message.to_markdown())
        return "".join(parts)


async def _ready(value: LanguageModelToolResult) -> LanguageModelToolResult:
    """Helper to create a completed task (like Task::ready)."""
    return value
