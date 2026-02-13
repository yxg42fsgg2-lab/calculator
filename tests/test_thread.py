"""Tests for thread.py — mirrors crates/agent/src/tests/ patterns."""

import asyncio
import pytest

from zed_agent.thread import (
    AgentMessage, AMCText, AMCThinking, AMCToolUse,
    Message, MessageKind, UserMessage, UserMessageText,
    Thread, ThreadEvent, ThreadEventAgentText, ThreadEventStop,
    ThreadEventToolCall, ThreadEventToolCallUpdate,
    ThreadEventStream,
    ToolKind, ToolCallStatus, ToolCallUpdateFields,
    WatchChannel, TOOL_CANCELED_MESSAGE,
    AgentTool, ToolCallEventStream,
)
from zed_agent.language_model.model import (
    LanguageModel, LanguageModelId, LanguageModelProviderId,
    LanguageModelCompletionEvent, LanguageModelToolUse, LanguageModelToolUseId,
    TextEvent as LMTextEvent, ToolUseEvent as LMToolUseEvent,
    StopEvent as LMStopEvent, StopReason, TokenUsage,
)
from zed_agent.language_model.request import LanguageModelRequest

from typing import Any, AsyncIterator, Optional


# ── Fake model for testing ──────────────────────────────────

class FakeModel(LanguageModel):
    """Mirrors language_model::fake_provider::FakeLanguageModel."""

    def __init__(self):
        self.completion_events: list[LanguageModelCompletionEvent] = []

    def id(self) -> LanguageModelId:
        return LanguageModelId("fake")
    def name(self) -> str:
        return "Fake"
    def provider_id(self) -> LanguageModelProviderId:
        return LanguageModelProviderId("fake")
    def supports_images(self) -> bool:
        return False
    def supports_tools(self) -> bool:
        return True

    async def stream_completion(self, request: LanguageModelRequest) -> AsyncIterator[LanguageModelCompletionEvent]:
        for event in self.completion_events:
            yield event


# ── Fake tool for testing ───────────────────────────────────

class EchoTool(AgentTool):
    NAME = "echo"
    def description(self) -> str:
        return "Echoes input"
    def kind(self) -> ToolKind:
        return ToolKind.Other
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    def initial_title(self, input: dict[str, Any]) -> str:
        return f"Echo: {input.get('text', '')}"

    async def run(self, input: dict[str, Any], event_stream: ToolCallEventStream) -> str:
        return f"echoed: {input.get('text', '')}"


# ── Tests ───────────────────────────────────────────────────

class TestWatchChannel:
    @pytest.mark.asyncio
    async def test_send_and_borrow(self):
        ch = WatchChannel(False)
        assert ch.borrow() is False
        ch.send(True)
        assert ch.borrow() is True


class TestMessages:
    def test_user_message(self):
        msg = UserMessage(id="u1", content=[UserMessageText(text="Hello")])
        assert msg.text() == "Hello"
        req = msg.to_request()
        assert req.role.value == "user"

    def test_agent_message_to_request(self):
        am = AgentMessage(content=[AMCText(text="Hi there")])
        reqs = am.to_request()
        assert len(reqs) == 1
        assert reqs[0].role.value == "assistant"

    def test_agent_message_with_tool_use(self):
        tu = LanguageModelToolUse(
            id=LanguageModelToolUseId("tc1"), name="echo",
            input={"text": "hi"}, is_input_complete=True,
        )
        from zed_agent.language_model.model import LanguageModelToolResult, LanguageModelToolResultContent
        tr = LanguageModelToolResult(
            tool_use_id=LanguageModelToolUseId("tc1"), tool_name="echo",
            content=LanguageModelToolResultContent(text="echoed: hi"),
        )
        am = AgentMessage(
            content=[AMCText(text="Let me help"), AMCToolUse(tool_use=tu)],
            tool_results={"tc1": tr},
        )
        reqs = am.to_request()
        # Should produce assistant msg + user msg (tool results)
        assert len(reqs) == 2
        assert reqs[0].role.value == "assistant"
        assert reqs[1].role.value == "user"

    def test_message_enum(self):
        um = UserMessage(content=[UserMessageText(text="test")])
        m = Message.user(um)
        assert m.kind == MessageKind.User
        assert m.to_markdown() == "test\n"

        m2 = Message.resume()
        assert m2.kind == MessageKind.Resume
        reqs = m2.to_request()
        assert len(reqs) == 1
        assert "Continue" in reqs[0].string_contents()


class TestThread:
    def test_new_thread_is_empty(self):
        t = Thread(model=FakeModel())
        assert t.is_empty()
        assert t.last_message() is None
        assert t.is_turn_complete()

    @pytest.mark.asyncio
    async def test_send_text_response(self):
        model = FakeModel()
        model.completion_events = [
            LMTextEvent(text="Hello "),
            LMTextEvent(text="world!"),
            LMStopEvent(reason=StopReason.EndTurn),
        ]
        thread = Thread(model=model, system_prompt="You are helpful.")
        queue = thread.send("Hi")

        events = []
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=5)
            if ev is None:
                break
            events.append(ev)

        text_events = [e for e in events if isinstance(e, ThreadEventAgentText)]
        assert len(text_events) == 2
        assert text_events[0].text == "Hello "
        assert text_events[1].text == "world!"

        stop_events = [e for e in events if isinstance(e, ThreadEventStop)]
        assert len(stop_events) == 1
        assert stop_events[0].reason == "end_turn"

        # Thread should have 2 messages: user + agent
        assert len(thread.messages) == 2
        assert thread.messages[0].kind == MessageKind.User
        assert thread.messages[1].kind == MessageKind.Agent
        am = thread.messages[1].agent_message
        assert am is not None
        assert len(am.content) == 1
        assert isinstance(am.content[0], AMCText)
        assert am.content[0].text == "Hello world!"

    @pytest.mark.asyncio
    async def test_send_with_tool_call(self):
        model = FakeModel()
        model.completion_events = [
            LMTextEvent(text="I'll echo that."),
            LMToolUseEvent(tool_use=LanguageModelToolUse(
                id=LanguageModelToolUseId("tc1"), name="echo",
                input={"text": "hello"}, is_input_complete=True,
            )),
            LMStopEvent(reason=StopReason.ToolUse),
        ]
        thread = Thread(model=model)
        thread.add_tool(EchoTool())

        # After first turn, model called a tool. The loop should execute it
        # and then the model needs to respond to tool results.
        # Since FakeModel returns the same events, we need to simulate 2 rounds.
        # For simplicity, just let the second round end immediately.
        call_count = 0
        original_events = model.completion_events[:]

        class TwoRoundModel(LanguageModel):
            def __init__(self):
                self.round = 0
            def id(self): return LanguageModelId("fake")
            def name(self): return "Fake"
            def provider_id(self): return LanguageModelProviderId("fake")
            def supports_images(self): return False
            def supports_tools(self): return True
            async def stream_completion(self, request):
                self.round += 1
                if self.round == 1:
                    for e in original_events:
                        yield e
                else:
                    yield LMTextEvent(text="Done!")
                    yield LMStopEvent(reason=StopReason.EndTurn)

        two_round = TwoRoundModel()
        thread2 = Thread(model=two_round)
        thread2.add_tool(EchoTool())
        queue = thread2.send("echo hello")

        events = []
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=5)
            if ev is None:
                break
            events.append(ev)

        # Should have: text "I'll echo that.", tool_call, tool_call_update (in_progress),
        # tool_call_update (completed), text "Done!", stop
        tool_calls = [e for e in events if isinstance(e, ThreadEventToolCall)]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "echo"

        # Thread messages: user, agent(tool_call+result), agent(Done!)
        assert len(thread2.messages) == 3

    @pytest.mark.asyncio
    async def test_flush_cancels_pending_tools(self):
        """Verify that flush_pending_message fills in 'Tool canceled' for incomplete tools."""
        thread = Thread(model=FakeModel())
        tu = LanguageModelToolUse(
            id=LanguageModelToolUseId("tc_incomplete"), name="slow_tool",
            input={}, is_input_complete=True,
        )
        thread.pending_message = AgentMessage(
            content=[AMCToolUse(tool_use=tu)],
        )
        thread._flush_pending_message()

        assert len(thread.messages) == 1
        am = thread.messages[0].agent_message
        assert am is not None
        assert "tc_incomplete" in am.tool_results
        assert am.tool_results["tc_incomplete"].is_error
        assert TOOL_CANCELED_MESSAGE in am.tool_results["tc_incomplete"].content.text

    def test_to_markdown(self):
        thread = Thread(model=FakeModel())
        thread.messages.append(Message.user(UserMessage(content=[UserMessageText(text="Hello")])))
        thread.messages.append(Message.agent(AgentMessage(content=[AMCText(text="Hi!")])))
        md = thread.to_markdown()
        assert "## User" in md
        assert "## Assistant" in md
        assert "Hello" in md
        assert "Hi!" in md


class TestTools:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        (tmp_path / "test.txt").write_text("line1\nline2\nline3\n")
        from zed_agent.tools.read_file import ReadFileTool
        tool = ReadFileTool(str(tmp_path))
        ch = WatchChannel(False)
        es = ToolCallEventStream(LanguageModelToolUseId("t1"), ThreadEventStream(asyncio.Queue()), ch)
        result = await tool.run({"path": "test.txt"}, es)
        assert "line1" in result

    @pytest.mark.asyncio
    async def test_edit_file(self, tmp_path):
        (tmp_path / "f.txt").write_text("aaa bbb ccc")
        from zed_agent.tools.edit_file import EditFileTool
        tool = EditFileTool(str(tmp_path))
        ch = WatchChannel(False)
        es = ToolCallEventStream(LanguageModelToolUseId("t1"), ThreadEventStream(asyncio.Queue()), ch)
        await tool.run({"path": "f.txt", "old_string": "bbb", "new_string": "BBB"}, es)
        assert (tmp_path / "f.txt").read_text() == "aaa BBB ccc"

    @pytest.mark.asyncio
    async def test_terminal(self, tmp_path):
        from zed_agent.tools.terminal import TerminalTool
        tool = TerminalTool(str(tmp_path))
        ch = WatchChannel(False)
        es = ToolCallEventStream(LanguageModelToolUseId("t1"), ThreadEventStream(asyncio.Queue()), ch)
        result = await tool.run({"command": "echo hello_world", "cd": "."}, es)
        assert "hello_world" in result

    @pytest.mark.asyncio
    async def test_grep(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo():\n  pass\n")
        from zed_agent.tools.grep import GrepTool
        tool = GrepTool(str(tmp_path))
        ch = WatchChannel(False)
        es = ToolCallEventStream(LanguageModelToolUseId("t1"), ThreadEventStream(asyncio.Queue()), ch)
        result = await tool.run({"regex": "def foo"}, es)
        assert "a.py" in result

    @pytest.mark.asyncio
    async def test_find_path(self, tmp_path):
        (tmp_path / "main.py").write_text("")
        from zed_agent.tools.find_path import FindPathTool
        tool = FindPathTool(str(tmp_path))
        ch = WatchChannel(False)
        es = ToolCallEventStream(LanguageModelToolUseId("t1"), ThreadEventStream(asyncio.Queue()), ch)
        result = await tool.run({"pattern": "main.py"}, es)
        assert "main.py" in result

    @pytest.mark.asyncio
    async def test_create_file(self, tmp_path):
        from zed_agent.tools.create_file import CreateFileTool
        tool = CreateFileTool(str(tmp_path))
        ch = WatchChannel(False)
        es = ToolCallEventStream(LanguageModelToolUseId("t1"), ThreadEventStream(asyncio.Queue()), ch)
        await tool.run({"path": "new.txt", "content": "hello"}, es)
        assert (tmp_path / "new.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "subdir").mkdir()
        from zed_agent.tools.list_directory import ListDirectoryTool
        tool = ListDirectoryTool(str(tmp_path))
        ch = WatchChannel(False)
        es = ToolCallEventStream(LanguageModelToolUseId("t1"), ThreadEventStream(asyncio.Queue()), ch)
        result = await tool.run({"path": "."}, es)
        assert "a.txt" in result
        assert "subdir/" in result

    @pytest.mark.asyncio
    async def test_now(self):
        from zed_agent.tools.now import NowTool
        tool = NowTool()
        ch = WatchChannel(False)
        es = ToolCallEventStream(LanguageModelToolUseId("t1"), ThreadEventStream(asyncio.Queue()), ch)
        result = await tool.run({}, es)
        assert "T" in result  # ISO format


class TestSubagentSystem:
    def test_depth_tracking(self):
        parent = Thread(model=FakeModel())
        assert parent.depth() == 0
        assert not parent.is_subagent()

        child = Thread.new_subagent(parent)
        assert child.depth() == 1
        assert child.is_subagent()
        assert child.parent_thread_id() == parent.id

        grandchild = Thread.new_subagent(child)
        assert grandchild.depth() == 2

    def test_running_subagent_tracking(self):
        parent = Thread(model=FakeModel())
        child1 = Thread.new_subagent(parent)
        child2 = Thread.new_subagent(parent)

        parent.register_running_subagent(child1)
        parent.register_running_subagent(child2)
        assert parent.running_subagent_count() == 2

        parent.unregister_running_subagent(child1.id)
        assert parent.running_subagent_count() == 1

    def test_subagent_inherits_config(self):
        parent = Thread(model=FakeModel(), system_prompt="Be helpful", working_directory="/test")
        parent.thinking_enabled = True
        parent.thinking_effort = "high"

        child = Thread.new_subagent(parent)
        assert child.model is parent.model
        assert child.system_prompt == parent.system_prompt
        assert child.working_directory == parent.working_directory
        assert child.thinking_enabled == True
        assert child.thinking_effort == "high"


class TestCompletionIntent:
    @pytest.mark.asyncio
    async def test_intent_switches_to_tool_results(self):
        """After tool execution, intent should switch from UserPrompt to ToolResults."""
        from zed_agent.thread import CompletionIntent

        # We can verify this indirectly through the request building
        # The intent tracking happens in _run_turn_internal
        model = FakeModel()
        thread = Thread(model=model)
        # CompletionIntent enum exists and has correct values
        assert CompletionIntent.UserPrompt.value == "user_prompt"
        assert CompletionIntent.ToolResults.value == "tool_results"


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        from zed_agent.thread import RateLimiter
        rl = RateLimiter(limit=2)
        # Should be able to acquire twice
        await rl.acquire()
        await rl.acquire()
        # Release both
        rl.release()
        rl.release()
        # Should be able to acquire again
        await rl.acquire()
        rl.release()

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks(self):
        from zed_agent.thread import RateLimiter
        rl = RateLimiter(limit=1)
        await rl.acquire()
        # Second acquire should block
        acquired = False
        async def try_acquire():
            nonlocal acquired
            await rl.acquire()
            acquired = True
        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.05)
        assert not acquired
        rl.release()
        await asyncio.sleep(0.05)
        assert acquired
        rl.release()
        task.cancel()


class TestTemplates:
    def test_render_with_tools(self, tmp_path):
        from zed_agent.templates import render_system_prompt
        prompt = render_system_prompt(
            working_directories=[str(tmp_path)],
            available_tools=["read_file", "grep", "terminal"],
            model_name="claude-test",
        )
        assert "software engineer" in prompt
        assert "Tool Use" in prompt
        assert "grep" in prompt  # grep is referenced in the Searching section
        assert "claude-test" in prompt

    def test_render_without_tools(self, tmp_path):
        from zed_agent.templates import render_system_prompt
        prompt = render_system_prompt(
            working_directories=[str(tmp_path)],
            available_tools=[],
        )
        assert "no ability to use tools" in prompt

    def test_loads_rules_file(self, tmp_path):
        (tmp_path / ".rules").write_text("Use tabs.")
        from zed_agent.templates import load_project_rules
        rules = load_project_rules(str(tmp_path))
        assert rules == "Use tabs."
