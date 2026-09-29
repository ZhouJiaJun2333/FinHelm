"""运行状态（从事件折叠出来）和流式输出。"""

from __future__ import annotations

import io
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace

import anthropic
import pytest
from openai.types.chat import ChatCompletionChunk

from data_agent.cli import make_console_sink
from data_agent.core.agent import AwaitingUser
from data_agent.core.events import (
    LLMResponded,
    StepStarted,
    TextDelta,
    TurnEnded,
    TurnStarted,
)
from data_agent.core.messages import LLMResponse, Message, ToolCall, Usage
from data_agent.core.state import AgentState, fold
from data_agent.db.connection import QueryResult
from data_agent.llm.anthropic_provider import AnthropicProvider
from data_agent.llm.openai_provider import OpenAICompatibleProvider
from data_agent.tools.ask_user import AskUserTool
from data_agent.tools.sql.results import ResultStore

from fakes import EchoTool, ScriptedProvider, make_agent

FINAL = LLMResponse(text="华东最高，810 万", stop_reason="end_turn", usage=Usage(input=200, output=20))


def call(n: int, name: str = "echo", **args) -> LLMResponse:
    return LLMResponse(text=f"先查第 {n} 次", tool_calls=[ToolCall(f"c{n}", name, args or {"text": str(n)})],
                       stop_reason="tool_use", usage=Usage(input=100, output=10))


class CharStreamProvider(ScriptedProvider):
    """一个字一个字地流：先把思考吐完，再吐正文。"""

    def stream(self, messages, tools=None, system=None, max_tokens=None, *, on_delta):
        response = self.chat(messages, tools=tools, system=system)
        for ch in "想一想":
            on_delta(ch, True)
        for ch in response.text:
            on_delta(ch, False)
        return response


# ================================================================ 状态
def test_状态就是事件折叠出来的_和agent身上的一样():
    agent, events = make_agent([call(1), call(2), FINAL])
    agent.run("哪个大区最高")

    assert fold(events) == agent.state
    s = agent.state
    assert (s.status, s.question, s.answer, s.error, s.step) == ("idle", "哪个大区最高", "华东最高，810 万", "", 3)
    assert [(t.call_id, t.status) for t in s.tools] == [("c1", "done"), ("c2", "done")]
    assert s.pending_tools == () and not s.running
    assert s.turn_usage == Usage(input=400, output=40)
    assert s.context_tokens == 220


def test_订阅者收到事件时状态已经更新():
    seen = []
    agent, _ = make_agent([call(1), FINAL])
    agent.on_event = lambda e: seen.append((type(e).__name__, agent.state.status, len(agent.state.pending_tools)))
    agent.run("q")

    assert seen[0] == ("TurnStarted", "thinking", 0)
    assert ("ToolStarted", "tools", 1) in seen
    assert ("ToolFinished", "tools", 0) in seen
    assert seen[-1] == ("TurnEnded", "idle", 0)


def test_新的一轮清掉上一轮的工具_上下文用量留着():
    agent, _ = make_agent([call(1), FINAL, FINAL])
    agent.run("第一轮")
    agent.run("第二轮")
    assert agent.state.tools == () and agent.state.question == "第二轮"
    assert agent.state.context_tokens == 220


def test_出错_状态记下原因_没有挂着的工具():
    agent, events = make_agent([call(1), RuntimeError("限流")])
    with pytest.raises(RuntimeError):
        agent.run("q")

    assert isinstance(events[-1], TurnEnded) and events[-1].interrupted == "RuntimeError: 限流"
    assert (agent.state.status, agent.state.error, agent.state.answer) == ("idle", "RuntimeError: 限流", "")


class Boom(EchoTool):
    name = "boom"
    description = "执行到一半被 Ctrl-C"

    def run(self, args) -> str:
        raise KeyboardInterrupt


def test_工具执行到一半断了_记成出错_不会一直挂着():
    agent, _ = make_agent([call(1, "boom")], tools=[Boom()])
    with pytest.raises(KeyboardInterrupt):
        agent.run("q")
    assert [(t.name, t.status) for t in agent.state.tools] == [("boom", "error")]
    assert agent.state.pending_tools == ()


def test_问用户_状态停在asking_回答后那次调用算完成():
    agent, _ = make_agent([call(1, "ask_user", question="门槛多少？", options=["30 万"]), FINAL],
                          tools=[EchoTool(), AskUserTool()])
    with pytest.raises(AwaitingUser):
        agent.run("有多少大客户")
    s = agent.state
    assert (s.status, s.asking.question, s.asking.call_id, s.error) == ("asking", "门槛多少？", "c1", "")
    assert [t.status for t in s.tools] == ["asking"]

    agent.resume("30 万")
    s = agent.state
    assert (s.status, s.asking, s.answer) == ("idle", None, "华东最高，810 万")
    assert [t.status for t in s.tools] == ["done"]


def test_reset_状态回到初始():
    agent, _ = make_agent([FINAL])
    agent.run("q")
    agent.reset()
    assert agent.state == AgentState()


# ================================================================ 流式
def test_流式_逐字发事件_拼起来就是回答_历史和不流式一样():
    script = [call(1), FINAL]
    agent, events = make_agent(script, stream=True)
    agent.llm = CharStreamProvider(script)
    snapshots = []
    agent.on_event = lambda e: (events.append(e), snapshots.append(agent.state.streaming_text))
    agent.run("q")

    deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(d.text for d in deltas if not d.thinking and d.step == 2) == FINAL.text
    assert "".join(d.text for d in deltas if d.thinking and d.step == 1) == "想一想"
    assert "华东最高" in snapshots, "输出到一半时状态里有半句话"
    assert agent.state.streaming_text == "" and agent.state.streaming_thinking == ""

    plain, _ = make_agent(script)
    plain.run("q")
    assert agent.context.render() == plain.context.render()


def test_流式_事件顺序_步开始_增量_模型回复():
    agent, events = make_agent([FINAL], stream=True)
    agent.run("q")
    kinds = [type(e) for e in events]
    assert kinds == [TurnStarted, StepStarted, TextDelta, LLMResponded, TurnEnded]


def test_不支持流式的provider_整段给一次_没字就不发():
    agent, events = make_agent([LLMResponse(text="", tool_calls=[ToolCall("c1", "echo", {"text": "x"})],
                                            stop_reason="tool_use"), FINAL], stream=True)
    agent.run("q")
    assert [(e.step, e.text) for e in events if isinstance(e, TextDelta)] == [(2, FINAL.text)]


def test_不开流式_没有增量事件():
    agent, events = make_agent([FINAL])
    agent.run("q")
    assert not any(isinstance(e, TextDelta) for e in events)


# ================================================================ provider 拼分块
def _chunk(delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> ChatCompletionChunk:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    return ChatCompletionChunk.model_validate({"id": "x", "object": "chat.completion.chunk", "created": 0,
                                               "model": "m", "choices": choices, "usage": usage})


DS_USAGE = {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340, "prompt_cache_hit_tokens": 128}


def test_openai流式_拼回原生message_思考和工具参数都在(monkeypatch):
    """照 DeepSeek 实际的分块（2026-09-30 抓的）：思考、正文、工具参数一段段来，用量在带 finish_reason 的最后一块。"""
    chunks = [
        _chunk({"role": "assistant", "reasoning_content": ""}),
        _chunk({"reasoning_content": "要调"}), _chunk({"reasoning_content": " echo"}),
        _chunk({"content": "好的，"}), _chunk({"content": "我来回显"}),
        _chunk({"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                                "function": {"name": "echo", "arguments": ""}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"text"'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": ': "hi"}'}}]}),
        _chunk({"content": ""}, finish="tool_calls", usage=DS_USAGE),
    ]
    provider = OpenAICompatibleProvider(api_key="test", model="m")
    sent = {}
    monkeypatch.setattr(provider.client.chat.completions, "create", lambda **kw: sent.update(kw) or iter(chunks))
    deltas = []
    r = provider.stream([Message.user("q")], on_delta=lambda t, th: deltas.append((t, th)))

    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert deltas == [("要调", True), (" echo", True), ("好的，", False), ("我来回显", False)]
    assert r.text == "好的，我来回显" and r.stop_reason == "tool_calls"
    assert r.tool_calls == [ToolCall("call_0", "echo", {"text": "hi"})]
    assert r.usage == Usage(input=172, output=40, cache_read=128)
    assert r.raw_content == {
        "role": "assistant", "content": "好的，我来回显", "reasoning_content": "要调 echo",
        "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "echo", "arguments": '{"text": "hi"}'}}],
    }


def test_openai流式_用量单独一块_不思考的模型不带reasoning_content(monkeypatch):
    chunks = [_chunk({"role": "assistant", "content": "2"}), _chunk({}, finish="stop"),
              _chunk(None, usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11})]
    provider = OpenAICompatibleProvider(api_key="test", model="m")
    monkeypatch.setattr(provider.client.chat.completions, "create", lambda **kw: iter(chunks))
    r = provider.stream([Message.user("1+1")], on_delta=lambda *a: None)
    assert r.raw_content == {"role": "assistant", "content": "2"}
    assert (r.usage, r.stop_reason) == (Usage(input=10, output=1), "stop")


def test_anthropic流式_增量交出去_最终回复和chat走同一个转换(monkeypatch):
    final = anthropic.types.Message.model_validate({
        "id": "m", "type": "message", "role": "assistant", "model": "claude", "stop_reason": "tool_use",
        "content": [{"type": "text", "text": "查一下"},
                    {"type": "tool_use", "id": "t1", "name": "echo", "input": {"text": "hi"}}],
        "usage": {"input_tokens": 10, "output_tokens": 5,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    })
    events = [SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="嗯")),
              SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="查一下")),
              SimpleNamespace(type="message_stop")]

    class Stream:
        def __iter__(self):
            return iter(events)

        def get_final_message(self):
            return final

    @contextmanager
    def stream(**kw):
        yield Stream()

    provider = AnthropicProvider(api_key="test")
    monkeypatch.setattr(provider.client.messages, "stream", stream)
    monkeypatch.setattr(provider.client.messages, "create", lambda **kw: final)
    deltas = []
    r = provider.stream([Message.user("q")], on_delta=lambda t, th: deltas.append((t, th)))
    assert deltas == [("嗯", True), ("查一下", False)]
    assert r == provider.chat([Message.user("q")])


# ================================================================ 终端怎么打
def _console(events, results=None) -> str:
    sink = make_console_sink(False, results or ResultStore())
    out = io.StringIO()
    with redirect_stdout(out):
        for e in events:
            sink(e)
    return out.getvalue()


def test_终端_流式打过的回答不再打一遍_没流式的在一轮结束时打():
    agent, events = make_agent([FINAL], stream=True)
    agent.run("q")
    assert _console(events).count(FINAL.text) == 1

    agent, events = make_agent([FINAL])
    agent.run("q")
    assert _console(events).count(FINAL.text) == 1


def test_终端_流式回答里的结果编号_说完补上表格():
    results = ResultStore()
    results.add("SELECT 1", QueryResult(["region", "gmv"], [("华东", 810)], False, 1))
    agent, events = make_agent([LLMResponse(text="见 {{r1}}", stop_reason="end_turn")], stream=True)
    agent.run("q")
    out = _console(events, results)
    assert "见 {{r1}}" in out and "| 华东 | 810 |" in out
