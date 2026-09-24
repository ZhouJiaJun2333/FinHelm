"""强制整理：API 报上下文超长时压缩再重试，以及手动 /compact。

「强制」= 不看阈值，能清的清、能压的压。阈值是给估算用的；
API 都说超了，说明估算不可信，再看阈值就没意义了。
"""

from __future__ import annotations

import pytest

from data_agent.core.context import ClearOldToolResults, CompactHistory, Context, Summarize, Summary
from data_agent.core.errors import CompactionFailed, ContextOverflow
from data_agent.core.events import ContextEdited, ContextOverflowed
from data_agent.core.messages import LLMResponse, Message, ToolCall
from data_agent.core.tokens import estimate_context

from fakes import make_agent

OK = LLMResponse(text="答", stop_reason="end_turn")
OVERFLOW = ContextOverflow("prompt is too long")


def summarizer(calls: list) -> Summarize:
    def summarize(messages, prompt=None):
        calls.append(messages)
        return Summary("摘要")
    return summarize


def lazy_compactor(calls: list) -> CompactHistory:
    """阈值高到永远不会自己触发 —— 只有强制时才压。"""
    return CompactHistory(summarizer(calls), trigger_tokens=10**9, keep_recent_tokens=1)


def measure(msgs):
    return estimate_context(msgs).tokens


# ================================================================ 超长重试
def test_超长时强制压缩再重试_成功就当什么都没发生():
    calls: list = []
    agent, events = make_agent([OK, OVERFLOW, OK], context=Context([lazy_compactor(calls)]))
    agent.run("问题1")
    assert agent.run("问题2") == "答"

    assert len(calls) == 1, "阈值没到，但 API 说超了，就强制压了一次"
    kinds = [type(e) for e in events if isinstance(e, (ContextOverflowed, ContextEdited))]
    assert kinds == [ContextOverflowed, ContextEdited]
    retry = agent.llm.seen[-1]
    assert retry[0].content.startswith(CompactHistory.SUMMARY_HEADER + "摘要")
    assert retry[0].content.endswith("问题2")


def test_只重试一次_还超就抛出去_这一轮回滚():
    calls: list = []
    agent, _ = make_agent([OK, OVERFLOW, OVERFLOW, OK], context=Context([lazy_compactor(calls)]))
    agent.run("问题1")
    before = agent.context.history

    with pytest.raises(ContextOverflow):
        agent.run("问题2")
    assert agent.llm.calls == 3, "第一次 + 重试一次，不会无限重试"
    assert agent.context.history == before, "连压缩标记一起回滚"


def test_整理不出东西就不重试():
    """只有当前这一轮，没有可压的 —— 重试也还是超，不如直接报错，省一次调用。"""
    agent, events = make_agent([OVERFLOW, OK], context=Context([lazy_compactor([])]))
    with pytest.raises(ContextOverflow):
        agent.run("问题1")
    assert agent.llm.calls == 1
    assert any(isinstance(e, ContextOverflowed) for e in events)


def test_别的错误不触发整理():
    calls: list = []
    agent, _ = make_agent([OK, RuntimeError("网络断了")], context=Context([lazy_compactor(calls)]))
    agent.run("问题1")
    with pytest.raises(RuntimeError):
        agent.run("问题2")
    assert calls == []


# ================================================================ /compact
def test_手动压缩不看阈值():
    calls: list = []
    agent, events = make_agent([OK], context=Context([lazy_compactor(calls)]))
    agent.run("问题1")
    agent.run("问题2")

    done = agent.compact()
    assert [e.description for e in done] == ["把较早的 1 轮对话压缩成了摘要（保留最近 1 轮原文）"]
    assert done[0] in events, "和自动压缩一样通过事件告诉界面"
    assert agent.context.render()[0].content.endswith("问题2")


def test_对话太短时手动压缩什么都不做():
    agent, _ = make_agent([OK], context=Context([lazy_compactor([])]))
    agent.run("问题1")
    assert agent.compact() == []


def test_手动压缩失败时上下文不变():
    def broken(messages, prompt=None):
        raise CompactionFailed("写摘要的请求返回了空内容。")

    agent, _ = make_agent([OK], context=Context([
        ClearOldToolResults(trigger_tokens=10**9, keep_recent=0),
        CompactHistory(broken, trigger_tokens=10**9, keep_recent_tokens=1),
    ]))
    agent.context.add(Message.user("问题0"))
    agent.context.add(Message(role="assistant", tool_calls=[ToolCall("c0", "echo", {})]))
    agent.context.add(Message.tool_result("c0", "结果"))
    agent.context.add(Message.assistant("答0"))
    agent.run("问题1")
    before = agent.context.history

    with pytest.raises(CompactionFailed):
        agent.compact()
    assert agent.context.history == before, "前面清理工具结果的标记也一起撤掉"


# ================================================================ 强制清理
def test_强制清理不看阈值_也不看最少要省多少():
    """已经超长了，能省一点是一点；平时为了缓存攒一批才清的规矩这时不适用。"""
    clear = ClearOldToolResults(trigger_tokens=10**9, keep_recent=0, clear_at_least=10**9)
    ctx = Context([clear])
    ctx.add(Message.user("q"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("c1", "run_sql", {})]))
    ctx.add(Message.tool_result("c1", "| 华东 | 8100531.47 |\n" * 20))   # 比占位长，但远不到 clear_at_least

    assert ctx.maintain(measure) == []
    [event] = ctx.maintain(measure, force=True)
    assert event.description == "清理了 1 条较早的工具结果"


def test_强制时没有可清的也不留空标记():
    ctx = Context([ClearOldToolResults(keep_recent=0)])
    ctx.add(Message.user("q"))
    assert ctx.maintain(measure, force=True) == []
    assert ctx.history == [Message.user("q")]


def test_强制压缩只留当前这一轮_不管保留多少token():
    """按 2 万 token 留原文的话，短对话 /compact 什么都压不掉 —— 用户明确要压，就该压。"""
    calls: list = []
    compact = CompactHistory(summarizer(calls), trigger_tokens=10**9, keep_recent_tokens=10**9)
    ctx = Context([compact])
    for n in range(3):
        ctx.add(Message.user(f"问题{n}"))
        ctx.add(Message.assistant(f"答案{n}"))

    assert ctx.maintain(measure, force=False) == []
    [event] = ctx.maintain(measure, force=True)
    assert "保留最近 1 轮" in event.description
    assert [m.content for m in calls[0]] == ["问题0", "答案0", "问题1", "答案1"]
