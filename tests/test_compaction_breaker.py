"""熔断：自动压缩连续失败 3 次就不再自动试，免得每一步都白花一次写摘要的钱。"""

from __future__ import annotations

import pytest

from data_agent.core.agent import MAX_COMPACTION_FAILURES
from data_agent.core.context import ClearOldToolResults, CompactHistory, Context, Summary
from data_agent.core.errors import CompactionFailed, ContextOverflow
from data_agent.core.events import AutoCompactionPaused, ContextEdited, ContextEditFailed
from data_agent.core.messages import LLMResponse, Message, ToolCall, Usage

from fakes import make_agent

OK = LLMResponse(text="答", stop_reason="end_turn")


class ScriptedSummarizer:
    """按剧本走：True = 写出摘要，False = 失败。剧本用完之后一直失败。"""

    def __init__(self, *script: bool) -> None:
        self.script = list(script)
        self.calls = 0

    def __call__(self, messages, prompt=None) -> Summary:
        self.calls += 1
        if self.script and self.script.pop(0):
            return Summary(f"摘要{self.calls}")
        raise CompactionFailed("写摘要的请求返回了空内容。", Usage(input=100))


def agent_with(summarize, *extra_edits):
    compact = CompactHistory(summarize, trigger_tokens=1, keep_recent_tokens=1)
    return make_agent([OK], context=Context([*extra_edits, compact]))


def failures(events) -> int:
    return sum(isinstance(e, ContextEditFailed) for e in events)


def test_连续失败三次就不再自动压缩():
    summarize = ScriptedSummarizer()
    agent, events = agent_with(summarize)
    for n in range(6):
        agent.run(f"问题{n}")

    assert summarize.calls == MAX_COMPACTION_FAILURES == 3
    assert [e.failures for e in events if isinstance(e, AutoCompactionPaused)] == [3], "只提示一次"
    assert agent.session_usage.input == 300, "失败的三次都记账"


def test_熔断之后清理照常():
    summarize = ScriptedSummarizer()
    clear = ClearOldToolResults(trigger_tokens=1, keep_recent=0, clear_at_least=0)
    agent, events = agent_with(summarize, clear)
    for n in range(4):
        agent.context.add(Message.user(f"问题{n}"))
        agent.context.add(Message(role="assistant", tool_calls=[ToolCall(f"c{n}", "echo", {})]))
        agent.context.add(Message.tool_result(f"c{n}", "| 华东 | 8100531.47 |\n" * 20))
        agent.context.add(Message.assistant(f"答案{n}"))
        agent.run(f"追问{n}")

    assert summarize.calls == 3
    cleared = [e for e in events if isinstance(e, ContextEdited) and e.kind == "ToolResultsCleared"]
    assert len(cleared) == 4, "熔断只停调模型的工序"


def test_成功一次就重新计数():
    summarize = ScriptedSummarizer(False, False, True, False, False)
    agent, events = agent_with(summarize)
    for n in range(6):
        agent.run(f"问题{n}")

    # 第一轮没有可压的；之后 5 次：失败 2 次、成功、再失败 2 次 —— 从没连续失败 3 次
    assert summarize.calls == 5
    assert not any(isinstance(e, AutoCompactionPaused) for e in events)


def test_熔断之后手动压缩照样试_成功就恢复自动压缩():
    summarize = ScriptedSummarizer(False, False, False, True, False)
    agent, events = agent_with(summarize)
    for n in range(4):
        agent.run(f"问题{n}")
    assert summarize.calls == 3

    assert agent.compact(), "强制压缩不看熔断"
    agent.run("问题4")
    assert summarize.calls == 5, "手动压成功之后，自动压缩又开始试了"
    assert failures(events) == 4


def test_超长时的强制压缩不看熔断_失败照样抛出并记账():
    summarize = ScriptedSummarizer()
    compact = CompactHistory(summarize, trigger_tokens=1, keep_recent_tokens=1)
    agent, _ = make_agent([OK, OK, OK, OK, ContextOverflow("prompt is too long")],
                          context=Context([compact]))
    for n in range(4):
        agent.run(f"问题{n}")
    assert summarize.calls == 3

    with pytest.raises(CompactionFailed):
        agent.run("问题4")
    assert summarize.calls == 4
    assert agent.session_usage.input == 400


def test_reset之后重新计数():
    summarize = ScriptedSummarizer()
    agent, _ = agent_with(summarize)
    for n in range(4):
        agent.run(f"问题{n}")
    agent.reset()
    agent.run("问题a")
    agent.run("问题b")
    assert summarize.calls == 4
