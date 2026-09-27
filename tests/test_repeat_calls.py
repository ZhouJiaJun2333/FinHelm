"""原地打转：同一轮里同一个工具、同样的参数调到第 3 次，在结果后面提醒模型换思路（不拦）。"""

from __future__ import annotations

from data_agent.core.agent import REPEAT_WARN_AT
from data_agent.core.events import ToolCallRepeated, ToolFinished
from data_agent.core.messages import LLMResponse, ToolCall
from evals.runner import Trial, digest

from fakes import make_agent


def call(text: str, id_: str = "c") -> LLMResponse:
    return LLMResponse(text="", tool_calls=[ToolCall(id_, "echo", {"text": text})], stop_reason="tool_use")


DONE = LLMResponse(text="好了", stop_reason="end_turn")


def tool_results(agent) -> list[str]:
    return [m.content for m in agent.context.render() if m.role == "tool"]


def test_同样的参数第3次起附提醒():
    agent, events = make_agent([call("x"), call("x"), call("x"), call("x"), DONE], max_steps=10)
    agent.run("q")
    results = tool_results(agent)
    assert ["提醒" in r for r in results] == [False, False, True, True]
    assert "echo 3 次" in results[2] and results[2].startswith("echo: x")
    assert [e.count for e in events if isinstance(e, ToolCallRepeated)] == [3, 4]
    # 事件里的内容就是模型看到的那份
    assert [e.content for e in events if isinstance(e, ToolFinished)] == results
    assert REPEAT_WARN_AT == 3


def test_参数不同不算重复():
    agent, events = make_agent([call("a"), call("b"), call("a"), call("b"), DONE], max_steps=10)
    agent.run("q")
    assert not any(isinstance(e, ToolCallRepeated) for e in events)


def test_参数顺序不同也算同一种调用():
    same = [LLMResponse(text="", tool_calls=[ToolCall("c", "echo", args)], stop_reason="tool_use")
            for args in ({"text": "x"}, {"text": "x"}, {"text": "x"})]
    agent, events = make_agent([*same, DONE], max_steps=10)
    agent.run("q")
    assert any(isinstance(e, ToolCallRepeated) for e in events)


def test_每轮重新数():
    agent, events = make_agent([call("x"), call("x"), DONE, call("x"), DONE], max_steps=10)
    agent.run("第一轮")
    agent.run("第二轮")
    assert not any(isinstance(e, ToolCallRepeated) for e in events)


def test_评测记下提醒了几次():
    agent, events = make_agent([call("x"), call("x"), call("x"), DONE], max_steps=10)
    t = Trial("x", 1, answer=agent.run("q"))
    digest(t, events)
    assert t.repeat_warnings == 1
